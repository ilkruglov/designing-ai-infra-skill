import json
import math
import unittest
from pathlib import Path

from infra_calc import training

ANCHORS = (
    "references/source-book/chapter3.md:384",
    "references/source-book/chapter7.md:938",
    "references/source-book/chapter10.md:151",
    "references/source-book/chapter10.md:322",
    "references/source-book/chapter10.md:553",
    "calculations/results/checkpoint-interval-book.json#sha256=e986d06ba47b7d0c13b54a99cc2abde1744d674eb470b26d9cf00e417ac0c6ad",
)
AUTHOR = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "author_results.json").read_text()
)["checkpoint-interval-book"]


class PipelineTest(unittest.TestCase):
    def test_bubble(self) -> None:
        # chapter10.md:336: «При $p=4,m=8$ получаем $u=8/11\approx72.7\%$.»
        self.assertEqual(round(training.pipeline_utilization(4, 8) * 100, 1), 72.7)
        # chapter10.md:338: «Без учёта передачи данных планирование fill–drain занимает
        # $(8+4-1)\times30=330$ ms»; прямой проход 10 ms, обратный 20 ms
        self.assertEqual(training.fill_drain_seconds(4, 8, 10e-3, 20e-3), 11 * 30e-3)

    def test_rejects_empty_pipeline(self) -> None:
        # ноль стадий или микропакетов дал бы u > 1 или деление на ноль
        for stages, microbatches in ((0, 8), (4, 0), (-1, 8)):
            with self.subTest(stages=stages, microbatches=microbatches):
                with self.assertRaises(ValueError):
                    training.pipeline_utilization(stages, microbatches)
                with self.assertRaises(ValueError):
                    training.fill_drain_seconds(stages, microbatches, 10e-3, 20e-3)

    def test_rejects_fractional_and_nan_counts(self) -> None:
        # pipeline_utilization(nan, 4) раньше возвращал nan, дробные стадии — правдоподобное число
        for stages, microbatches in (
            (math.nan, 4),
            (4, math.nan),
            (4.0, 8),
            (4, 8.5),
            (True, 8),
        ):
            with self.subTest(stages=stages, microbatches=microbatches):
                with self.assertRaises(ValueError):
                    training.pipeline_utilization(stages, microbatches)
                with self.assertRaises(ValueError):
                    training.fill_drain_seconds(stages, microbatches, 10e-3, 20e-3)

    def test_fill_drain_rejects_invalid_times(self) -> None:
        for forward, backward in (
            (-10e-3, 20e-3),
            (10e-3, -20e-3),
            (math.nan, 20e-3),
            (10e-3, math.inf),
        ):
            with (
                self.subTest(forward=forward, backward=backward),
                self.assertRaises(ValueError),
            ):
                training.fill_drain_seconds(4, 8, forward, backward)


class CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.save = (
            AUTHOR["checkpoint_payload_bytes"]
            / AUTHOR["save_bandwidth_bytes_per_second"]
        )
        self.rate = AUTHOR["devices"] / AUTHOR["device_mtbf_seconds"]

    def test_payload_is_14_bytes_per_parameter(self) -> None:
        # checkpoint-interval-book.json: 14 байт на параметр, checkpoint_payload_bytes
        self.assertEqual(
            training.state_bytes(AUTHOR["parameters"], 14),
            AUTHOR["checkpoint_payload_bytes"],
        )

    def test_book_job_mtbf(self) -> None:
        # chapter10.md:570: «задание на 1024 ускорителях прерывалось в среднем раз в 7,9 часа»
        self.assertEqual(round(1 / self.rate / 3600, 1), 7.9)

    def test_first_order_optimum_and_loss(self) -> None:
        # checkpoint-interval-book.json: first_order_optimal_useful_interval_seconds и
        # first_order_loss для интервала 60 s; chapter10.md:570: «около 965 s»
        self.assertAlmostEqual(
            training.checkpoint_optimal_interval(self.save, self.rate),
            AUTHOR["first_order_optimal_useful_interval_seconds"],
            places=6,
        )
        self.assertAlmostEqual(
            training.checkpoint_first_order_loss(
                60, self.save, self.rate, AUTHOR["recovery_seconds"]
            ),
            AUTHOR["first_order_loss_at_60"],
            places=12,
        )

    def test_poisson_optimum(self) -> None:
        # checkpoint-interval-book.json: poisson_optimal_useful_interval_seconds
        self.assertAlmostEqual(
            training.checkpoint_poisson_optimal_interval(self.save, self.rate),
            AUTHOR["poisson_optimal_useful_interval_seconds"],
            places=6,
        )

    def test_optima_reject_invalid_inputs(self) -> None:
        # без сбоев оптимум бесконечен, без стоимости сохранения — ноль
        for save, rate in (
            (self.save, 0),
            (self.save, -self.rate),
            (self.save, math.nan),
            (self.save, math.inf),
            (0, self.rate),
            (-1, self.rate),
            (math.inf, self.rate),
        ):
            with self.subTest(save=save, rate=rate):
                with self.assertRaises(ValueError):
                    training.checkpoint_optimal_interval(save, rate)
                with self.assertRaises(ValueError):
                    training.checkpoint_poisson_optimal_interval(save, rate)

    def test_loss_rejects_invalid_inputs(self) -> None:
        for args in (
            (0, self.save, self.rate, 120),
            (-60, self.save, self.rate, 120),
            (math.inf, self.save, self.rate, 120),
            (60, -self.save, self.rate, 120),
            (60, self.save, -self.rate, 120),
            (60, self.save, math.nan, 120),
            (60, self.save, self.rate, -120),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                training.checkpoint_first_order_loss(*args)


class ShardingTest(unittest.TestCase):
    def test_full_training_state_18_bytes(self) -> None:
        # chapter3.md:414: «на каждый параметр требуется $2+4+4+4+4=18$ байт, всего 147,433 GB»
        self.assertEqual(
            round(training.state_bytes(8_190_735_360, 18) / 1e9, 2), 147.43
        )

    def test_zero_stages_on_eight_gpus(self) -> None:
        # chapter10.md:190-193: таблица «Обычный DP | 122.1», «ZeRO-1 | 42.0»,
        # «ZeRO-2 | 28.6», «ZeRO-3 | 15.3» GiB на GPU; градиенты BF16, 16 байт на параметр
        gib = [
            round(
                training.sharded_state_bytes_per_device(8_190_735_360, 8, stage)
                / 2**30,
                1,
            )
            for stage in range(4)
        ]
        self.assertEqual(gib, [122.1, 42.0, 28.6, 15.3])

    def test_unknown_stage(self) -> None:
        for stage in (4, -1, 1.0, True, math.nan):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                training.sharded_state_bytes_per_device(1, 8, stage)

    def test_state_bytes_rejects_invalid_inputs(self) -> None:
        for args in ((-1, 14), (1.5, 14), (1, -14), (1, math.nan), (1, math.inf)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                training.state_bytes(*args)

    def test_sharding_rejects_invalid_inputs(self) -> None:
        # ноль или дробь устройств — деление на ноль или ложная доля, отрицательные байты — отрицательная память
        for args, kwargs in (
            ((8_190_735_360, 0, 3), {}),
            ((8_190_735_360, 8.0, 3), {}),
            ((8_190_735_360, math.nan, 3), {}),
            ((-1, 8, 0), {}),
            ((8_190_735_360, 8, 0), {"weight_bytes": -2}),
            ((8_190_735_360, 8, 0), {"grad_bytes": -2}),
            ((8_190_735_360, 8, 0), {"optimizer_bytes": -12}),
            ((8_190_735_360, 8, 0), {"optimizer_bytes": math.inf}),
        ):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                training.sharded_state_bytes_per_device(*args, **kwargs)


class DurationTest(unittest.TestCase):
    def test_training_seconds(self) -> None:
        # chapter7.md:942: Qwen3-32B, «число параметров $P=32\times10^9$», 1024 ускорителя,
        # «За одно обновление всегда обрабатывается $2^{20}$ токенов»;
        # chapter7.md:970: «41% от пиковой производительности H100 SXM для плотных вычислений
        # BF16, составляющей 989.4 TFLOP/s», «$6P\times2^{20}/(1024\times405.7\ \mathrm{TFLOP/s})\approx0.485$ s»
        seconds = training.training_seconds(6 * 32e9 * 2**20, 1024, 989.4e12, 0.41)
        self.assertEqual(round(seconds, 3), 0.485)

    def test_training_seconds_rejects_invalid_inputs(self) -> None:
        # MFU вне (0, 1], ноль или дробь устройств и бесконечный пик дают ложный срок
        for args in (
            (431.368e12, 0, 989.4e12, 0.4),
            (431.368e12, 1.0, 989.4e12, 0.4),
            (431.368e12, 1, 0, 0.4),
            (431.368e12, 1, math.inf, 0.4),
            (431.368e12, 1, 989.4e12, 0),
            (431.368e12, 1, 989.4e12, 1.1),
            (431.368e12, 1, 989.4e12, math.nan),
            (-431.368e12, 1, 989.4e12, 0.4),
            (math.inf, 1, 989.4e12, 0.4),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                training.training_seconds(*args)


if __name__ == "__main__":
    unittest.main()
