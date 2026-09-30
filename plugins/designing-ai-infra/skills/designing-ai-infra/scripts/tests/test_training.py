import json
import math
import unittest
from pathlib import Path

from infra_calc import training

ANCHORS = (
    "references/source-book/chapter3.md:384",
    "references/source-book/chapter7.md:938",
    "references/source-book/chapter10.md:151",
    "references/source-book/chapter10.md:328",
    "references/source-book/chapter10.md:559",
    "references/source-book/chapter10.md:606",
    "calculations/results/checkpoint-interval-book.json#sha256=e986d06ba47b7d0c13b54a99cc2abde1744d674eb470b26d9cf00e417ac0c6ad",
    "calculations/results/checkpoint-interval-common-shock.json#sha256=599b6d27d8ff541886c69eb066952d49e43e1e025b8eb006c5235adabebf89da",
    "calculations/results/training-pipeline-interleaved-m8.json#sha256=773f52ffe9134ea65961825a000bba925cd1734beb6937b2488ef400710703a3",
    "calculations/results/training-pipeline-interleaved-m16.json#sha256=978e1644d72dbd4a059e1afeb3fc8d3a01747fba8a94bb7c4b96bd7ef930cae4",
    "calculations/results/training-state-book.json#sha256=0cdea3fbd70fac7846e6655282e76624d3f5b14fd0e2627f5393fe178b6f3cd5",
    "calculations/results/training-state-fp32-gradient.json#sha256=f5924f46eece9b76bb7eebcb04ed0f08eb262b46397f65715c0f40a3c5f791cc",
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


class InterleavedPipelineTest(unittest.TestCase):
    def test_bubble_ratio_book_and_author(self) -> None:
        # chapter10.md:365: доля пузырей чередующегося 1F1B «\\frac{1}{v}\\cdot\\frac{p-1}{m}»;
        # training-pipeline-interleaved-m8.json: bubble_bound_1f1b_fraction 0.375,
        # bubble_bound_interleaved_fraction 0.1875 (p = 4, m = 8, v = 2)
        self.assertEqual(training.pipeline_bubble_ratio(4, 8), 0.375)
        self.assertEqual(training.pipeline_bubble_ratio(4, 8, 2), 0.1875)
        # chapter10.md:406: «при 16 micro-batch доля пузырей 1F1B снижается с 37.5% до 18.8%»;
        # training-pipeline-interleaved-m16.json: bubble_bound_interleaved_fraction 0.09375
        self.assertEqual(round(training.pipeline_bubble_ratio(4, 16) * 100, 1), 18.8)
        self.assertEqual(training.pipeline_bubble_ratio(4, 16, 2), 0.09375)

    def test_bubble_seconds_author(self) -> None:
        # training-pipeline-interleaved-m8.json: bubble_bound_1f1b_seconds 0.09,
        # bubble_bound_interleaved_seconds 0.045 при t_f = 10 ms, t_b = 20 ms
        self.assertAlmostEqual(training.pipeline_bubble_seconds(4, 10e-3, 20e-3), 0.09)
        self.assertAlmostEqual(
            training.pipeline_bubble_seconds(4, 10e-3, 20e-3, 2), 0.045
        )

    def test_step_seconds(self) -> None:
        # chapter10.md:338: fill–drain «(8+4-1)\\times30=330» ms без передач
        self.assertAlmostEqual(
            training.pipeline_step_seconds(4, 8, 10e-3, 20e-3), 330e-3
        )
        # вывод вручную: m(t_f + t_b) + (p − 1)(t_f + t_b)/v = 8·30 + 3·30/2 = 285 ms
        self.assertAlmostEqual(
            training.pipeline_step_seconds(4, 8, 10e-3, 20e-3, 2), 285e-3
        )

    def test_interleaved_utilization(self) -> None:
        # вывод вручную: u = m/(m + (p − 1)/v); p = 4, m = 8, v = 2 — 8/9.5;
        # p = 8, m = 32, v = 4 — 32/33.75
        self.assertAlmostEqual(training.pipeline_utilization(4, 8, 2), 8 / 9.5)
        self.assertAlmostEqual(training.pipeline_utilization(8, 32, 4), 32 / 33.75)
        self.assertAlmostEqual(training.pipeline_bubble_ratio(8, 32, 4), 7 / 128)

    def test_interleaved_needs_microbatches_divisible_by_stages(self) -> None:
        # порядок Megatron-LM, которому следует автор (training_pipeline_schedule.py):
        # «interleaved_1f1b requires microbatches divisible by the pipeline depth»
        for function in (training.pipeline_utilization, training.pipeline_bubble_ratio):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError) as caught:
                    function(4, 6, 2)
                self.assertIn("кратн", str(caught.exception))
        with self.assertRaises(ValueError):
            training.pipeline_step_seconds(4, 6, 10e-3, 20e-3, 2)

    def test_rejects_invalid_virtual_stages(self) -> None:
        for virtual in (0, -1, 1.5, True):
            with self.subTest(virtual=virtual), self.assertRaises(ValueError):
                training.pipeline_bubble_ratio(4, 8, virtual)
            with self.subTest(virtual=virtual), self.assertRaises(ValueError):
                training.pipeline_bubble_seconds(4, 10e-3, 20e-3, virtual)


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


class CommonShockTest(unittest.TestCase):
    SAVE = (
        114_670_295_040 / 7e9
    )  # checkpoint-interval-book.json: 14 байт × P при 7 GB/s

    def test_author_common_job_rate(self) -> None:
        # checkpoint-interval-common-shock.json: devices 1024, device_mtbf_seconds
        # 29122560, common_job_mtbf_seconds 86400; job_failure_rate_exact_per_second
        # «319/6825600»: общий удар прибавляется один раз, а не на каждое устройство
        rate = training.job_failure_rate(1024, 29_122_560, 86_400)
        self.assertAlmostEqual(rate, 319 / 6_825_600, places=15)
        # first_order_optimal_useful_interval_seconds 837.2719042643316,
        # poisson_optimal_useful_interval_seconds 826.3867220685526
        self.assertAlmostEqual(
            training.checkpoint_optimal_interval(self.SAVE, rate),
            837.2719042643316,
            places=6,
        )
        self.assertAlmostEqual(
            training.checkpoint_poisson_optimal_interval(self.SAVE, rate),
            826.3867220685526,
            places=6,
        )
        # checkpoint_interval_rows: first_order_loss 0.046931494800562586 при 600 s
        self.assertAlmostEqual(
            training.checkpoint_first_order_loss(600, self.SAVE, rate, 120),
            0.046931494800562586,
            places=12,
        )

    def test_loss_spike_book(self) -> None:
        # chapter10.md:619-620: λ = λ_hw + λ_spike; chapter10.md:623: 48 ускорителей,
        # MTBF около 337 дней, всплеск раз в семь дней: «При том же τ=600 s
        # дополнительные затраты увеличиваются с 2,80% до 2,87%, а оптимальный период
        # сохранения сокращается примерно с 4 458 s до 3 150 s. При интервале 1800 s
        # ... возрастают с 1,08% примерно до 1,25%»
        hw = training.job_failure_rate(48, 29_122_560)
        both = training.job_failure_rate(48, 29_122_560, 7 * 86_400)
        loss = training.checkpoint_first_order_loss
        self.assertEqual(round(loss(600, self.SAVE, hw, 120) * 100, 2), 2.80)
        self.assertEqual(round(loss(600, self.SAVE, both, 120) * 100, 2), 2.87)
        self.assertEqual(
            round(training.checkpoint_optimal_interval(self.SAVE, hw)), 4458
        )
        self.assertEqual(
            round(training.checkpoint_optimal_interval(self.SAVE, both), -1), 3150
        )
        self.assertEqual(round(loss(1800, self.SAVE, hw, 120) * 100, 2), 1.08)
        self.assertEqual(round(loss(1800, self.SAVE, both, 120) * 100, 2), 1.25)

    def test_rate_rejects_invalid_inputs(self) -> None:
        for args in ((0, 1e6), (8, 0), (8, -1e6), (8, 1e6, 0), (8, 1e6, math.nan)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                training.job_failure_rate(*args)


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

    def test_components_match_author(self) -> None:
        # training-state-book.json: training_state_stages[*].components per_rank_bytes
        # (weights_bf16; gradients; master_weights_fp32 + adam_m_fp32 + adam_v_fp32),
        # 8 участников, градиенты BF16
        optimizer_full = 3 * 32_762_941_440
        optimizer_shard = 3 * 4_095_367_680
        expected = {
            0: (16_381_470_720, 16_381_470_720, optimizer_full),
            1: (16_381_470_720, 16_381_470_720, optimizer_shard),
            2: (16_381_470_720, 2_047_683_840, optimizer_shard),
            3: (2_047_683_840, 2_047_683_840, optimizer_shard),
        }
        for stage, parts in expected.items():
            with self.subTest(stage=stage):
                self.assertEqual(
                    training.sharded_state_components(8_190_735_360, 8, stage), parts
                )

    def test_components_with_fp32_gradients(self) -> None:
        # training-state-fp32-gradient.json: gradients 32 762 941 440 до stage 2,
        # 4 095 367 680 со stage 2; persistent_bytes_per_rank stage 0 — 147 433 236 480
        # (chapter3.md:414: 18 байт на параметр)
        parts = training.sharded_state_components(8_190_735_360, 8, 0, grad_bytes=4)
        self.assertEqual(parts[1], 32_762_941_440)
        self.assertEqual(sum(parts), 147_433_236_480)
        parts = training.sharded_state_components(8_190_735_360, 8, 3, grad_bytes=4)
        self.assertEqual(parts, (2_047_683_840, 4_095_367_680, 12_286_103_040))

    def test_components_hand_derived(self) -> None:
        # вывод вручную: P = 10^9, d = 4, stage 2 — веса 2P, градиенты 2P/4, оптимизатор 12P/4
        self.assertEqual(
            training.sharded_state_components(10**9, 4, 2),
            (2 * 10**9, 5 * 10**8, 3 * 10**9),
        )
        self.assertEqual(
            sum(training.sharded_state_components(10**9, 4, 2)),
            training.sharded_state_bytes_per_device(10**9, 4, 2),
        )

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
