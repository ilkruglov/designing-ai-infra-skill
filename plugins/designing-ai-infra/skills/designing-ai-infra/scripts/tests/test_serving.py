import math
import unittest

from infra_calc import serving

ANCHORS = (
    "references/source-book/chapter8.md:52",
    "references/source-book/chapter8.md:268",
    "references/source-book/chapter1.md:189",
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter1.md:450",
)
GIB = 2**30
MIB = 2**20


class ServingTest(unittest.TestCase):
    def test_concurrency_from_memory_inequality(self) -> None:
        # chapter8.md:75-76: «Короткий диалог | 288 MiB | 324 MiB | 37»,
        # «Длинный контекст | 1152 MiB | 1188 MiB | 10»; chapter8.md:78:
        # «$\lfloor12288/324\rfloor$ и $\lfloor12288/1188\rfloor$» — 12 GiB под KV
        self.assertEqual(serving.max_concurrent_requests(12 * GIB, 0, 324 * MIB), 37)
        self.assertEqual(serving.max_concurrent_requests(12 * GIB, 0, 1188 * MIB), 10)
        # chapter8.md:78: без резерва на генерацию «в 12 GiB поместятся 42» (288 MiB)
        self.assertEqual(serving.max_concurrent_requests(12 * GIB, 0, 288 * MIB), 42)

    def test_shared_prefix_is_stored_once(self) -> None:
        # chapter8.md:292: «16 длинных запросов совместно используют первые 6144 входных
        # токена, KV которых занимает в общей сложности 864 MiB. Собственная часть
        # каждого запроса содержит 8192−6144+255=2303 токена ... 2304 токена, то есть
        # 324 MiB»; chapter8.md:298: «$b$ однотипных длинных запросов занимают
        # $864+324b$ MiB, поэтому предел ёмкости увеличивается с 10 независимых запросов
        # до 35 запросов с совместным использованием»
        self.assertEqual(
            serving.max_concurrent_requests(
                12 * GIB, 0, 324 * MIB, shared_bytes=864 * MIB
            ),
            35,
        )
        self.assertEqual(serving.max_concurrent_requests(12 * GIB, 0, 1188 * MIB), 10)
        # chapter8.md:295: M_KV = 864 + 16 × 324 = 6048 MiB для 16 запросов
        self.assertEqual(864 + 16 * 324, 6048)
        # общий префикс сам не помещается — ноль запросов, а не отрицательное число
        self.assertEqual(
            serving.max_concurrent_requests(
                800 * MIB, 0, 324 * MIB, shared_bytes=864 * MIB
            ),
            0,
        )
        for bad in (-1.0, math.nan):
            with self.subTest(shared=bad), self.assertRaises(ValueError):
                serving.max_concurrent_requests(
                    12 * GIB, 0, 324 * MIB, shared_bytes=bad
                )

    def test_weights_do_not_fit(self) -> None:
        # chapter1.md:201: «Весам объёмом 141.11 GB недостаточно номинальных 80 GB
        # видеопамяти одной H100 SXM» — ноль запросов, а не отрицательное число
        self.assertEqual(serving.max_concurrent_requests(80e9, 141.11e9, 1e9), 0)

    def test_tpot_reads_shared_prefix_once(self) -> None:
        # вариант B главы 8.6.3 (chapter8.md:666): 16 запросов, общий префикс 6144 токена;
        # вывод вручную: префикс один раз на batch —
        # (15 136 811 008 + 6144·147 456 + 16·2048·147 456) / 1.792e12 ≈ 11.65 ms
        kv = 147_456
        step = serving.tpot_lower_bound_seconds(
            16, 19_968_032_768, 15_136_811_008, 2048 * kv, 503.8e12, 1.792e12,
            shared_read_bytes=6144 * kv,
        )  # fmt: skip
        self.assertAlmostEqual(step * 1e3, 11.648782857, places=6)
        for bad in (-1.0, math.nan):
            with self.subTest(shared=bad), self.assertRaises(ValueError):
                serving.tpot_lower_bound_seconds(
                    1, 0, 0, 0, 1e12, 1e12, shared_read_bytes=bad
                )

    def test_tpot_bound_batch_eight(self) -> None:
        # chapter1.md:268: чтение весов «\approx20{,}90\ \mathrm{ms}»;
        # chapter1.md:293: восемь запросов делят одно чтение, граница «по-прежнему равна 20,90 мс»
        step = serving.tpot_lower_bound_seconds(8, 140e9, 70e9, 0, 989.4e12, 3.35e12)
        self.assertEqual(round(step * 1e3, 2), 20.90)
        # chapter1.md:295: «примерно с 47,9 токена/с в модели одного запроса до 383 токенов/с»
        self.assertEqual(round(serving.tokens_per_second(8, step)), 383)
        single = serving.tpot_lower_bound_seconds(1, 140e9, 70e9, 0, 989.4e12, 3.35e12)
        self.assertEqual(round(serving.tokens_per_second(1, single), 1), 47.9)

    def test_rtx_pro_6000_bounds(self) -> None:
        # chapter1.md:464: при 503.8 TFLOP/s «матричные операции prefill занимают около 58.9 ms»;
        # при 1.792 TB/s чтение весов 15.14 GB и KV 0.302 GB за шаг decode «занимает около 8.62 ms»
        ttft = serving.ttft_lower_bound_seconds(29.69e12, 16.38e9, 503.8e12, 1.792e12)
        self.assertEqual(round(ttft * 1e3, 1), 58.9)
        tpot = serving.tpot_lower_bound_seconds(
            1, 16.34e9, 15.14e9, 0.302e9, 503.8e12, 1.792e12
        )
        self.assertEqual(round(tpot * 1e3, 2), 8.62)

    def test_cost_per_million_tokens(self) -> None:
        # Арифметическое тождество, не число книги: 3.6 $/ч = 0.001 $/с,
        # при 1000 токенов/с миллион токенов обходится в 1 $
        self.assertEqual(serving.cost_per_million_tokens(3.6, 1000), 1.0)

    def test_concurrency_rejects_invalid_sizes(self) -> None:
        # нулевой KV на запрос дал бы деление на ноль, отрицательный — отрицательную ёмкость
        for kwargs in (
            {"kv_bytes_per_request": 0},
            {"kv_bytes_per_request": -1.0},
            {"kv_bytes_per_request": 1e9, "reserve_bytes": -1.0},
            {"kv_bytes_per_request": 1e9, "weight_bytes": -1.0},
            {"kv_bytes_per_request": 1e9, "memory_bytes": -1.0},
        ):
            args = {"memory_bytes": 96e9, "weight_bytes": 16e9, **kwargs}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                serving.max_concurrent_requests(**args)

    def test_tpot_rejects_empty_batch(self) -> None:
        for batch in (0, -1):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                serving.tpot_lower_bound_seconds(
                    batch, 140e9, 70e9, 0, 989.4e12, 3.35e12
                )
        with self.assertRaises(ValueError):
            serving.tpot_lower_bound_seconds(8, 140e9, 70e9, -1.0, 989.4e12, 3.35e12)

    def test_throughput_rejects_non_positive_step(self) -> None:
        for args in ((8, 0), (8, -0.02), (-1, 0.02)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                serving.tokens_per_second(*args)

    def test_cost_rejects_invalid_inputs(self) -> None:
        for args in ((3.6, 0), (3.6, -1000), (-3.6, 1000)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                serving.cost_per_million_tokens(*args)

    def test_fractional_batch_and_nan_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            serving.tpot_lower_bound_seconds(2.5, 140e9, 70e9, 0, 989.4e12, 3.35e12)
        with self.assertRaises(ValueError):
            serving.tokens_per_second(2.5, 0.02)
        with self.assertRaises(ValueError):
            serving.tokens_per_second(8, math.nan)
        with self.assertRaises(ValueError):
            serving.max_concurrent_requests(96e9, 16e9, math.nan)


if __name__ == "__main__":
    unittest.main()
