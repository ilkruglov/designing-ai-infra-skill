import math
import unittest

from infra_calc import roofline

ANCHORS = (
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter2.md:236",
)


class RooflineTest(unittest.TestCase):
    def test_book_single_request_bounds(self) -> None:
        # chapter1.md:268: «\frac{70\ \mathrm{GB}}{3350\ \mathrm{GB/s}}\approx20{,}90\ \mathrm{ms}»,
        # «\frac{140\ \mathrm{GFLOPs}}{989400\ \mathrm{GFLOP/s}}\approx0{,}1415\ \mathrm{ms}»
        self.assertEqual(round(roofline.memory_seconds(70e9, 3.35e12) * 1e3, 2), 20.90)
        self.assertEqual(
            round(roofline.compute_seconds(140e9, 989.4e12) * 1e3, 4), 0.1415
        )
        self.assertEqual(
            round(
                roofline.lower_bound_seconds(140e9, 70e9, 989.4e12, 3.35e12) * 1e3, 2
            ),
            20.90,
        )

    def test_batch_of_eight_shares_weight_read(self) -> None:
        # chapter1.md:293: «нижняя граница времени матричных вычислений составляет около
        # 1,13 мс, а нижняя граница чтения по-прежнему равна 20,90 мс»
        self.assertEqual(
            round(roofline.compute_seconds(8 * 140e9, 989.4e12) * 1e3, 2), 1.13
        )
        self.assertEqual(
            round(
                roofline.lower_bound_seconds(8 * 140e9, 70e9, 989.4e12, 3.35e12) * 1e3,
                2,
            ),
            20.90,
        )

    def test_batch_threshold_example_2_2(self) -> None:
        # chapter2.md:262: «при $H=8192$ минимальный целочисленный размер батча равен 13,
        # а при $H=2048$ — 51»
        self.assertEqual(roofline.batch_threshold(15_136_811_008, 147_456, 8192), 13)
        self.assertEqual(roofline.batch_threshold(15_136_811_008, 147_456, 2048), 51)

    def test_ridge_point_gives_batch_transition(self) -> None:
        # chapter1.md:307: B_* = b_W·Π/(2β); при $b_W=1$ «получаем $B_*\approx147{,}7$»,
        # то есть Π/β = 2·B_* при двух FLOPs на параметр
        self.assertEqual(round(roofline.ridge_point(989.4e12, 3.35e12) / 2, 1), 147.7)

    def test_compute_bound_batch_book(self) -> None:
        # chapter1.md:307: B_* = b_W·Π/(2β); «при $b_W=1$ ... получаем $B_*\\approx147{,}7$»,
        # «После примерно 148 запросов время вычислений превышает время чтения весов».
        # Модель книги: F = 2N = 140 GFLOPs на запрос, R_W = b_W·N = 70 GB, без KV
        point = roofline.compute_bound_batch(140e9, 70e9, 0, 989.4e12, 3.35e12)
        self.assertEqual(round(point, 1), 147.7)
        self.assertEqual(math.ceil(point), 148)

    def test_compute_bound_batch_with_context(self) -> None:
        # вывод вручную: B_* = (R_W/β) / (F/Π − R_KV/β); F = 16 GFLOPs, R_W = 15 GB,
        # R_KV = 1 MB, Π = 1 PFLOP/s, β = 3 TB/s — 5e-3 / (1.6e-5 − 3.333e-7) ≈ 319.15
        point = roofline.compute_bound_batch(16e9, 15e9, 1e6, 1e15, 3e12)
        self.assertAlmostEqual(point, 5e-3 / (1.6e-5 - 1e6 / 3e12))
        self.assertEqual(math.ceil(point), 320)

    def test_compute_bound_batch_unreachable(self) -> None:
        # Qwen3-8B на H100 при контексте 2048 (chapter1.md:459-461: 16.34 GFLOPs,
        # 144 KiB × 2048): на запрос F/Π ≈ 16.5 μs меньше R_KV/β ≈ 90.1 μs — шаг
        # decode не становится вычислительно ограниченным ни при каком батче
        self.assertIsNone(
            roofline.compute_bound_batch(
                16_344_154_112, 15_136_811_008, 147_456 * 2048, 989.4e12, 3.35e12
            )
        )

    def test_compute_bound_batch_rejects_invalid_inputs(self) -> None:
        for args in (
            (-1, 70e9, 0, 989.4e12, 3.35e12),
            (140e9, -1, 0, 989.4e12, 3.35e12),
            (140e9, 70e9, -1, 989.4e12, 3.35e12),
            (140e9, 70e9, 0, 0, 3.35e12),
            (140e9, 70e9, 0, 989.4e12, math.nan),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                roofline.compute_bound_batch(*args)

    def test_arithmetic_intensity(self) -> None:
        # Арифметическое тождество на числах chapter1.md:265 (F = 140 GFLOPs, R_W = 70 GB):
        # книга не приводит отношение, 140/70 = 2 FLOP/byte
        self.assertEqual(roofline.arithmetic_intensity(140e9, 70e9), 2.0)

    def test_non_positive_rates_are_rejected(self) -> None:
        # нулевой или отрицательный пик/пропускная способность дают бессмысленную границу
        for peak, bw in (
            (0, 3.35e12),
            (-1.0, 3.35e12),
            (989.4e12, 0),
            (989.4e12, -1.0),
        ):
            with self.subTest(peak=peak, bandwidth=bw), self.assertRaises(ValueError):
                roofline.lower_bound_seconds(140e9, 70e9, peak, bw)
            with self.subTest(ridge=(peak, bw)), self.assertRaises(ValueError):
                roofline.ridge_point(peak, bw)

    def test_negative_work_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roofline.compute_seconds(-1.0, 989.4e12)
        with self.assertRaises(ValueError):
            roofline.memory_seconds(-1.0, 3.35e12)
        with self.assertRaises(ValueError):
            roofline.arithmetic_intensity(140e9, 0)

    def test_batch_threshold_rejects_empty_context(self) -> None:
        for args in (
            (15_136_811_008, 0, 8192),
            (15_136_811_008, 147_456, 0),
            (-1, 147_456, 8192),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                roofline.batch_threshold(*args)

    def test_nan_and_fractional_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roofline.compute_seconds(math.nan, 989.4e12)
        with self.assertRaises(ValueError):
            roofline.memory_seconds(70e9, math.nan)
        with self.assertRaises(ValueError):
            roofline.batch_threshold(15_136_811_008, 147_456, 2048.5)


if __name__ == "__main__":
    unittest.main()
