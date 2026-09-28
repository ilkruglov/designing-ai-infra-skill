import math
import unittest

from infra_calc import roofline

ANCHORS = (
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter2.md:236",
)


class RooflineTest(unittest.TestCase):
    def test_book_single_request_bounds(self) -> None:
        # chapter1.md:263: 70 GB / 3350 GB/s ≈ 20.90 ms; 140 GFLOPs / 989.4 TFLOPs ≈ 0.1415 ms
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
        # chapter1.md:263: восемь запросов делят одно чтение 70 GB — граница прежняя
        self.assertEqual(
            round(
                roofline.lower_bound_seconds(8 * 140e9, 70e9, 989.4e12, 3.35e12) * 1e3,
                2,
            ),
            20.90,
        )

    def test_batch_threshold_example_2_2(self) -> None:
        # chapter2.md:236, пример 2-2: 13 при H=8192, 51 при H=2048
        self.assertEqual(roofline.batch_threshold(15_136_811_008, 147_456, 8192), 13)
        self.assertEqual(roofline.batch_threshold(15_136_811_008, 147_456, 2048), 51)

    def test_intensity_and_ridge(self) -> None:
        self.assertEqual(roofline.arithmetic_intensity(140e9, 70e9), 2.0)
        self.assertAlmostEqual(
            roofline.ridge_point(989.4e12, 3.35e12), 295.343, places=3
        )

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
