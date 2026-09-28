import unittest

from infra_calc import edge, units

ANCHORS = ("references/source-book/chapter12.md:11",)


class EdgeTest(unittest.TestCase):
    def test_upload_dominates(self) -> None:
        # chapter12.md:11: 30 MB вверх по 20 Mbit/s, R = 0.1 s, T_c = 0.3 s, 5 MB вниз по 100 Mbit/s
        up = units.megabits_per_second_to_bytes(20)
        down = units.megabits_per_second_to_bytes(100)
        self.assertAlmostEqual(edge.serial_seconds(30e6, up, 0.1, 0.3, 5e6, down), 12.8)

    def test_compression_gain(self) -> None:
        up = units.megabits_per_second_to_bytes(20)
        self.assertAlmostEqual(edge.compression_gain_seconds(30e6, 10e6, up, 0.5), 7.5)

    def test_serial_rejects_invalid_inputs(self) -> None:
        # нулевая полоса — бесконечное время, отрицательные слагаемые — заниженное
        for args in (
            (-30e6, 2.5e6, 0.1, 0.3, 5e6, 12.5e6),
            (30e6, 0, 0.1, 0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, -0.1, 0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, -0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, 0.3, -5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, 0.3, 5e6, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                edge.serial_seconds(*args)

    def test_compression_gain_rejects_invalid_inputs(self) -> None:
        for args in (
            (-30e6, 10e6, 2.5e6, 0.5),
            (30e6, -10e6, 2.5e6, 0.5),
            (30e6, 10e6, 0, 0.5),
            (30e6, 10e6, 2.5e6, -0.5),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                edge.compression_gain_seconds(*args)


if __name__ == "__main__":
    unittest.main()
