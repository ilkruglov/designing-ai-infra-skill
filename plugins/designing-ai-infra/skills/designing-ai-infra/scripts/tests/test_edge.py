import math
import unittest

from infra_calc import edge, units

ANCHORS = (
    "references/source-book/chapter12.md:11",
    "references/source-book/chapter12.md:13",
)


class EdgeTest(unittest.TestCase):
    def test_upload_dominates(self) -> None:
        # chapter12.md:33-37: 30 MB вверх по 20 Mbit/s, R = 0.1 s, T_c = 0.3 s, 5 MB вниз по 100 Mbit/s,
        # «T_{\mathrm{serial}}=\frac{30\times8}{20}+0.1+0.3+\frac{5\times8}{100} =12.8\ \mathrm{s}»
        up = units.megabits_per_second_to_bytes(20)
        down = units.megabits_per_second_to_bytes(100)
        self.assertAlmostEqual(edge.serial_seconds(30e6, up, 0.1, 0.3, 5e6, down), 12.8)

    def test_faster_uplink(self) -> None:
        # chapter12.md:42: «Если повысить скорость восходящего канала до 100 Mbit/s, загрузка займёт
        # 2,4 секунды, а вся задача — только 3,2 секунды»
        link = units.megabits_per_second_to_bytes(100)
        self.assertAlmostEqual(
            edge.serial_seconds(30e6, link, 0.1, 0.3, 5e6, link), 3.2
        )

    def test_compression_gain(self) -> None:
        # chapter12.md:42: сжатие вдвое (30 → 15 MB), «кодирование и декодирование вместе занимают
        # 0,15 секунды. Тогда отправка займёт только 6 секунд»; chapter12.md:47:
        # «\Delta T=\frac{S_u-S'_u}{B_u}-T_e», то есть 6 − 0.15 = 5.85 s
        up = units.megabits_per_second_to_bytes(20)
        self.assertAlmostEqual(
            edge.compression_gain_seconds(30e6, 15e6, up, 0.15), 5.85
        )

    def test_compression_gain_vanishes_on_fast_uplink(self) -> None:
        # chapter12.md:50: «при 100 Mbit/s — 1,2 секунды; при 800 Mbit/s — лишь 0,15 секунды,
        # и выигрыш становится нулевым»
        fast = units.megabits_per_second_to_bytes(100)
        self.assertAlmostEqual(
            edge.compression_gain_seconds(30e6, 15e6, fast, 0.15), 1.2 - 0.15
        )
        faster = units.megabits_per_second_to_bytes(800)
        self.assertAlmostEqual(
            edge.compression_gain_seconds(30e6, 15e6, faster, 0.15), 0
        )

    def test_serial_rejects_invalid_inputs(self) -> None:
        # нулевая или бесконечная полоса, отрицательные или NaN слагаемые дают ложное время
        for args in (
            (-30e6, 2.5e6, 0.1, 0.3, 5e6, 12.5e6),
            (30e6, 0, 0.1, 0.3, 5e6, 12.5e6),
            (30e6, math.inf, 0.1, 0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, -0.1, 0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, math.nan, 0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, -0.3, 5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, 0.3, -5e6, 12.5e6),
            (30e6, 2.5e6, 0.1, 0.3, 5e6, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                edge.serial_seconds(*args)

    def test_compression_gain_rejects_invalid_inputs(self) -> None:
        for args in (
            (-30e6, 10e6, 2.5e6, 0.5),
            (math.inf, 10e6, 2.5e6, 0.5),
            (30e6, -10e6, 2.5e6, 0.5),
            (30e6, 10e6, 0, 0.5),
            (30e6, 10e6, 2.5e6, -0.5),
            (30e6, 10e6, 2.5e6, math.nan),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                edge.compression_gain_seconds(*args)


if __name__ == "__main__":
    unittest.main()
