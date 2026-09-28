import unittest

from infra_calc import units

ANCHORS = (
    "references/source-book/chapter1.md:189",
    "references/source-book/chapter8.md:52",
    "references/source-book/chapter12.md:11",
)


class UnitsTest(unittest.TestCase):
    def test_weight_bytes_in_decimal_and_binary_units(self):
        # chapter8.md:52: веса Qwen3-8B BF16 — 16.38 GB, около 15.3 GiB
        weights = 16_381_470_720
        self.assertEqual(round(units.to_unit(weights, "GB"), 2), 16.38)
        self.assertEqual(round(units.to_unit(weights, "GiB"), 2), 15.26)

    def test_kv_per_token_and_per_context(self):
        # chapter8.md:52: 144 KiB на токен, 288 MiB для 2048, 1152 MiB для 8192
        self.assertEqual(units.to_unit(147_456, "KiB"), 144)
        self.assertEqual(units.to_unit(147_456 * 2048, "MiB"), 288)
        self.assertEqual(units.to_unit(147_456 * 8192, "MiB"), 1152)

    def test_parse_bytes_requires_explicit_unit(self):
        self.assertEqual(units.parse_bytes("144KiB"), 147_456)
        self.assertEqual(units.parse_bytes("16.38 GB"), 16.38e9)
        for ambiguous in ("16G", "16", "16 gb", "16 Gb"):
            with self.assertRaises(ValueError):
                units.parse_bytes(ambiguous)

    def test_dtype_bytes(self):
        self.assertEqual(units.dtype_bytes("bf16"), 2)
        self.assertEqual(units.dtype_bytes("FP4"), 0.5)
        with self.assertRaises(ValueError):
            units.dtype_bytes("fp12")

    def test_megabits_per_second(self):
        # chapter12.md:11: MB и Mbit десятичные, 20 Mbit/s = 2.5e6 байт/с
        self.assertEqual(units.megabits_per_second_to_bytes(20), 2_500_000)


if __name__ == "__main__":
    unittest.main()
