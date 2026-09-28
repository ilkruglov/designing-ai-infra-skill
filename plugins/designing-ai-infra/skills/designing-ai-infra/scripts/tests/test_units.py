import unittest

from infra_calc import units

ANCHORS = (
    "references/source-book/chapter1.md:189",
    "references/source-book/chapter2.md:236",
    "references/source-book/chapter8.md:52",
    "references/source-book/chapter12.md:11",
)


class UnitsTest(unittest.TestCase):
    def test_weight_bytes_in_decimal_and_binary_units(self) -> None:
        # chapter8.md:52: веса Qwen3-8B BF16 — 16.38 GB, около 15.3 GiB
        weights = 16_381_470_720
        self.assertEqual(round(units.to_unit(weights, "GB"), 2), 16.38)
        self.assertEqual(round(units.to_unit(weights, "GiB"), 2), 15.26)

    def test_kv_per_token_and_per_context(self) -> None:
        # chapter8.md:52: 144 KiB на токен, 288 MiB для 2048, 1152 MiB для 8192
        self.assertEqual(units.to_unit(147_456, "KiB"), 144)
        self.assertEqual(units.to_unit(147_456 * 2048, "MiB"), 288)
        self.assertEqual(units.to_unit(147_456 * 8192, "MiB"), 1152)

    def test_parse_bytes_requires_explicit_unit(self) -> None:
        # chapter8.md:52: 144 KiB KV на токен = 147 456 байт
        self.assertEqual(units.parse_bytes("144KiB"), 147_456)
        # chapter1.md:450: веса 16.38 GB (GB = 10⁹, chapter1.md:189)
        self.assertEqual(units.parse_bytes("16.38 GB"), 16.38e9)
        # без явной единицы GB/GiB не отличить (поведение, чисел книги нет)
        for ambiguous in ("16G", "16", "16 gb", "16 Gb"):
            with self.assertRaises(ValueError):
                units.parse_bytes(ambiguous)

    def test_dtype_bytes(self) -> None:
        # chapter1.md:176: «BF16 — формат с плавающей точкой, в котором каждое число
        # занимает 16 бит (2 байта)»
        self.assertEqual(units.dtype_bytes("bf16"), 2)
        # chapter2.md:498: FP4 — «каждое значение занимает 4 бита»
        self.assertEqual(units.dtype_bytes("FP4"), 0.5)
        # незнакомый формат — отказ (поведение)
        with self.assertRaises(ValueError):
            units.dtype_bytes("fp12")

    def test_dtype_aliases_from_config_torch_dtype(self) -> None:
        # chapter2.md:236: config.json пишет dtype как "bfloat16"
        self.assertEqual(units.dtype_bytes("bfloat16"), 2)
        self.assertEqual(units.dtype_bytes("BFloat16"), 2)
        self.assertEqual(units.dtype_bytes("float16"), 2)
        self.assertEqual(units.dtype_bytes("Float32"), 4)

    def test_megabits_per_second(self) -> None:
        # chapter12.md:11: MB и Mbit десятичные, 20 Mbit/s = 2.5e6 байт/с
        self.assertEqual(units.megabits_per_second_to_bytes(20), 2_500_000)


if __name__ == "__main__":
    unittest.main()
