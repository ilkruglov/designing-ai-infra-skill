import unittest

from infra_calc import hardware

ANCHORS = ("references/source-book/chapter1.md:263",)


class HardwareTest(unittest.TestCase):
    def test_h100_values_used_in_book(self) -> None:
        # chapter1.md:263: 3350 GB/s и 989.4 TFLOPs BF16
        h100 = hardware.device("h100-sxm")
        self.assertEqual(hardware.bandwidth(h100), 3.35e12)
        self.assertEqual(hardware.peak_flops(h100, "BF16"), 989.4e12)
        self.assertEqual(h100.memory_bytes, 80 * 10**9)

    def test_dense_is_default_not_structured_sparsity(self) -> None:
        # Review Focus 3
        h100 = hardware.device("h100-sxm")
        self.assertLess(
            hardware.peak_flops(h100, "BF16"),
            hardware.peak_flops(h100, "BF16", sparsity="structured"),
        )

    def test_unknown_precision_lists_available(self) -> None:
        h100 = hardware.device("h100-sxm")
        with self.assertRaises(ValueError) as caught:
            hardware.peak_flops(h100, "FP4")
        self.assertIn("BF16", str(caught.exception))

    def test_missing_bandwidth_asks_for_argument(self) -> None:
        missing = [d for d in hardware.load_devices().values() if d.bandwidth is None]
        self.assertTrue(missing)
        with self.assertRaises(ValueError) as caught:
            hardware.bandwidth(missing[0])
        self.assertIn("--bandwidth", str(caught.exception))

    def test_unknown_device_suggests_ids(self) -> None:
        with self.assertRaises(KeyError) as caught:
            hardware.device("h100")
        self.assertIn("h100-sxm", str(caught.exception))

    def test_snapshot_is_named(self) -> None:
        self.assertIn("56ecb425", hardware.SNAPSHOT)


if __name__ == "__main__":
    unittest.main()
