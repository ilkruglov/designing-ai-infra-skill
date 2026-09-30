import unittest
from pathlib import Path

from infra_calc import hardware

ANCHORS = (
    "references/source-book/chapter1.md:152",
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter8.md:3",
)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class HardwareTest(unittest.TestCase):
    def test_h100_values_used_in_book(self) -> None:
        # chapter1.md:268: «\frac{70\ \mathrm{GB}}{3350\ \mathrm{GB/s}}» и
        # «\frac{140\ \mathrm{GFLOPs}}{989400\ \mathrm{GFLOP/s}}»
        h100 = hardware.device("h100-sxm")
        self.assertEqual(hardware.bandwidth(h100), 3.35e12)
        self.assertEqual(hardware.peak_flops(h100, "BF16"), 989.4e12)
        # chapter1.md:182: «Ёмкость видеопамяти (GB) | 24 | 80 | 80»
        self.assertEqual(h100.memory_bytes, 80 * 10**9)

    def test_reference_table_of_section_1_2_2(self) -> None:
        # chapter1.md:178: пик плотных вычислений «с входными данными BF16 и накоплением FP32»
        # chapter1.md:182-184: 24/80/80 GB, 1.008/2.039/3.35 TB/s, 165.2/312/989.4 TFLOP/s
        for device_id, capacity, bandwidth, peak in (
            ("rtx4090", 24e9, 1.008e12, 165.2e12),
            ("a100-80gb-sxm", 80e9, 2.039e12, 312e12),
            ("h100-sxm", 80e9, 3.35e12, 989.4e12),
        ):
            with self.subTest(device=device_id):
                dev = hardware.device(device_id)
                self.assertEqual(dev.memory_bytes, capacity)
                self.assertEqual(hardware.bandwidth(dev), bandwidth)
                self.assertEqual(hardware.peak_flops(dev, "BF16"), peak)

    def test_rtx_pro_6000_workstation(self) -> None:
        # chapter1.md:464: «При пиковой производительности матричных вычислений 503.8 TFLOP/s»,
        # «при пропускной способности видеопамяти 1.792 TB/s»
        dev = hardware.device("rtx-pro6000-blackwell-ws")
        self.assertEqual(hardware.peak_flops(dev, "BF16"), 503.8e12)
        self.assertEqual(hardware.bandwidth(dev), 1.792e12)

    def test_accumulator_defaults_to_fp32(self) -> None:
        # chapter1.md:178: книга считает с накоплением FP32; у RTX 4090 в снимке
        # FP16 с накоплением FP32 — 165.2, с накоплением FP16 — 330.3 (data/hardware.json)
        rtx = hardware.device("rtx4090")
        self.assertEqual(hardware.peak_flops(rtx, "FP16"), 165.2e12)
        self.assertEqual(hardware.peak_flops(rtx, "FP16", accumulator="FP16"), 330.3e12)

    def test_ambiguous_accumulator_is_refused(self) -> None:
        rtx = hardware.device("rtx4090")
        with self.assertRaises(ValueError) as caught:
            hardware.peak_flops(rtx, "FP16", accumulator=None)
        self.assertIn("FP16", str(caught.exception))
        self.assertIn("FP32", str(caught.exception))

    def test_dense_is_default_not_structured_sparsity(self) -> None:
        # Review Focus 3: 989.4 dense против 1978.9 structured (data/hardware.json)
        h100 = hardware.device("h100-sxm")
        self.assertLess(
            hardware.peak_flops(h100, "BF16"),
            hardware.peak_flops(h100, "BF16", sparsity="structured"),
        )

    def test_unspecified_sparsity_is_never_a_denominator(self) -> None:
        # в снимке у rtx-pro6000-blackwell-server BF16 только запись 1000 с sparsity
        # «unspecified»; плотный пик книги — 503.8 (chapter1.md:464)
        server = hardware.device("rtx-pro6000-blackwell-server")
        with self.assertRaises(ValueError):
            hardware.peak_flops(server, "BF16", accumulator=None)
        with self.assertRaises(ValueError):
            hardware.peak_flops(
                server, "BF16", sparsity="unspecified", accumulator=None
            )

    def test_integer_tops_are_not_flops(self) -> None:
        h100 = hardware.device("h100-sxm")
        with self.assertRaises(ValueError) as caught:
            hardware.peak_flops(h100, "INT8", accumulator="INT32")
        self.assertIn("TOPS", str(caught.exception))

    def test_unknown_precision_lists_available(self) -> None:
        h100 = hardware.device("h100-sxm")
        with self.assertRaises(ValueError) as caught:
            hardware.peak_flops(h100, "FP4")
        message = str(caught.exception)
        self.assertIn("BF16/FP32/tensor/dense", message)
        self.assertIn("d0cc188b", message)
        self.assertNotIn("INT8", message)

    def test_missing_bandwidth_asks_for_argument(self) -> None:
        missing = [
            d
            for d in hardware.load_devices().values()
            if d.bandwidth is None and d.scope == "single_device"
        ]
        self.assertTrue(missing)
        with self.assertRaises(ValueError) as caught:
            hardware.bandwidth(missing[0])
        self.assertIn("--bandwidth", str(caught.exception))
        self.assertIn("d0cc188b", str(caught.exception))

    def test_unknown_device_suggests_ids(self) -> None:
        with self.assertRaises(KeyError) as caught:
            hardware.device("h100")
        self.assertIn("h100-sxm", str(caught.exception))

    def test_snapshot_is_named(self) -> None:
        self.assertIn("d0cc188b", hardware.SNAPSHOT)

    def test_snapshot_is_dated(self) -> None:
        # дата коммита d0cc188b: git -C .tmp/upcalc log -1 --format=%cs d0cc188b
        self.assertIn("2026-09-30", hardware.SNAPSHOT)


class ScopeTest(unittest.TestCase):
    def test_single_device_scope(self) -> None:
        h100 = hardware.device("h100-sxm")
        self.assertEqual(h100.scope, "single_device")
        self.assertEqual(h100.device_count, 1)
        self.assertFalse(h100.shared_with_cpu)

    def test_aggregate_requires_opt_in(self) -> None:
        # стойка GB200 NVL72 в снимке — сумма по 72 GPU; как одна карта она не выдаётся
        with self.assertRaises(ValueError) as caught:
            hardware.device("gb200-nvl72")
        self.assertIn("72", str(caught.exception))
        self.assertIn("allow_aggregate", str(caught.exception))

    def test_aggregate_reports_scope_and_count(self) -> None:
        rack = hardware.device("gb200-nvl72", allow_aggregate=True)
        self.assertEqual(rack.scope, "gpu_aggregate")
        self.assertEqual(rack.device_count, 72)
        node = hardware.device("atlas-800i-a3-8npu", allow_aggregate=True)
        self.assertEqual(node.scope, "npu_aggregate")
        self.assertEqual(node.device_count, 8)

    def test_unified_memory_is_visible(self) -> None:
        # chapter8.md:9: «эксперимента 8-7, выполненного на Apple M2 Max»; общая с CPU
        # память — поле shared_with_cpu снимка
        self.assertTrue(hardware.device("m2-max-38gpu-96gb").shared_with_cpu)


class LoaderTest(unittest.TestCase):
    def test_loaded_snapshot_is_read_only(self) -> None:
        devices = hardware.load_devices()
        with self.assertRaises(TypeError):
            devices["h100-sxm"] = devices["rtx4090"]  # type: ignore[index]
        self.assertEqual(hardware.device("h100-sxm").id, "h100-sxm")

    def test_peaks_of_cached_snapshot_are_read_only(self) -> None:
        # снимок кэшируется: запись в dev.peaks[i] изменила бы пик для всех следующих
        # вызовов; chapter1.md:268 — пик H100 BF16 989.4 TFLOP/s остаётся прежним
        dev = hardware.device("h100-sxm")
        peak = dev.peaks[0]
        with self.assertRaises(TypeError):
            peak["tera_ops_per_second"] = 1  # type: ignore[index]
        nested = next(p for p in dev.peaks if p.get("supporting_evidence"))
        self.assertIsInstance(nested["supporting_evidence"], tuple)
        with self.assertRaises(TypeError):
            nested["supporting_evidence"][0]["claim"] = ""  # type: ignore[index]
        self.assertEqual(
            hardware.peak_flops(hardware.device("h100-sxm"), "BF16"), 989.4e12
        )

    def test_unknown_capacity_unit_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            hardware.load_devices(FIXTURES / "hardware-unknown-unit.json")
        self.assertIn("TB", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
