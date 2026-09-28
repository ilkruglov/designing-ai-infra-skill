import unittest
from pathlib import Path

from infra_calc import accounting, model

ANCHORS = (
    "references/source-book/chapter1.md:189",
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter2.md:236",
    "references/source-book/chapter2.md:418",
    "references/source-book/chapter2.md:526",
    "references/source-book/chapter8.md:52",
    "calculations/results/kv-comparison-n8192-b1.json#sha256=1a45a55edf05115855d798572a84cec00dddee2418589e2fdc3aed968441a257",
    "calculations/results/v3-forward-b64.json#sha256=91bf7eff7450033791c452788a1b2a249665cb263df6e131f823e77976d00598",
)
CONFIGS = Path(__file__).resolve().parent / "fixtures" / "configs"


def spec(name: str) -> model.ModelSpec:
    return model.parse_spec(model.load_config(CONFIGS / f"{name}.json"))


class DenseTest(unittest.TestCase):
    def test_qwen3_8b_reference_numbers(self) -> None:
        s = spec("qwen3-8b")
        self.assertEqual(accounting.parameter_count(s), 8_190_735_360)
        # chapter1.md:450: веса BF16 16.38 GB
        self.assertEqual(accounting.weight_bytes(s), 16_381_470_720)
        # chapter8.md:52: 2 × 36 × 8 × 128 × 2 = 147 456 байт
        self.assertEqual(accounting.kv_bytes_per_token(s), 147_456)
        # chapter2.md:236 (пример 2-2): чтение общих весов за шаг decode
        self.assertEqual(accounting.decode_weight_read_bytes(s), 15_136_811_008)
        # kv-comparison-n8192-b1.json: global_history_bytes
        self.assertEqual(accounting.kv_resident_bytes(s, 8192), 1_207_959_552)
        self.assertEqual(
            accounting.parameter_count(s, active=True), accounting.parameter_count(s)
        )
        # таблица эмбеддингов 151 936 × 4096, исключается из чтения за шаг decode
        self.assertEqual(accounting.embedding_parameters(s), 622_329_856)

    def test_llama_70b(self) -> None:
        s = spec("deepseek-r1-distill-llama-70b")
        # chapter1.md:189: веса BF16 141.11 GB
        self.assertEqual(round(accounting.weight_bytes(s) / 1e9, 2), 141.11)
        self.assertEqual(accounting.kv_bytes_per_token(s), 327_680)


class MoeTest(unittest.TestCase):
    def test_qwen3_30b_a3b(self) -> None:
        s = spec("qwen3-30b-a3b")
        # qwen3-30b-a3b-decode-b1-s8192.json: parameters
        self.assertEqual(accounting.parameter_count(s), 30_532_122_624)
        # parameters − routed_expert_total + routed_expert_active_per_token
        self.assertEqual(
            accounting.parameter_count(s, active=True),
            30_532_122_624 - 28_991_029_248 + 1_811_939_328,
        )
        self.assertEqual(accounting.kv_bytes_per_token(s), 98_304)


class MlaTest(unittest.TestCase):
    def test_deepseek_v3(self) -> None:
        s = spec("deepseek-v3")
        # v3-forward-b64.json: logical_base_parameters
        self.assertEqual(accounting.parameter_count(s), 671_026_419_200)
        # chapter2.md:526: около 37B активных параметров на токен
        self.assertEqual(int(accounting.parameter_count(s, active=True) / 1e9), 37)
        # kv-comparison: 61 × (512 + 64) × 2 байта
        self.assertEqual(accounting.kv_bytes_per_token(s), 70_272)


class HybridTest(unittest.TestCase):
    def test_qwen35_state(self) -> None:
        s = spec("qwen3.5-397b-a17b")
        # kv-comparison: 15 слоёв полного внимания, фиксированное состояние 45 линейных
        self.assertEqual(accounting.kv_bytes_per_token(s), 30_720)
        self.assertEqual(accounting.fixed_state_bytes(s), (188_743_680, 4_423_680))

    def test_hybrid_parameters_are_not_guessed(self) -> None:
        with self.assertRaises(model.UnsupportedArchitecture):
            accounting.parameter_count(spec("qwen3.5-397b-a17b"))


class WindowTest(unittest.TestCase):
    def test_window_caps_resident_kv(self) -> None:
        s = model.ModelSpec(
            model_type="mistral",
            family="dense",
            layers=2,
            hidden=64,
            heads=4,
            kv_heads=2,
            head_dim=16,
            vocab=100,
            tied_embeddings=False,
            intermediate=128,
            window=1024,
        )
        per_token = accounting.kv_bytes_per_token(s)
        self.assertEqual(accounting.kv_resident_bytes(s, 4096), per_token * 1024)
        self.assertEqual(accounting.kv_resident_bytes(s, 512), per_token * 512)


if __name__ == "__main__":
    unittest.main()
