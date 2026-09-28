import unittest
from pathlib import Path

from infra_calc import flops, model

ANCHORS = (
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter3.md:442",
)
CONFIGS = Path(__file__).resolve().parent / "fixtures" / "configs"
QWEN = model.parse_spec(model.load_config(CONFIGS / "qwen3-8b.json"))


class ForwardTest(unittest.TestCase):
    def test_prefill_2048(self) -> None:
        # chapter1.md:450: prefill 2048 токенов — 29.69 TFLOPs
        value = flops.forward_matrix_flops(QWEN, new_tokens=2048)
        self.assertEqual(round(value / 1e12, 2), 29.69)

    def test_decode_step_at_context_2048(self) -> None:
        # chapter1.md:450: шаг decode — 16.34 GFLOPs; контекст 2048 с новым токеном
        value = flops.forward_matrix_flops(QWEN, new_tokens=1, history=2047)
        self.assertEqual(round(value / 1e9, 2), 16.34)

    def test_batch_scales_linearly(self) -> None:
        one = flops.forward_matrix_flops(QWEN, new_tokens=1, history=2047)
        self.assertEqual(
            flops.forward_matrix_flops(QWEN, new_tokens=1, history=2047, batch=8),
            8 * one,
        )

    def test_mla_forward_is_refused(self) -> None:
        v3 = model.parse_spec(model.load_config(CONFIGS / "deepseek-v3.json"))
        with self.assertRaises(model.UnsupportedArchitecture):
            flops.forward_matrix_flops(v3, new_tokens=1)


class TrainingTest(unittest.TestCase):
    def test_qwen3_8b_8192(self) -> None:
        # chapter3.md:442: 143.789 / 287.579 / 431.368 TFLOPs, внимание 59.381
        forward, backward, total = flops.training_matrix_flops(QWEN, tokens=8192)
        self.assertEqual(round(forward / 1e12, 3), 143.789)
        self.assertEqual(round(backward / 1e12, 3), 287.579)
        self.assertEqual(round(total / 1e12, 3), 431.368)
        self.assertEqual(
            round(3 * flops.attention_matrix_flops(QWEN, 8192) / 1e12, 3), 59.381
        )

    def test_six_nd(self) -> None:
        # chapter3.md:442: 6ND = 402.591 TFLOPs
        self.assertEqual(
            round(flops.six_nd_flops(8_190_735_360, 8192) / 1e12, 3), 402.591
        )


if __name__ == "__main__":
    unittest.main()
