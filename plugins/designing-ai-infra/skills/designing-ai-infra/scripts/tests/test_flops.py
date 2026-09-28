import unittest
from dataclasses import replace
from pathlib import Path

from infra_calc import flops, model

ANCHORS = (
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter3.md:442",
    "calculations/results/qwen3-30b-a3b-decode-b1-s8192.json#sha256=fbf0b07f78bd8d7ab3765f6fc9ad5f6992cc95192449c2f8f1ee0fd01b5bc75b",
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

    def test_window_forward_is_refused(self) -> None:
        windowed = replace(QWEN, window=4096)
        with self.assertRaises(model.UnsupportedArchitecture):
            flops.forward_matrix_flops(windowed, new_tokens=1)

    def test_hybrid_forward_is_refused(self) -> None:
        hybrid = model.parse_spec(model.load_config(CONFIGS / "qwen3.5-397b-a17b.json"))
        with self.assertRaises(model.UnsupportedArchitecture):
            flops.forward_matrix_flops(hybrid, new_tokens=1)

    def test_invalid_token_counts_are_rejected(self) -> None:
        for kwargs in (
            {"new_tokens": 0},
            {"new_tokens": 1, "history": -1},
            {"new_tokens": 1, "batch": 0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                flops.forward_matrix_flops(QWEN, **kwargs)
        for args in ((0, 0), (1, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                flops.attention_matrix_flops(QWEN, *args)

    def test_fractional_counts_are_rejected(self) -> None:
        for kwargs in (
            {"new_tokens": 2.5},
            {"new_tokens": 1, "history": 0.5},
            {"new_tokens": 1, "batch": 1.5},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                flops.forward_matrix_flops(QWEN, **kwargs)


class MoeForwardTest(unittest.TestCase):
    def test_qwen3_30b_a3b_decode_step(self) -> None:
        moe = model.parse_spec(model.load_config(CONFIGS / "qwen3-30b-a3b.json"))
        # qwen3-30b-a3b-decode-b1-s8192.json: matrix_flops
        self.assertEqual(
            flops.forward_matrix_flops(moe, new_tokens=1, history=8192),
            12_526_551_040,
        )
        # qwen3-30b-a3b-decode-b1-s8192.json: causal_attention_matrix_flops
        self.assertEqual(flops.attention_matrix_flops(moe, 1, 8192), 6_443_237_376)


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
