import unittest
from pathlib import Path

from infra_calc import model

ANCHORS = (
    "references/source-book/chapter2.md:236",
    "references/source-book/chapter2.md:418",
    "references/source-book/chapter2.md:935",
)
CONFIGS = Path(__file__).resolve().parent / "fixtures" / "configs"


def spec(name: str) -> model.ModelSpec:
    return model.parse_spec(model.load_config(CONFIGS / f"{name}.json"))


class ParseSpecTest(unittest.TestCase):
    def test_dense_qwen3(self):
        s = spec("qwen3-8b")
        self.assertEqual(
            (s.family, s.layers, s.kv_heads, s.head_dim, s.qk_norm),
            ("dense", 36, 8, 128, True),
        )

    def test_dense_llama(self):
        s = spec("deepseek-r1-distill-llama-70b")
        self.assertEqual(
            (s.family, s.layers, s.kv_heads, s.qk_norm), ("dense", 80, 8, False)
        )

    def test_moe_qwen3(self):
        s = spec("qwen3-30b-a3b")
        self.assertEqual(
            (s.family, s.experts, s.experts_per_token, s.moe_intermediate),
            ("moe", 128, 8, 768),
        )

    def test_mla_deepseek_v3(self):
        s = spec("deepseek-v3")
        self.assertEqual(
            (s.family, s.first_dense_layers, s.kv_lora_rank, s.qk_rope_dim),
            ("mla_moe", 3, 512, 64),
        )

    def test_hybrid_qwen35_inside_text_config(self):
        # Review Focus 5: обёртка text_config мультимодального конфига
        s = spec("qwen3.5-397b-a17b")
        self.assertEqual(
            (s.family, s.full_attention_layers, s.linear_layers, s.conv_kernel),
            ("hybrid_linear", 15, 45, 4),
        )

    def test_unknown_architecture_names_fields_and_author_command(self):
        # Review Focus 1: никакого расчёта как плотной модели
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 4096,
            "compress_ratios": [4],
            "hc_mult": 4,
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["compress_ratios", "hc_mult"])
        self.assertIn("v4-forward", str(caught.exception))

    def test_attention_bias_is_refused(self):
        config = model.load_config(CONFIGS / "qwen3-8b.json") | {"attention_bias": True}
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["attention_bias"])

    def test_moe_with_dense_only_layers_is_refused(self):
        config = model.load_config(CONFIGS / "qwen3-30b-a3b.json") | {
            "mlp_only_layers": [0]
        }
        with self.assertRaises(model.UnsupportedArchitecture):
            model.parse_spec(config)


class WrapperTest(unittest.TestCase):
    # chapter2.md:236: обёртка text_config мультимодального конфига
    def test_wrapper_model_type_is_recorded(self) -> None:
        self.assertEqual(spec("qwen3.5-397b-a17b").wrapper_model_type, "qwen3_5_moe")
        self.assertIsNone(spec("qwen3-8b").wrapper_model_type)

    def test_text_config_that_is_not_an_object_is_refused(self) -> None:
        for inner in (None, [1, 2], "qwen3"):
            with self.assertRaises(model.UnsupportedArchitecture) as caught:
                model.parse_spec({"model_type": "x", "text_config": inner})
            self.assertEqual(caught.exception.fields, ["text_config"])
            self.assertEqual(caught.exception.model_type, "x")

    def test_inner_config_inherits_outer_tie_word_embeddings(self) -> None:
        config = model.load_config(CONFIGS / "qwen3.5-397b-a17b.json")
        self.assertNotIn("tie_word_embeddings", config["text_config"])
        self.assertTrue(
            model.parse_spec(config | {"tie_word_embeddings": True}).tied_embeddings
        )
        self.assertFalse(model.parse_spec(config).tied_embeddings)

    def test_inner_tie_word_embeddings_wins_over_outer(self) -> None:
        config = model.load_config(CONFIGS / "qwen3.5-397b-a17b.json")
        config["tie_word_embeddings"] = True
        config["text_config"] = config["text_config"] | {"tie_word_embeddings": False}
        self.assertFalse(model.parse_spec(config).tied_embeddings)


class HybridRulesTest(unittest.TestCase):
    # chapter2.md:418: состав состояния при гибридном внимании
    def _text_config(self, **changes: object) -> dict[str, object]:
        config = model.load_config(CONFIGS / "qwen3.5-397b-a17b.json")
        config["text_config"] = config["text_config"] | changes
        return config

    def test_attention_bias_is_refused(self) -> None:
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(self._text_config(attention_bias=True))
        self.assertEqual(caught.exception.fields, ["attention_bias"])

    def test_dense_only_layers_are_refused(self) -> None:
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(self._text_config(mlp_only_layers=[0]))
        self.assertEqual(caught.exception.fields, ["mlp_only_layers"])

    def test_layer_types_derived_from_full_attention_interval(self) -> None:
        config = self._text_config()
        del config["text_config"]["layer_types"]
        s = model.parse_spec(config)
        # те же 15 слоёв полного внимания и 45 линейных, что и в фикстуре
        self.assertEqual((s.full_attention_layers, s.linear_layers), (15, 45))
        self.assertEqual(s, spec("qwen3.5-397b-a17b"))

    def test_missing_layer_types_and_interval_is_refused(self) -> None:
        config = self._text_config()
        del config["text_config"]["layer_types"]
        del config["text_config"]["full_attention_interval"]
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["layer_types"])


if __name__ == "__main__":
    unittest.main()
