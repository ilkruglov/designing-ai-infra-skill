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
    def test_dense_qwen3(self) -> None:
        # config.json Qwen3-8B: num_hidden_layers 36, num_key_value_heads 8, head_dim 128;
        # chapter8.md:52: 2 × 36 × 8 × 128 × 2 = 147 456 байт KV на токен
        s = spec("qwen3-8b")
        self.assertEqual(
            (s.family, s.layers, s.kv_heads, s.head_dim, s.qk_norm),
            ("dense", 36, 8, 128, True),
        )

    def test_dense_llama(self) -> None:
        # config.json DeepSeek-R1-Distill-Llama-70B: 80 слоёв, 8 голов KV, без QK-norm
        # (chapter1.md:193: плотная модель на архитектуре Llama)
        s = spec("deepseek-r1-distill-llama-70b")
        self.assertEqual(
            (s.family, s.layers, s.kv_heads, s.qk_norm), ("dense", 80, 8, False)
        )

    def test_moe_qwen3(self) -> None:
        # config.json Qwen3-30B-A3B: num_experts 128, num_experts_per_tok 8,
        # moe_intermediate_size 768 (qwen3-30b-a3b-decode-b1-s8192.json: 128 экспертов, top-8)
        s = spec("qwen3-30b-a3b")
        self.assertEqual(
            (s.family, s.experts, s.experts_per_token, s.moe_intermediate),
            ("moe", 128, 8, 768),
        )

    def test_mla_deepseek_v3(self) -> None:
        # config.json DeepSeek-V3: first_k_dense_replace 3, kv_lora_rank 512,
        # qk_rope_head_dim 64; kv-comparison: 61 × (512 + 64) × 2 = 70 272 байт на токен
        s = spec("deepseek-v3")
        self.assertEqual(
            (s.family, s.first_dense_layers, s.kv_lora_rank, s.qk_rope_dim),
            ("mla_moe", 3, 512, 64),
        )

    def test_hybrid_qwen35_inside_text_config(self) -> None:
        # Review Focus 5: обёртка text_config мультимодального конфига; chapter2.md:418 и
        # config.json: 60 слоёв, full_attention_interval 4 — 15 полного внимания и 45
        # линейных, linear_conv_kernel_dim 4
        s = spec("qwen3.5-397b-a17b")
        self.assertEqual(
            (s.family, s.full_attention_layers, s.linear_layers, s.conv_kernel),
            ("hybrid_linear", 15, 45, 4),
        )
        # kv-comparison: 30 720 = 2 × 15 × kv_heads 2 × head_dim 256 × 2 байта
        self.assertEqual((s.kv_heads, s.head_dim), (2, 256))

    def test_unknown_architecture_names_fields_and_author_command(self) -> None:
        # Review Focus 1: никакого расчёта как плотной модели (поведение, чисел книги нет)
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

    def test_generation_keys_are_not_listed_as_unknown(self) -> None:
        # поля генерации из старых config.json не описывают архитектуру и в перечень
        # незнакомых полей не попадают (поведение, чисел книги нет)
        config = {
            "model_type": "gemma2",
            "hidden_size": 4096,
            "query_pre_attn_scalar": 256,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "do_sample": False,
            "num_beams": 1,
            "max_length": 20,
            "repetition_penalty": 1.0,
            "output_attentions": False,
            "return_dict": True,
            "is_encoder_decoder": False,
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["query_pre_attn_scalar"])

    def test_architectural_generation_defaults_are_listed_when_set(self) -> None:
        # is_encoder_decoder, add_cross_attention и pruned_heads — значения PretrainedConfig
        # по умолчанию, но, когда они включены, меняют архитектуру (кодировщик,
        # перекрёстное внимание, удалённые головы) и попадают в перечень полей
        base = {"model_type": "gemma2", "hidden_size": 4096}
        quiet = base | {
            "is_encoder_decoder": False,
            "add_cross_attention": False,
            "pruned_heads": {},
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(quiet)
        self.assertEqual(caught.exception.fields, ["model_type"])
        loud = base | {
            "is_encoder_decoder": True,
            "add_cross_attention": True,
            "pruned_heads": {"0": [1]},
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(loud)
        self.assertEqual(
            caught.exception.fields,
            ["add_cross_attention", "is_encoder_decoder", "pruned_heads"],
        )

    def test_unknown_model_type_with_familiar_fields_says_so(self) -> None:
        # все поля знакомы, незнаком только model_type: перечень не пустой «—»
        config = model.load_config(CONFIGS / "qwen3-8b.json") | {"model_type": "qwen9"}
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["model_type"])
        self.assertIn("незнакомый model_type", str(caught.exception))
        self.assertNotIn("поля: —", str(caught.exception))
        # без model_type — то же поле
        config = model.load_config(CONFIGS / "qwen3-8b.json")
        del config["model_type"]
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["model_type"])
        self.assertIn("model_type не указан", str(caught.exception))

    def test_mistral_window_is_read(self) -> None:
        # config.json mistral: sliding_window — окно на всех слоях (синтетический конфиг
        # на форме Llama; поведение разбора, чисел книги нет)
        base = model.load_config(CONFIGS / "deepseek-r1-distill-llama-70b.json")
        s = model.parse_spec(base | {"model_type": "mistral", "sliding_window": 4096})
        self.assertEqual((s.family, s.window, s.qk_norm), ("dense", 4096, False))
        full = model.parse_spec(
            base | {"model_type": "mistral", "sliding_window": None}
        )
        self.assertIsNone(full.window)

    def test_known_adapter_refuses_encoder_decoder_flags(self) -> None:
        # is_encoder_decoder, add_cross_attention и pruned_heads меняют архитектуру:
        # незнакомую архитектуру отвергают с перечнем полей — и знакомый адаптер тоже
        base = model.load_config(CONFIGS / "qwen3-8b.json")
        for key, value in (
            ("is_encoder_decoder", True),
            ("add_cross_attention", True),
            ("pruned_heads", {"0": [1]}),
        ):
            with self.subTest(key=key):
                with self.assertRaises(model.UnsupportedArchitecture) as caught:
                    model.parse_spec(base | {key: value})
                self.assertEqual(caught.exception.fields, [key])
        # значения по умолчанию архитектуру не меняют
        spec = model.parse_spec(
            base | {"is_encoder_decoder": False, "add_cross_attention": False,
                    "pruned_heads": {}}
        )  # fmt: skip
        self.assertEqual(spec.model_type, "qwen3")
        moe = model.load_config(CONFIGS / "qwen3-30b-a3b.json")
        with self.assertRaises(model.UnsupportedArchitecture):
            model.parse_spec(moe | {"is_encoder_decoder": True})

    def test_qwen3_partial_window_is_refused(self) -> None:
        # use_sliding_window у qwen3 включает окно только на части слоёв (max_window_layers)
        config = model.load_config(CONFIGS / "qwen3-8b.json") | {
            "use_sliding_window": True
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["use_sliding_window"])

    def test_deepseek_sparse_moe_layers_are_refused(self) -> None:
        # moe_layer_freq ≠ 1: MoE не на каждом слое после плотных, формула не подходит
        config = model.load_config(CONFIGS / "deepseek-v3.json") | {"moe_layer_freq": 2}
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["moe_layer_freq"])

    def test_attention_bias_is_refused(self) -> None:
        config = model.load_config(CONFIGS / "qwen3-8b.json") | {"attention_bias": True}
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["attention_bias"])

    def test_moe_with_dense_only_layers_is_refused(self) -> None:
        config = model.load_config(CONFIGS / "qwen3-30b-a3b.json") | {
            "mlp_only_layers": [0]
        }
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["mlp_only_layers"])


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
        self.assertEqual(
            caught.exception.fields, ["layer_types", "full_attention_interval"]
        )

    def test_invalid_full_attention_interval_names_both_fields(self) -> None:
        # без layer_types типы слоёв выводятся из full_attention_interval; негодный
        # интервал — отказ с обоими полями, из которых можно было взять типы
        for interval in (0, -4, True, 4.0, "4"):
            config = self._text_config(full_attention_interval=interval)
            del config["text_config"]["layer_types"]
            with self.subTest(interval=interval):
                with self.assertRaises(model.UnsupportedArchitecture) as caught:
                    model.parse_spec(config)
                self.assertEqual(
                    caught.exception.fields, ["layer_types", "full_attention_interval"]
                )

    def test_inconsistent_layer_types_name_only_layer_types(self) -> None:
        # список layer_types есть, но короче num_hidden_layers: виноват он, а не интервал
        config = self._text_config()
        config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:-1]
        with self.assertRaises(model.UnsupportedArchitecture) as caught:
            model.parse_spec(config)
        self.assertEqual(caught.exception.fields, ["layer_types"])


if __name__ == "__main__":
    unittest.main()
