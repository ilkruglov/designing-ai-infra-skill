"""Архитектура модели из официального config.json в формате Hugging Face.

Поддержаны семейства, формулы которых сверены с числами книги и результатами
автора: плотные (qwen3, llama, mistral), MoE (qwen3_moe), MLA с MoE
(deepseek_v3) и гибридное линейное внимание (qwen3_5_moe_text, qwen3_next).
Незнакомая архитектура не считается как плотная — это правило книги и кода
автора; вместо этого перечисляются незнакомые поля.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

AUTHOR_COMMANDS = {
    "deepseek_v4": "Используйте `python3 calculations/calc.py v4-forward` в репозитории оригинала на коммите 56ecb425.",
    "kimi_linear": "Используйте `python3 calculations/calc.py k3-forward` в репозитории оригинала на коммите 56ecb425.",
}
# Поля архитектуры, которые читают адаптеры поддержанных семейств.
KNOWN_KEYS = {
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "tie_word_embeddings",
    "attention_bias",
    "mlp_bias",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "decoder_sparse_step",
    "mlp_only_layers",
    "shared_expert_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "first_k_dense_replace",
    "moe_layer_freq",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "layer_types",
    "full_attention_interval",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "text_config",
}
# Поля, которые на веса, KV и FLOPs не влияют: служебные, нормировка, RoPE, обучение.
RUNTIME_KEYS = {
    "architectures",
    "torch_dtype",
    "dtype",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "max_position_embeddings",
    "initializer_range",
    "hidden_act",
    "use_cache",
    "transformers_version",
    "attention_dropout",
    "_name_or_path",
    "auto_map",
    "norm_topk_prob",
    "router_aux_loss_coef",
    "output_router_logits",
}
# Значения PretrainedConfig по умолчанию для генерации, токенов и вывода, которые
# старые config.json записывают целиком. Архитектуру они не описывают; без этого
# списка отказ для незнакомой архитектуры тонул бы в них.
GENERATION_KEYS = {
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sep_token_id",
    "decoder_start_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "max_length",
    "min_length",
    "do_sample",
    "early_stopping",
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "bad_words_ids",
    "num_return_sequences",
    "output_scores",
    "return_dict_in_generate",
    "remove_invalid_values",
    "exponential_decay_length_penalty",
    "suppress_tokens",
    "begin_suppress_tokens",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "torchscript",
    "use_bfloat16",
    "tf_legacy_loss",
    "pruned_heads",
    "chunk_size_feed_forward",
    "is_encoder_decoder",
    "is_decoder",
    "add_cross_attention",
    "tie_encoder_decoder",
    "cross_attention_hidden_size",
    "finetuning_task",
    "id2label",
    "label2id",
    "prefix",
    "problem_type",
    "task_specific_params",
    "tokenizer_class",
}
BENIGN_KEYS = RUNTIME_KEYS | GENERATION_KEYS


class UnsupportedArchitecture(ValueError):
    def __init__(
        self,
        model_type: str,
        fields: list[str],
        hint: str = "",
        *,
        detail: str = "",
    ) -> None:
        self.model_type = model_type
        self.fields = fields
        # detail заменяет перечень полей, когда причина — не поле, а сам model_type
        detail = detail or f"поля: {', '.join(fields) or '—'}"
        message = f"архитектура {model_type!r} не поддержана калькулятором; {detail}"
        if hint:
            message += f". {hint}"
        super().__init__(message)


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    family: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool
    qk_norm: bool = False
    intermediate: int = 0
    window: int | None = None
    moe_intermediate: int = 0
    experts: int = 0
    experts_per_token: int = 0
    shared_experts: int = 0
    first_dense_layers: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    qk_nope_dim: int = 0
    v_head_dim: int = 0
    full_attention_layers: int = 0
    linear_layers: int = 0
    linear_key_heads: int = 0
    linear_value_heads: int = 0
    linear_key_dim: int = 0
    linear_value_dim: int = 0
    conv_kernel: int = 0
    wrapper_model_type: str | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_spec(config: dict[str, Any]) -> ModelSpec:
    if "text_config" not in config:
        return _parse_text(config)
    outer_type = str(config.get("model_type", "")) or "не указан"
    inner = config["text_config"]
    if not isinstance(inner, dict):
        raise UnsupportedArchitecture(
            outer_type,
            ["text_config"],
            "Ожидается объект text_config с параметрами текстовой модели.",
        )
    cfg = dict(inner)
    # tie_word_embeddings мультимодальные конфиги пишут то внутри, то снаружи
    if "tie_word_embeddings" not in cfg:
        cfg["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", False))
    return replace(_parse_text(cfg), wrapper_model_type=outer_type)


def _parse_text(cfg: dict[str, Any]) -> ModelSpec:
    model_type = str(cfg.get("model_type", ""))
    if model_type in {"qwen3", "llama", "mistral"}:
        return _dense(cfg, model_type)
    if model_type == "qwen3_moe":
        return _qwen3_moe(cfg)
    if model_type == "deepseek_v3":
        return _deepseek_v3(cfg)
    if model_type in {"qwen3_5_moe_text", "qwen3_next"}:
        return _hybrid(cfg, model_type)
    hint = AUTHOR_COMMANDS.get(model_type, "Нужен адаптер с эталонным тестом.")
    if not model_type:
        raise UnsupportedArchitecture(
            "не указан", ["model_type"], hint, detail="model_type не указан в config"
        )
    unknown = sorted(set(cfg) - KNOWN_KEYS - BENIGN_KEYS)
    if not unknown:
        raise UnsupportedArchitecture(
            model_type,
            ["model_type"],
            hint,
            detail="незнакомый model_type; остальные поля config калькулятору знакомы",
        )
    raise UnsupportedArchitecture(model_type, unknown, hint)


def _common(cfg: dict[str, Any], model_type: str, family: str) -> dict[str, Any]:
    heads = int(cfg["num_attention_heads"])
    return {
        "model_type": model_type,
        "family": family,
        "layers": int(cfg["num_hidden_layers"]),
        "hidden": int(cfg["hidden_size"]),
        "heads": heads,
        "kv_heads": int(cfg.get("num_key_value_heads") or heads),
        "head_dim": int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads),
        "vocab": int(cfg["vocab_size"]),
        "tied_embeddings": bool(cfg.get("tie_word_embeddings", False)),
    }


def _refuse(model_type: str, bad: list[str], hint: str) -> None:
    if bad:
        raise UnsupportedArchitecture(model_type, bad, hint)


def _dense(cfg: dict[str, Any], model_type: str) -> ModelSpec:
    bad = [key for key in ("attention_bias", "mlp_bias") if cfg.get(key)]
    _refuse(model_type, bad, "Смещения в проекциях формулой не учтены.")
    window = None
    if model_type == "qwen3":
        _refuse(
            model_type,
            ["use_sliding_window"] if cfg.get("use_sliding_window") else [],
            "Окно только на части слоёв (max_window_layers) не поддержано.",
        )
    elif cfg.get("sliding_window"):
        window = int(cfg["sliding_window"])
    return ModelSpec(
        **_common(cfg, model_type, "dense"),
        qk_norm=model_type == "qwen3",
        intermediate=int(cfg["intermediate_size"]),
        window=window,
    )


def _qwen3_moe(cfg: dict[str, Any]) -> ModelSpec:
    bad = []
    if cfg.get("attention_bias"):
        bad.append("attention_bias")
    if cfg.get("mlp_only_layers"):
        bad.append("mlp_only_layers")
    if cfg.get("decoder_sparse_step", 1) != 1:
        bad.append("decoder_sparse_step")
    if cfg.get("shared_expert_intermediate_size"):
        bad.append("shared_expert_intermediate_size")
    _refuse(
        "qwen3_moe",
        bad,
        "Поддержан только вариант, где каждый слой — MoE без общих экспертов.",
    )
    return ModelSpec(
        **_common(cfg, "qwen3_moe", "moe"),
        qk_norm=True,
        moe_intermediate=int(cfg["moe_intermediate_size"]),
        experts=int(cfg["num_experts"]),
        experts_per_token=int(cfg["num_experts_per_tok"]),
    )


def _deepseek_v3(cfg: dict[str, Any]) -> ModelSpec:
    _refuse(
        "deepseek_v3",
        ["moe_layer_freq"] if cfg.get("moe_layer_freq", 1) != 1 else [],
        "Поддержан только вариант, где MoE идёт каждым слоем после плотных.",
    )
    common = _common(cfg, "deepseek_v3", "mla_moe")
    common["kv_heads"] = common["heads"]
    common["head_dim"] = int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"])
    return ModelSpec(
        **common,
        intermediate=int(cfg["intermediate_size"]),
        moe_intermediate=int(cfg["moe_intermediate_size"]),
        experts=int(cfg["n_routed_experts"]),
        experts_per_token=int(cfg["num_experts_per_tok"]),
        shared_experts=int(cfg.get("n_shared_experts") or 0),
        first_dense_layers=int(cfg.get("first_k_dense_replace") or 0),
        q_lora_rank=int(cfg.get("q_lora_rank") or 0),
        kv_lora_rank=int(cfg["kv_lora_rank"]),
        qk_rope_dim=int(cfg["qk_rope_head_dim"]),
        qk_nope_dim=int(cfg["qk_nope_head_dim"]),
        v_head_dim=int(cfg["v_head_dim"]),
    )


def _hybrid(cfg: dict[str, Any], model_type: str) -> ModelSpec:
    bad = []
    if cfg.get("attention_bias"):
        bad.append("attention_bias")
    if cfg.get("mlp_only_layers"):
        bad.append("mlp_only_layers")
    _refuse(
        model_type,
        bad,
        "Поддержан только вариант без смещений в проекциях внимания, где каждый слой — MoE.",
    )
    listed = cfg.get("layer_types")
    types = listed or _layer_types_from_interval(cfg)
    full = sum(1 for t in types if t == "full_attention")
    linear = sum(1 for t in types if t == "linear_attention")
    if len(types) != int(cfg["num_hidden_layers"]) or full + linear != len(types):
        # без списка типы выводятся из интервала — тогда виноваты оба поля
        fields = (
            ["layer_types"] if listed else ["layer_types", "full_attention_interval"]
        )
        raise UnsupportedArchitecture(
            model_type,
            fields,
            "Нужен полный список типов слоёв layer_types или целый "
            "full_attention_interval ≥ 1.",
        )
    return ModelSpec(
        **_common(cfg, model_type, "hybrid_linear"),
        moe_intermediate=int(cfg.get("moe_intermediate_size") or 0),
        experts=int(cfg.get("num_experts") or 0),
        experts_per_token=int(cfg.get("num_experts_per_tok") or 0),
        full_attention_layers=full,
        linear_layers=linear,
        linear_key_heads=int(cfg["linear_num_key_heads"]),
        linear_value_heads=int(cfg["linear_num_value_heads"]),
        linear_key_dim=int(cfg["linear_key_head_dim"]),
        linear_value_dim=int(cfg["linear_value_head_dim"]),
        conv_kernel=int(cfg["linear_conv_kernel_dim"]),
    )


def _layer_types_from_interval(cfg: dict[str, Any]) -> list[str]:
    """Полное внимание — каждый interval-й слой (1-based), остальные линейные."""
    interval = cfg.get("full_attention_interval")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        return []
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(int(cfg["num_hidden_layers"]))
    ]
