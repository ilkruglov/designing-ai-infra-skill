"""Параметры, память весов, KV и фиксированное состояние (глава 2).

Формулы перенесены из calculations/src/infra_calc/models оригинала и сверены с
результатами автора: qwen3 и llama — плотные, qwen3_moe — MoE,
deepseek_v3 — MLA с MoE, qwen3.5 — гибридное линейное внимание.
"""

from __future__ import annotations

from .model import ModelSpec, UnsupportedArchitecture


def embedding_parameters(spec: ModelSpec) -> int:
    return spec.vocab * spec.hidden


def _head_parameters(spec: ModelSpec) -> int:
    return 0 if spec.tied_embeddings else spec.vocab * spec.hidden


def _attention_matmul(spec: ModelSpec) -> int:
    h = spec.hidden
    if spec.family == "mla_moe":
        qk = spec.qk_nope_dim + spec.qk_rope_dim
        if spec.q_lora_rank:
            q = h * spec.q_lora_rank + spec.q_lora_rank * spec.heads * qk
        else:
            q = h * spec.heads * qk
        kv_down = h * (spec.kv_lora_rank + spec.qk_rope_dim)
        kv_up = spec.kv_lora_rank * spec.heads * (spec.qk_nope_dim + spec.v_head_dim)
        return q + kv_down + kv_up + spec.heads * spec.v_head_dim * h
    q_and_o = 2 * h * spec.heads * spec.head_dim
    return q_and_o + 2 * h * spec.kv_heads * spec.head_dim


def _layer_norms(spec: ModelSpec) -> int:
    norms = 2 * spec.hidden
    if spec.qk_norm:
        norms += 2 * spec.head_dim
    if spec.family == "mla_moe":
        norms += spec.q_lora_rank + spec.kv_lora_rank
    return norms


def _is_dense_mlp_layer(spec: ModelSpec, layer: int) -> bool:
    return spec.family == "dense" or (
        spec.family == "mla_moe" and layer < spec.first_dense_layers
    )


def _mlp_matmul(spec: ModelSpec, layer: int, active: bool) -> int:
    h = spec.hidden
    if _is_dense_mlp_layer(spec, layer):
        return 3 * h * spec.intermediate
    routed = spec.experts_per_token if active else spec.experts
    expert = 3 * h * spec.moe_intermediate
    return (routed + spec.shared_experts) * expert + h * spec.experts


def _layer_extra(spec: ModelSpec, layer: int) -> int:
    if spec.family == "mla_moe" and not _is_dense_mlp_layer(spec, layer):
        return spec.experts
    return 0


def parameter_count(spec: ModelSpec, active: bool = False) -> int:
    """Логические параметры основной модели по config.json.

    Слои MTP (num_nextn_predict_layers) не считаются: DeepSeek-V3 даёт 671B,
    как logical_base_parameters автора, хотя чекпойнт с MTP весит около 685B.
    """
    if spec.family == "hybrid_linear":
        raise UnsupportedArchitecture(
            spec.model_type,
            ["linear_attention"],
            "Число параметров гибридной модели возьмите из карточки модели и передайте аргументом --params.",
        )
    total = embedding_parameters(spec) + _head_parameters(spec) + spec.hidden
    for layer in range(spec.layers):
        total += _attention_matmul(spec) + _layer_norms(spec)
        total += _mlp_matmul(spec, layer, active) + _layer_extra(spec, layer)
    return total


def weight_bytes(
    spec: ModelSpec, bytes_per_param: float = 2.0, active: bool = False
) -> int:
    return int(parameter_count(spec, active) * bytes_per_param)


def kv_bytes_per_token(spec: ModelSpec, kv_bytes: float = 2.0) -> int:
    if spec.family == "mla_moe":
        return int(spec.layers * (spec.kv_lora_rank + spec.qk_rope_dim) * kv_bytes)
    layers = (
        spec.full_attention_layers if spec.family == "hybrid_linear" else spec.layers
    )
    return int(2 * layers * spec.kv_heads * spec.head_dim * kv_bytes)


def kv_resident_bytes(
    spec: ModelSpec, context_tokens: int, kv_bytes: float = 2.0
) -> int:
    if context_tokens < 0:
        raise ValueError(
            f"context_tokens не может быть отрицательным: {context_tokens}"
        )
    tokens = min(context_tokens, spec.window) if spec.window else context_tokens
    return kv_bytes_per_token(spec, kv_bytes) * tokens


def fixed_state_bytes(
    spec: ModelSpec, state_bytes: float = 4.0, conv_bytes: float = 2.0
) -> tuple[int, int]:
    if spec.family != "hybrid_linear":
        return (0, 0)
    recurrent = (
        spec.linear_layers
        * spec.linear_value_heads
        * spec.linear_key_dim
        * spec.linear_value_dim
    )
    conv_channels = (
        2 * spec.linear_key_heads * spec.linear_key_dim
        + spec.linear_value_heads * spec.linear_value_dim
    )
    conv = spec.linear_layers * conv_channels * spec.conv_kernel
    return (int(recurrent * state_bytes), int(conv * conv_bytes))


def decode_weight_read_bytes(spec: ModelSpec, bytes_per_param: float = 2.0) -> int:
    """Чтение весов за один шаг decode одного запроса.

    Без общих весов таблица эмбеддингов читается одной строкой на токен и
    исключается, а словарная голова читается целиком. При общих весах
    (tie_word_embeddings) это одна матрица: голова читает её целиком на каждом
    шаге, поэтому она остаётся в чтении и объём равен случаю без общих весов.
    Для MoE учитываются активные эксперты одного токена; при батче читается
    объединение экспертов, которое эта функция не моделирует.
    """
    lookup_only = 0 if spec.tied_embeddings else embedding_parameters(spec)
    return int((parameter_count(spec, active=True) - lookup_only) * bytes_per_param)
