"""Параметры, память весов, KV и фиксированное состояние (глава 2).

Формулы перенесены из calculations/src/infra_calc/models оригинала и сверены с
результатами автора: qwen3 и llama — плотные, qwen3_moe — MoE,
deepseek_v3 — MLA с MoE, qwen3.5 — гибридное линейное внимание.
"""

from __future__ import annotations

from .checks import require_finite, require_int_at_least
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
            "Число параметров гибридной модели возьмите из карточки модели; "
            "в CLI передайте его как `calc.py model --params`.",
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
    require_int_at_least("context_tokens", context_tokens, 0)
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


def _check_tp(spec: ModelSpec, tp: int) -> None:
    """Головы — минимальная единица распределения TP (глава 6.2.2)."""
    require_int_at_least("tp", tp, 1)
    if spec.heads % tp:
        raise ValueError(
            f"TP={tp} не делит {spec.heads} голов внимания {spec.model_type}: "
            "голова — минимальная единица распределения (глава 6.2.2)"
        )
    if spec.family != "mla_moe":
        kv = spec.kv_heads
        if (tp <= kv and kv % tp) or (tp > kv and tp % kv):
            raise ValueError(
                f"TP={tp} и {kv} голов KV {spec.model_type}: головы KV не "
                "распределяются по картам поровну"
            )
    if spec.family == "hybrid_linear":
        for label, count in (
            ("линейных голов ключей", spec.linear_key_heads),
            ("линейных голов значений", spec.linear_value_heads),
        ):
            if count % tp:
                raise ValueError(
                    f"TP={tp} не делит {count} {label} {spec.model_type}: состояние "
                    "линейных слоёв не делится по картам поровну"
                )


def tp_kv_divisor(spec: ModelSpec, tp: int) -> int:
    """Во сколько раз KV на токен одной карты TP меньше KV всей модели (глава 6.2.2).

    GQA и MHA делят KV по головам KV: min(TP, kv_heads); при TP больше числа
    голов KV каждая голова хранится на TP/kv_heads картах. Латентный KV MLA —
    один вектор на токен для всех голов (глава 2.3.2), поэтому каждая карта TP
    хранит его целиком: делитель 1.
    """
    _check_tp(spec, tp)
    if spec.family == "mla_moe":
        return 1
    return min(tp, spec.kv_heads)


def tp_fixed_state_divisor(spec: ModelSpec, tp: int) -> int:
    """Делитель фиксированного состояния линейных слоёв на карту TP.

    Рекуррентное и свёрточное состояние гибридной модели относится к линейным
    головам и делится вместе с ними; таблица главы 6.2.2 (31/117 запросов
    Qwen3.6 на двух H100) получается именно при таком делении.
    """
    _check_tp(spec, tp)
    return tp if spec.family == "hybrid_linear" else 1


def expected_active_experts(experts: int, top_k: int, tokens: int) -> float:
    """E[E_active] = E[1 − (1 − k/E)^m], формула (6-8).

    Каждый из m токенов независимо и равномерно выбирает k экспертов из E.
    Реальная маршрутизация неравномерна: объединение лежит между k
    (все токены выбрали одних экспертов) и min(m·k, E).
    """
    require_int_at_least("experts", experts, 1)
    require_int_at_least("top_k", top_k, 1)
    require_int_at_least("tokens", tokens, 1)
    if top_k > experts:
        raise ValueError(f"top_k больше числа экспертов: {top_k} > {experts}")
    return experts * (1 - (1 - top_k / experts) ** tokens)


def batch_decode_weight_read_bytes(
    spec: ModelSpec, experts_per_layer: float, bytes_per_param: float = 2.0
) -> int:
    """Чтение весов за шаг decode батча MoE: R = R_общие + U·L_MoE·b_W·N_e (глава 2.4.2).

    U — различных маршрутизируемых экспертов на слой в батче; при U = k это
    decode_weight_read_bytes. Каждый выбранный эксперт читается один раз на шаг.
    """
    if not spec.experts:
        raise ValueError(
            f"{spec.model_type} — не MoE: чтение весов за шаг не зависит от батча"
        )
    require_finite("experts_per_layer", experts_per_layer)
    if not spec.experts_per_token <= experts_per_layer <= spec.experts:
        raise ValueError(
            f"экспертов на слой должно быть от k = {spec.experts_per_token} до "
            f"E = {spec.experts}: {experts_per_layer}"
        )
    moe_layers = sum(
        1 for layer in range(spec.layers) if not _is_dense_mlp_layer(spec, layer)
    )
    expert = 3 * spec.hidden * spec.moe_intermediate
    lookup_only = 0 if spec.tied_embeddings else embedding_parameters(spec)
    extra = (experts_per_layer - spec.experts_per_token) * moe_layers * expert
    parameters = parameter_count(spec, active=True) - lookup_only + extra
    return int(parameters * bytes_per_param)
