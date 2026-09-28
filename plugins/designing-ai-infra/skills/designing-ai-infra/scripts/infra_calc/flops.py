"""Матричные FLOPs прямого прохода и обучения (главы 1.3, 2.1, 3.4.3).

Причинное внимание учитывает n·history + n(n+1)/2 пар «запрос — ключ»,
QKᵀ и AV дают 4 × heads × head_dim FLOPs на пару. Словарная голова по
умолчанию считается только для последнего токена (инференс); при обучении
— для всех токенов.
"""

from __future__ import annotations

from collections.abc import Callable

from .accounting import _attention_matmul, _mlp_matmul
from .model import ModelSpec, UnsupportedArchitecture

HEAD_TOKENS: dict[str, Callable[[int], int]] = {
    "last": lambda n: 1,
    "all": lambda n: n,
    "none": lambda n: 0,
}


def _require_standard_attention(spec: ModelSpec) -> None:
    if spec.family not in ("dense", "moe") or spec.window:
        raise UnsupportedArchitecture(
            spec.model_type,
            ["attention_path"],
            "FLOPs для MLA, окна и гибридного внимания зависят от пути выполнения; "
            "используйте команды автора v3-forward, v41-forward, qwen35-forward.",
        )


def _check_tokens(new_tokens: int, history: int) -> None:
    if new_tokens < 1:
        raise ValueError(f"new_tokens должен быть не меньше 1: {new_tokens}")
    if history < 0:
        raise ValueError(f"history не может быть отрицательным: {history}")


def attention_matrix_flops(spec: ModelSpec, new_tokens: int, history: int = 0) -> int:
    _require_standard_attention(spec)
    _check_tokens(new_tokens, history)
    pairs = new_tokens * history + new_tokens * (new_tokens + 1) // 2
    return spec.layers * 4 * spec.heads * spec.head_dim * pairs


def forward_matrix_flops(
    spec: ModelSpec,
    new_tokens: int,
    history: int = 0,
    batch: int = 1,
    output_head: str = "last",
) -> int:
    _require_standard_attention(spec)
    _check_tokens(new_tokens, history)
    if batch < 1:
        raise ValueError(f"batch должен быть не меньше 1: {batch}")
    if output_head not in HEAD_TOKENS:
        raise ValueError(f"output_head должен быть одним из {', '.join(HEAD_TOKENS)}")
    per_token = sum(
        _attention_matmul(spec) + _mlp_matmul(spec, layer, active=True)
        for layer in range(spec.layers)
    )
    linear = 2 * per_token * new_tokens
    head = 2 * spec.vocab * spec.hidden * HEAD_TOKENS[output_head](new_tokens)
    return batch * (linear + head + attention_matrix_flops(spec, new_tokens, history))


def training_matrix_flops(spec: ModelSpec, tokens: int) -> tuple[int, int, int]:
    forward = forward_matrix_flops(spec, new_tokens=tokens, output_head="all")
    return forward, 2 * forward, 3 * forward


def six_nd_flops(parameters: int, tokens: int) -> int:
    return 6 * parameters * tokens
