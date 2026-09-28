"""Спекулятивное декодирование (глава 8.5.2)."""

from __future__ import annotations

from .checks import require_non_negative, require_positive


def expected_tokens_per_round(acceptance: float, draft_tokens: int) -> float:
    """E[N] = Σ_{i=0..k} a^i при независимом принятии позиций черновика."""
    if not 0 <= acceptance <= 1:
        raise ValueError(f"доля принятия должна лежать в [0, 1]: {acceptance}")
    require_non_negative("draft_tokens", draft_tokens)
    return sum(acceptance**i for i in range(draft_tokens + 1))


def mean_time_per_token(round_seconds: list[float], round_tokens: list[int]) -> float:
    """Среднее по токенам, а не по раундам: сумма времени на сумму токенов."""
    if not round_seconds:
        raise ValueError("нужен хотя бы один раунд")
    if len(round_seconds) != len(round_tokens):
        raise ValueError(
            f"длины round_seconds и round_tokens различаются: "
            f"{len(round_seconds)} и {len(round_tokens)}"
        )
    for seconds, tokens in zip(round_seconds, round_tokens):
        require_non_negative("время раунда", seconds)
        if tokens < 1:
            raise ValueError(f"раунд даёт не меньше одного токена: {tokens}")
    return sum(round_seconds) / sum(round_tokens)


def breakeven_round_seconds(expected_tokens: float, plain_step_seconds: float) -> float:
    """Раунд выгоден, пока он короче E[N] обычных шагов decode."""
    if expected_tokens < 1:
        raise ValueError(f"E[N] не может быть меньше 1: {expected_tokens}")
    require_positive("plain_step_seconds", plain_step_seconds)
    return expected_tokens * plain_step_seconds
