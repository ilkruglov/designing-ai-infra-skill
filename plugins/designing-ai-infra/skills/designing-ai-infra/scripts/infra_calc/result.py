"""Единый вид результата: формула, входные данные, число, якорь на книгу."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BOUNDS = {"lower": "нижняя граница", "upper": "верхняя граница"}


@dataclass(frozen=True)
class Result:
    """Одно вычисленное число и всё, что нужно, чтобы его проверить.

    value равно None, когда калькулятор отказывается считать; причина — в notes.
    bound отмечает оценку-границу: «lower» — реальное значение не меньше,
    «upper» — не больше.
    """

    name: str
    value: float | None
    unit: str
    formula: str
    inputs: dict[str, Any]
    anchor: str
    bound: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.bound is not None and self.bound not in BOUNDS:
            raise ValueError(
                f"bound должен быть одним из {', '.join(BOUNDS)} или None: {self.bound!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        bound = f" ({BOUNDS[self.bound]})" if self.bound else ""
        if self.value is None:
            shown = "не вычисляется"
        else:
            shown = f"{_number(self.value)} {self.unit}".rstrip()
        lines = [f"**{self.name}**{bound}: {shown}", f"- формула: `{self.formula}`"]
        if self.inputs:
            shown_inputs = ", ".join(f"{k}={_shown(v)}" for k, v in self.inputs.items())
            lines.append(f"- входные данные: {shown_inputs}")
        lines.append(f"- источник: `{self.anchor}`")
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def _number(value: float) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        # целые — полностью, с разрядами через пробел, как в книге: 8 190 735 360
        return f"{value:,}".replace(",", " ")
    return f"{value:.6g}"


def _shown(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number(value)
    return str(value)
