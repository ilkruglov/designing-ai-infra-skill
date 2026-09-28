"""Единый вид результата: формула, входные данные с единицами, число, якорь на книгу."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BOUNDS = {"lower": "нижняя граница", "upper": "верхняя граница"}


@dataclass(frozen=True)
class Result:
    """Одно вычисленное число и всё, что нужно, чтобы его проверить.

    value равно None, когда калькулятор отказывается считать; причина — в notes.
    bound отмечает оценку-границу: «lower» — реальное значение не меньше,
    «upper» — не больше. input_units задаёт единицу каждого входа: ключи
    совпадают с inputs, пустая строка — безразмерное число или метка.
    """

    name: str
    value: float | None
    unit: str
    formula: str
    inputs: dict[str, Any]
    anchor: str
    bound: str | None = None
    notes: tuple[str, ...] = ()
    input_units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bound is not None and self.bound not in BOUNDS:
            raise ValueError(
                f"bound должен быть одним из {', '.join(BOUNDS)} или None: {self.bound!r}"
            )
        if set(self.inputs) != set(self.input_units):
            missing = sorted(set(self.inputs) - set(self.input_units))
            extra = sorted(set(self.input_units) - set(self.inputs))
            raise ValueError(
                f"у результата {self.name} единицы входов не совпадают с входами: "
                f"без единицы {missing or '—'}, лишние {extra or '—'}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        bound = f" ({BOUNDS[self.bound]})" if self.bound else ""
        if self.value is None:
            shown = "не вычисляется"
        else:
            shown = f"{format_number(self.value)} {self.unit}".rstrip()
        lines = [f"**{self.name}**{bound}: {shown}", f"- формула: `{self.formula}`"]
        if self.inputs:
            shown_inputs = ", ".join(
                f"{key}={_shown(value)} {self.input_units[key]}".rstrip()
                for key, value in self.inputs.items()
            )
            lines.append(f"- входные данные: {shown_inputs}")
        lines.append(f"- источник: `{self.anchor}`")
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def format_number(value: float) -> str:
    """Число для текста: целые полностью с разрядами, дробные — 6 значащих цифр."""
    if isinstance(value, int) and not isinstance(value, bool):
        # целые — полностью, с разрядами через пробел, как в книге: 8 190 735 360
        return f"{value:,}".replace(",", " ")
    return f"{value:.6g}"


def _shown(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    return str(value)
