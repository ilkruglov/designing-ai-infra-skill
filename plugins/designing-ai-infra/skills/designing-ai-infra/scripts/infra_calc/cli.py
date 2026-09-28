"""CLI калькуляторов. Вывод: JSON или Markdown с формулой, входами и якорем.

Каждый результат несёт формулу, входные данные с единицами, тип границы и
якорь на заголовок книги. Если число взято из снимка железа, вывод называет
снимок; стойка или сервер из снимка берутся только с флагом --allow-aggregate.
Ошибка входа — и разбора аргументов, и проверки значений — завершает команду
с кодом 2 и сообщением на русском, которое называет флаг CLI.

collectives.all_to_all_phase и collectives.tp_step_seconds вызываются из
Python: матрицу назначений неудобно передавать аргументами.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from . import (
    accounting,
    collectives,
    cost,
    edge,
    flops,
    hardware,
    queueing,
    roofline,
    serving,
    speculative,
    training,
    units,
)
from .checks import (
    require_finite,
    require_int_at_least,
    require_non_negative,
    require_positive,
)
from .model import ModelSpec, UnsupportedArchitecture, load_config, parse_spec
from .result import Result

BOOK = "references/source-book"
A_CH1_KEYS = f"{BOOK}/chapter1.md:152"
A_CH1_UNITS = f"{BOOK}/chapter1.md:189"
A_CH1_TIME = f"{BOOK}/chapter1.md:263"
A_CH1_REF = f"{BOOK}/chapter1.md:450"
A_CH2_MEM = f"{BOOK}/chapter2.md:236"
A_CH2_HYBRID = f"{BOOK}/chapter2.md:418"
A_CH3_QUEUE = f"{BOOK}/chapter3.md:75"
A_CH3_PRICE = f"{BOOK}/chapter3.md:209"
A_CH3_STATE = f"{BOOK}/chapter3.md:384"
A_CH3_TRAIN = f"{BOOK}/chapter3.md:442"
A_CH6_RING = f"{BOOK}/chapter6.md:473"
A_CH7_TIME = f"{BOOK}/chapter7.md:938"
A_CH8_MEM = f"{BOOK}/chapter8.md:52"
A_CH8_SPEC = f"{BOOK}/chapter8.md:536"
A_CH10_ZERO = f"{BOOK}/chapter10.md:151"
A_CH10_PIPE = f"{BOOK}/chapter10.md:322"
A_CH10_CKPT = f"{BOOK}/chapter10.md:553"
A_CH11_LITTLE = f"{BOOK}/chapter11.md:62"
A_CH11_CALL = f"{BOOK}/chapter11.md:488"
A_CH11_TASK = f"{BOOK}/chapter11.md:515"
A_CH12_EDGE = f"{BOOK}/chapter12.md:11"

SINGLE = "single_device"
USER = "задано пользователем"
BF16_BYTES = 2.0
ZERO_FORMULAS = (
    "P × (2 + 2 + 12)",
    "P × (2 + 2 + 12/N)",
    "P × (2 + (2 + 12)/N)",
    "P × (2 + 2 + 12) / N",
)
QUANT_NOTE = (
    "квантизованные веса: parameters × bytes не учитывает scale групп и части "
    "модели, оставленные в BF16; для 8-битной DeepSeek-R1-Distill-Llama-70B книга "
    "даёт 73.73 GB против 70.55 GB по этой формуле (глава 1.2.3); известные "
    "накладные расходы передайте через --quant-overhead-bytes"
)
QUANT_READ_NOTE = (
    "квантизованные веса: scale групп и части модели в BF16 в чтение не включены"
)
RATES_SUFFIX = "; граница предполагает идеальное деление работы между устройствами"

# Единица каждого входа результата. Пустая строка у числа допустима только для
# безразмерных величин из DIMENSIONLESS; у строк и флагов это метка.
DIMENSIONLESS = frozenset(
    {
        "params",
        "parameters",
        "active_parameters",
        "device_count",
        "batch",
        "dp",
        "stage",
        "devices",
        "mfu",
        "stages",
        "microbatches",
        "acceptance",
        "calls",
        "success",
    }
)
LABELS = frozenset(
    {
        "config",
        "model_type",
        "wrapper_model_type",
        "weight_dtype",
        "kv_dtype",
        "device",
        "name",
        "scope",
        "shared_with_cpu",
        "precision",
        "accumulator",
        "sparsity",
        "peak_source",
        "bandwidth_source",
        "memory_source",
        "size_text",
        "dtype",
        "classes",
        "rates",
    }
)
INPUT_UNITS: dict[str, str] = {
    **dict.fromkeys(DIMENSIONLESS | LABELS, ""),
    "context": "tok",
    "tokens": "tok",
    "total_tokens": "tok",
    "draft": "tok",
    "input": "tok",
    "output": "tok",
    "cached": "tok",
    "cache_write": "tok",
    "flops": "FLOP",
    "decode_flops": "FLOP",
    "prefill_flops": "FLOP",
    "total_flops": "FLOP",
    "bytes": "B",
    "weights": "B",
    "weight_read": "B",
    "memory": "B",
    "reserve": "B",
    "kv_request": "B",
    "message": "B",
    "upload": "B",
    "download": "B",
    "size_bytes": "B",
    "quant_overhead": "B",
    "kv_per_token": "B/tok",
    "peak": "FLOP/s",
    "bandwidth": "B/s",
    "tokens_per_second": "tok/s",
    "prefill_capacity": "tok/s",
    "input_tokens_per_second": "tok/s",
    "decode_capacity": "step/s",
    "decode_steps_per_second": "step/s",
    "price_per_hour": "$/h",
    "cost_per_call": "$",
    "input_price": "$/Mtok",
    "output_price": "$/Mtok",
    "cache_read_price": "$/Mtok",
    "cache_write_price": "$/Mtok",
    "save_seconds": "s",
    "recovery": "s",
    "interval": "s",
    "plain_step": "s",
    "alpha": "s",
    "rtt": "s",
    "compute": "s",
    "time_in_system": "s",
    "failure_rate": "1/s",
    "arrival_rate": "1/s",
    "mbps": "Mbit/s",
    "up_mbps": "Mbit/s",
    "down_mbps": "Mbit/s",
}


def _result(
    name: str,
    value: float | None,
    unit: str,
    formula: str,
    inputs: dict[str, Any],
    anchor: str,
    *,
    bound: str | None = None,
    notes: Iterable[str] = (),
) -> Result:
    unknown = sorted(key for key in inputs if key not in INPUT_UNITS)
    if unknown:
        # ошибка программы, а не входа: не превращается в код 2
        raise RuntimeError(f"нет единицы для входов {unknown} результата {name}")
    return Result(
        name,
        value,
        unit,
        formula,
        inputs,
        anchor,
        bound=bound,
        notes=tuple(notes),
        input_units={key: INPUT_UNITS[key] for key in inputs},
    )


# --- проверка входов по флагам ----------------------------------------------


def _attribute(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _check(
    args: argparse.Namespace,
    *,
    positive: Sequence[str] = (),
    non_negative: Sequence[str] = (),
    at_least: Sequence[tuple[str, int]] = (),
) -> None:
    """Проверить заданные флаги до вызова калькулятора, чтобы ошибка называла флаг."""
    for flag in positive:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_positive(flag, value)
    for flag in non_negative:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_non_negative(flag, value)
    for flag, minimum in at_least:
        value = getattr(args, _attribute(flag))
        if value is not None:
            require_int_at_least(flag, value, minimum)


def _check_fraction(flag: str, value: float | None, zero_allowed: bool) -> None:
    if value is None:
        return
    require_finite(flag, value)
    if not 0 <= value <= 1 or (value == 0 and not zero_allowed):
        interval = "[0, 1]" if zero_allowed else "(0, 1]"
        raise ValueError(f"{flag} должен лежать в {interval}: {value}")


# --- общие части ------------------------------------------------------------


def _load_spec(path: str) -> ModelSpec:
    """config.json → ModelSpec; ошибки чтения и разбора — ValueError на русском."""
    try:
        config = load_config(path)
    except FileNotFoundError:
        raise ValueError(f"файл конфигурации не найден: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} — не JSON: {error.msg}, строка {error.lineno}"
        ) from None
    except OSError as error:
        raise ValueError(f"не удалось прочитать {path}: {error.strerror}") from None
    if not isinstance(config, dict):
        # ValueError, а не TypeError: все отказы входа CLI сообщает одним типом
        raise ValueError(f"{path}: config.json должен быть объектом JSON")  # noqa: TRY004
    try:
        return parse_spec(config)
    except UnsupportedArchitecture:
        raise
    except KeyError as error:
        raise ValueError(f"в {path} нет обязательного поля {error.args[0]!r}") from None
    except (TypeError, ValueError) as error:
        raise ValueError(f"в {path} поле неверного типа: {error}") from None


def _wrapper_notes(spec: ModelSpec) -> tuple[str, ...]:
    if spec.wrapper_model_type is None:
        return ()
    note = (
        f"config — обёртка {spec.wrapper_model_type} над текстовой моделью "
        f"{spec.model_type}: веса энкодеров (vision/audio) и проектора не учтены"
    )
    return (note,)


def _aggregate_note(
    dev: hardware.Device | None, labels: Sequence[str], suffix: str = ""
) -> tuple[str, ...]:
    """Примечание об агрегате: только для величин, взятых из снимка."""
    if dev is None or dev.scope == SINGLE or not labels:
        return ()
    n = dev.device_count
    return (
        (
            f"{dev.id} — агрегат {dev.scope} из {n} устройств; сумма по {n} "
            f"устройствам, а не значение одной карты: {', '.join(labels)}{suffix}"
        ),
    )


def _device(args: argparse.Namespace, per_device: str) -> hardware.Device | None:
    if not args.device:
        return None
    dev = hardware.device(args.device, allow_aggregate=True)
    if dev.scope != SINGLE and not args.allow_aggregate:
        raise ValueError(
            f"{dev.id} в снимке {hardware.SNAPSHOT} — агрегат {dev.scope} из "
            f"{dev.device_count} устройств: ёмкость, пропускная способность и пики "
            f"суммированы по {dev.device_count} устройствам. Для одной карты выберите "
            "одиночное устройство или задайте значения на одно устройство без "
            f"--device: {per_device}; агрегат целиком — флаг --allow-aggregate"
        )
    return dev


def _snapshot_peak(dev: hardware.Device, args: argparse.Namespace) -> float:
    try:
        return hardware.peak_flops(
            dev, args.precision, sparsity=args.sparsity, accumulator=args.accumulator
        )
    except ValueError as error:
        raise ValueError(
            f"{error}. Пик выбирается из снимка по --precision {args.precision}, "
            f"--accumulator {args.accumulator} и --sparsity {args.sparsity}: "
            "укажите накопление из списка доступных через --accumulator "
            "или задайте пик явно через --peak-tflops"
        ) from None


@dataclass(frozen=True)
class _Quantity:
    """Величина для расчёта: значение, входы для отчёта и откуда она взята."""

    value: float
    inputs: dict[str, Any]
    snapshot_label: str | None = None
    notes: tuple[str, ...] = ()


def _labels(*quantities: _Quantity) -> list[str]:
    return [q.snapshot_label for q in quantities if q.snapshot_label]


def _peak(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.peak_tflops is not None:
        require_positive("--peak-tflops", args.peak_tflops)
        peak = args.peak_tflops * 1e12
        return _Quantity(peak, {"peak": peak, "peak_source": USER})
    if dev is None:
        raise ValueError("укажите --device или --peak-tflops")
    peak = _snapshot_peak(dev, args)
    inputs = {
        "device": dev.id,
        "precision": args.precision,
        "accumulator": args.accumulator,
        "sparsity": args.sparsity,
        "peak": peak,
    }
    return _Quantity(peak, inputs, "пик")


def _bandwidth(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.bandwidth is not None:
        require_positive("--bandwidth", args.bandwidth)
        return _Quantity(
            args.bandwidth, {"bandwidth": args.bandwidth, "bandwidth_source": USER}
        )
    if dev is None:
        raise ValueError("укажите --device или --bandwidth")
    value = hardware.bandwidth(dev)
    return _Quantity(
        value, {"device": dev.id, "bandwidth": value}, "пропускная способность памяти"
    )


def _memory(args: argparse.Namespace, dev: hardware.Device | None) -> _Quantity:
    if args.memory is not None:
        return _Quantity(args.memory, {"memory": args.memory, "memory_source": USER})
    if dev is None:
        raise ValueError(
            "укажите --memory (байт памяти одного устройства) или --device"
        )
    if dev.memory_bytes is None:
        raise ValueError(
            f"у {dev.id} ёмкость памяти не опубликована в снимке {hardware.SNAPSHOT}; "
            "передайте --memory"
        )
    notes = [
        (
            "номинальная ёмкость по этикетке, а не доступный среде выполнения объём; "
            "резерв среды выполнения задайте через --reserve"
        )
    ]
    if dev.shared_with_cpu:
        notes.append("память общая с CPU и ОС: модели доступна только её часть")
    return _Quantity(
        dev.memory_bytes,
        {"memory": dev.memory_bytes, "memory_source": dev.id},
        "ёмкость памяти",
        tuple(notes),
    )


# --- команды -------------------------------------------------------------------


def _weight_estimate(
    args: argparse.Namespace, bytes_per_param: float
) -> tuple[int, str | None, tuple[str, ...]]:
    """Добавка к «параметры × байты», тип границы и примечания для квантизации."""
    quantized = bytes_per_param < BF16_BYTES
    overhead = args.quant_overhead_bytes
    if overhead is not None and not quantized:
        raise ValueError(
            "--quant-overhead-bytes задаёт scale и части модели в высокой точности "
            f"квантизованных весов; с --weight-dtype {args.weight_dtype} он не нужен"
        )
    if not quantized:
        return 0, None, ()
    if overhead is None:
        return 0, "lower", (QUANT_NOTE,)
    note = f"прибавлены накладные расходы квантизации из --quant-overhead-bytes: {overhead:.6g} B"
    return int(overhead), None, (note,)


def _model(args: argparse.Namespace) -> list[Result]:
    _check(args, non_negative=("--quant-overhead-bytes",), at_least=(("--context", 0),))
    spec = _load_spec(args.config)
    wb = units.dtype_bytes(args.weight_dtype)
    kb = units.dtype_bytes(args.kv_dtype)
    wrapper = _wrapper_notes(spec)
    overhead, weight_bound, weight_notes = _weight_estimate(args, wb)
    base: dict[str, Any] = {
        "config": Path(args.config).name,
        "model_type": spec.model_type,
    }
    if spec.wrapper_model_type:
        base["wrapper_model_type"] = spec.wrapper_model_type
    weights = {**base, "weight_dtype": args.weight_dtype}
    if args.quant_overhead_bytes is not None:
        weights["quant_overhead"] = args.quant_overhead_bytes
    out: list[Result] = []
    try:
        params: int | None = accounting.parameter_count(spec)
        refusal: UnsupportedArchitecture | None = None
    except UnsupportedArchitecture as error:
        params, refusal = None, error

    if params is not None:
        if args.params is not None:
            raise ValueError(
                "--params нужен только для архитектур, где калькулятор не считает "
                f"параметры; для {spec.model_type} по config.json получено {params}"
            )
        read_notes = wrapper
        if spec.experts:
            read_notes += (
                "эксперты одного токена; при батче читается объединение экспертов",
            )
        if weight_bound == "lower" or overhead:
            read_notes += (QUANT_READ_NOTE,)
        out += [
            _result(
                "parameters",
                params,
                "",
                "Σ матриц + нормализации + эмбеддинги (без MTP)",
                base,
                A_CH3_STATE,
                notes=wrapper,
            ),
            _result(
                "active_parameters",
                accounting.parameter_count(spec, active=True),
                "",
                "эксперты одного токена вместо всех",
                base,
                A_CH2_MEM,
                notes=wrapper,
            ),
            _result(
                "weight_bytes",
                accounting.weight_bytes(spec, wb) + overhead,
                "B",
                "parameters × bytes_per_param"
                + (" + quant_overhead" if overhead else ""),
                weights,
                A_CH1_UNITS,
                bound=weight_bound,
                notes=(*wrapper, *weight_notes),
            ),
            _result(
                "decode_weight_read_bytes",
                accounting.decode_weight_read_bytes(spec, wb),
                "B",
                "(active_parameters − таблица эмбеддингов без общих весов) × bytes_per_param",
                {k: v for k, v in weights.items() if k != "quant_overhead"},
                A_CH2_MEM,
                bound="lower" if wb < BF16_BYTES else None,
                notes=read_notes,
            ),
        ]
    elif args.params is not None and refusal is not None:
        given = {**base, "params": args.params}
        param_notes: tuple[str, ...] = (
            (
                f"калькулятор не считает параметры {refusal.model_type} "
                f"(поля: {', '.join(refusal.fields) or '—'}); значение взято из --params как есть"
            ),
        )
        if spec.wrapper_model_type:
            param_notes += (
                (
                    "включает ли число веса энкодеров обёртки "
                    f"{spec.wrapper_model_type}, определяет карточка модели"
                ),
            )
        out += [
            _result(
                "parameters",
                args.params,
                "",
                f"{USER} (--params)",
                given,
                A_CH3_STATE,
                notes=param_notes,
            ),
            _result(
                "weight_bytes",
                int(args.params * wb) + overhead,
                "B",
                f"parameters ({USER}) × bytes_per_param"
                + (" + quant_overhead" if overhead else ""),
                given | {k: v for k, v in weights.items() if k not in base},
                A_CH1_UNITS,
                bound=weight_bound,
                notes=(*param_notes[1:], *weight_notes),
            ),
        ]
    else:
        out.append(
            _result(
                "parameters",
                None,
                "",
                "не вычисляется",
                base,
                A_CH3_STATE,
                notes=(str(refusal), *wrapper),
            )
        )

    kv_formula = {
        "mla_moe": "layers × (kv_lora_rank + qk_rope_dim) × bytes",
        "hybrid_linear": "2 × full_attention_layers × kv_heads × head_dim × bytes",
    }.get(spec.family, "2 × layers × kv_heads × head_dim × bytes")
    kv_inputs = {**base, "kv_dtype": args.kv_dtype}
    out.append(
        _result(
            "kv_bytes_per_token",
            accounting.kv_bytes_per_token(spec, kb),
            "B",
            kv_formula,
            kv_inputs,
            A_CH8_MEM,
        )
    )
    if args.context:
        resident = accounting.kv_resident_bytes(spec, args.context, kb)
        window = f"min(context, {spec.window})" if spec.window else "context"
        out.append(
            _result(
                "kv_resident_bytes",
                resident,
                "B",
                f"kv_bytes_per_token × {window}",
                kv_inputs | {"context": args.context},
                A_CH8_MEM,
            )
        )
    recurrent, conv = accounting.fixed_state_bytes(spec)
    if recurrent or conv:
        out.append(
            _result(
                "fixed_state_bytes",
                recurrent + conv,
                "B",
                "линейные слои: рекуррентное (FP32) + свёрточное (BF16) состояние на запрос",
                base,
                A_CH2_HYBRID,
            )
        )
    if args.context:
        out += _forward_flops(spec, args.context, base)
    return out


def _forward_flops(spec: ModelSpec, context: int, base: dict[str, Any]) -> list[Result]:
    try:
        prefill = flops.forward_matrix_flops(spec, context)
        decode = flops.forward_matrix_flops(spec, 1, context - 1)
    except UnsupportedArchitecture as error:
        return [
            _result(
                "prefill_flops",
                None,
                "FLOP",
                "не вычисляется",
                base,
                A_CH1_REF,
                notes=(str(error),),
            )
        ]
    return [
        _result(
            "prefill_flops",
            prefill,
            "FLOP",
            "2·W_active·n + 2·V·h + 4·L·H·d·n(n+1)/2",
            base | {"tokens": context},
            A_CH1_REF,
        ),
        _result(
            "decode_step_flops",
            decode,
            "FLOP",
            "2·W_active + 2·V·h + 4·L·H·d·context",
            base | {"context": context},
            A_CH1_REF,
        ),
    ]


def _roofline(args: argparse.Namespace) -> list[Result]:
    _check(args, non_negative=("--flops", "--bytes"))
    dev = _device(args, "--peak-tflops и --bandwidth")
    peak = _peak(args, dev)
    bw = _bandwidth(args, dev)
    notes = _aggregate_note(dev, _labels(peak, bw), RATES_SUFFIX)
    inputs = {"flops": args.flops, "bytes": args.bytes, **peak.inputs, **bw.inputs}
    out = [
        _result(
            "step_lower_bound_seconds",
            roofline.lower_bound_seconds(args.flops, args.bytes, peak.value, bw.value),
            "s",
            "max(F/Π, R/β)",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
        _result(
            "compute_seconds",
            roofline.compute_seconds(args.flops, peak.value),
            "s",
            "F/Π",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
        _result(
            "memory_seconds",
            roofline.memory_seconds(args.bytes, bw.value),
            "s",
            "R/β",
            inputs,
            A_CH1_TIME,
            bound="lower",
            notes=notes,
        ),
    ]
    if args.bytes > 0:
        out.append(
            _result(
                "arithmetic_intensity",
                roofline.arithmetic_intensity(args.flops, args.bytes),
                "FLOP/B",
                "F/R",
                inputs,
                A_CH1_TIME,
            )
        )
    out.append(
        _result(
            "ridge_point",
            roofline.ridge_point(peak.value, bw.value),
            "FLOP/B",
            "Π/β",
            inputs,
            A_CH1_TIME,
            notes=notes,
        )
    )
    return out


def _serving(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--kv-per-token",),
        non_negative=(
            "--weights",
            "--weight-read",
            "--decode-flops",
            "--prefill-flops",
            "--memory",
            "--reserve",
            "--price-per-hour",
        ),
        at_least=(("--context", 1), ("--batch", 1)),
    )
    dev = _device(args, "--peak-tflops, --bandwidth и --memory")
    peak = _peak(args, dev)
    bw = _bandwidth(args, dev)
    memory = _memory(args, dev)
    rate_notes = _aggregate_note(dev, _labels(peak, bw), RATES_SUFFIX)
    memory_notes = (
        *memory.notes,
        *_aggregate_note(
            dev,
            _labels(memory),
            "; веса и KV считаются размещёнными во всём агрегате",
        ),
        "верхняя граница: активации, фрагментация и буферы сверх --reserve не учтены",
    )
    kv_request = args.kv_per_token * args.context
    memory_inputs = {
        **memory.inputs,
        "weights": args.weights,
        "kv_per_token": args.kv_per_token,
        "context": args.context,
        "reserve": args.reserve,
    }
    step = serving.tpot_lower_bound_seconds(
        args.batch,
        args.decode_flops,
        args.weight_read,
        kv_request,
        peak.value,
        bw.value,
    )
    step_inputs = {
        "batch": args.batch,
        "decode_flops": args.decode_flops,
        "weight_read": args.weight_read,
        "kv_request": kv_request,
        **peak.inputs,
        **bw.inputs,
    }
    throughput = serving.tokens_per_second(args.batch, step)
    out = [
        _result(
            "max_concurrent_requests",
            serving.max_concurrent_requests(
                memory.value, args.weights, kv_request, args.reserve
            ),
            "",
            "floor((C − M_w − reserve) / (kv_per_token × context))",
            memory_inputs,
            A_CH8_MEM,
            bound="upper",
            notes=memory_notes,
        ),
        _result(
            "tpot_lower_bound_seconds",
            step,
            "s",
            "max(B·F_decode/Π, (R_W + B·R_KV)/β)",
            step_inputs,
            A_CH1_TIME,
            bound="lower",
            notes=rate_notes,
        ),
        _result(
            "tokens_per_second_upper_bound",
            throughput,
            "tok/s",
            "B / T_step",
            step_inputs,
            A_CH1_TIME,
            bound="upper",
            notes=rate_notes,
        ),
    ]
    if args.prefill_flops is not None:
        ttft = serving.ttft_lower_bound_seconds(
            args.prefill_flops, args.weights, peak.value, bw.value
        )
        out.append(
            _result(
                "ttft_lower_bound_seconds",
                ttft,
                "s",
                "max(F_prefill/Π, M_w/β)",
                {
                    "prefill_flops": args.prefill_flops,
                    "weights": args.weights,
                    **peak.inputs,
                    **bw.inputs,
                },
                A_CH1_REF,
                bound="lower",
                notes=rate_notes,
            )
        )
    if args.price_per_hour is not None:
        price = serving.cost_per_million_tokens(args.price_per_hour, throughput)
        out.append(
            _result(
                "cost_per_million_tokens_lower_bound",
                price,
                "$",
                "price_per_hour / 3600 / tok_s × 10^6",
                {
                    "price_per_hour": args.price_per_hour,
                    "tokens_per_second": throughput,
                },
                A_CH3_PRICE,
                bound="lower",
            )
        )
    return out


def _training(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--total-tokens",),
        at_least=(("--tokens", 1), ("--dp", 1), ("--devices", 1)),
    )
    _check_fraction("--mfu", args.mfu, zero_allowed=False)
    spec = _load_spec(args.config)
    wrapper = _wrapper_notes(spec)
    _, _, total = flops.training_matrix_flops(spec, args.tokens)
    params = accounting.parameter_count(spec)
    active = accounting.parameter_count(spec, active=True)
    base = {"model_type": spec.model_type, "tokens": args.tokens}
    six_nd_notes = wrapper
    if active != params:
        six_nd_notes += (
            (
                f"MoE: N — {active} активных параметров из {params}; все параметры "
                "определяют ёмкость (веса, состояние ZeRO), активные — лишь грубую "
                "оценку вычислений"
            ),
        )
    out = [
        _result(
            "training_flops_per_sequence",
            total,
            "FLOP",
            "3 × forward, словарная голова на всех токенах",
            base,
            A_CH3_TRAIN,
            notes=wrapper,
        ),
        _result(
            "six_nd_flops_per_sequence",
            flops.six_nd_flops(active, args.tokens),
            "FLOP",
            "6 × N_active × D",
            base | {"active_parameters": active},
            A_CH3_TRAIN,
            notes=six_nd_notes,
        ),
    ]
    if args.dp is not None:
        zero_note = (
            "байт на параметр: веса BF16 2, градиенты BF16 2, оптимизатор 12 "
            "(глава 10); глава 3.4.1 считает градиенты в FP32 — 18 байт"
        )
        for stage, formula in enumerate(ZERO_FORMULAS):
            state = training.sharded_state_bytes_per_device(params, args.dp, stage)
            out.append(
                _result(
                    f"zero{stage}_state_bytes_per_gpu",
                    state,
                    "B",
                    formula,
                    {"parameters": params, "dp": args.dp, "stage": stage},
                    A_CH10_ZERO,
                    notes=(zero_note, *wrapper),
                )
            )
    duration = {
        "--total-tokens": args.total_tokens,
        "--devices": args.devices,
        "--mfu": args.mfu,
    }
    if any(value is not None for value in duration.values()):
        missing = [flag for flag, value in duration.items() if value is None]
        if missing:
            raise ValueError(
                "для срока обучения нужны --total-tokens, --devices и --mfu; "
                f"не хватает: {', '.join(missing)}"
            )
        dev = _device(args, "--peak-tflops")
        peak = _peak(args, dev)
        whole = total * args.total_tokens / args.tokens
        seconds = training.training_seconds(whole, args.devices, peak.value, args.mfu)
        inputs = {
            "total_flops": whole,
            "total_tokens": args.total_tokens,
            "devices": args.devices,
            "mfu": args.mfu,
            **peak.inputs,
        }
        note = f"FLOPs последовательности длиной {args.tokens} линейно масштабированы на все токены"
        out.append(
            _result(
                "training_seconds",
                seconds,
                "s",
                "F_total / (N · Π · MFU)",
                inputs,
                A_CH7_TIME,
                notes=(note, *_aggregate_note(dev, _labels(peak)), *wrapper),
            )
        )
    return out


def _checkpoint(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=(
            "--checkpoint-bytes",
            "--save-bandwidth",
            "--device-mtbf",
            "--interval",
        ),
        non_negative=("--recovery",),
        at_least=(("--devices", 1), ("--stages", 1), ("--microbatches", 1)),
    )
    if (args.stages is None) != (args.microbatches is None):
        raise ValueError(
            "для утилизации конвейера нужны оба значения: --stages и --microbatches"
        )
    save = args.checkpoint_bytes / args.save_bandwidth
    rate = args.devices / args.device_mtbf
    inputs = {"save_seconds": save, "failure_rate": rate, "recovery": args.recovery}
    first = training.checkpoint_optimal_interval(save, rate)
    interval = first if args.interval is None else args.interval
    loss_notes = (
        ()
        if args.interval is not None
        else ("τ — оптимум первого порядка; другой интервал задайте --interval",)
    )
    out = [
        _result(
            "first_order_optimal_interval_seconds",
            first,
            "s",
            "sqrt(2c/λ)",
            inputs,
            A_CH10_CKPT,
        ),
        _result(
            "poisson_optimal_interval_seconds",
            training.checkpoint_poisson_optimal_interval(save, rate),
            "s",
            "y − 1 + exp(−y − λc) = 0, τ = y/λ",
            inputs,
            A_CH10_CKPT,
        ),
        _result(
            "first_order_loss_at_interval",
            training.checkpoint_first_order_loss(interval, save, rate, args.recovery),
            "",
            "c/τ + λτ/2 + λr",
            inputs | {"interval": interval},
            A_CH10_CKPT,
            notes=loss_notes,
        ),
    ]
    if args.stages is not None:
        utilization = training.pipeline_utilization(args.stages, args.microbatches)
        out.append(
            _result(
                "pipeline_utilization",
                utilization,
                "",
                "m / (m + p − 1)",
                {"stages": args.stages, "microbatches": args.microbatches},
                A_CH10_PIPE,
                notes=("одинаковые стадии, передачи между стадиями не учтены",),
            )
        )
    return out


def _speculative(args: argparse.Namespace) -> list[Result]:
    _check_fraction("--acceptance", args.acceptance, zero_allowed=True)
    _check(args, positive=("--plain-step",), at_least=(("--draft", 0),))
    tokens = speculative.expected_tokens_per_round(args.acceptance, args.draft)
    inputs = {"acceptance": args.acceptance, "draft": args.draft}
    return [
        _result(
            "expected_tokens_per_round",
            tokens,
            "tok",
            "Σ a^i, i = 0..k",
            inputs,
            A_CH8_SPEC,
            notes=("позиции черновика принимаются независимо",),
        ),
        _result(
            "breakeven_round_seconds",
            speculative.breakeven_round_seconds(tokens, args.plain_step),
            "s",
            "E[N] × T_plain",
            inputs | {"plain_step": args.plain_step},
            A_CH8_SPEC,
            bound="upper",
            notes=("раунд длиннее этого времени медленнее обычного decode",),
        ),
    ]


def _ring(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--bandwidth",),
        non_negative=("--message", "--alpha"),
        at_least=(("--devices", 1),),
    )
    inputs = {
        "devices": args.devices,
        "message": args.message,
        "bandwidth": args.bandwidth,
        "alpha": args.alpha,
    }
    return [
        _result(
            "ring_allreduce_seconds",
            collectives.ring_allreduce_seconds(
                args.devices, args.message, args.bandwidth, args.alpha
            ),
            "s",
            "2(n−1)α + 2(n−1)M/(nB)",
            inputs,
            A_CH6_RING,
        ),
        _result(
            "bytes_sent_per_device",
            collectives.ring_bytes_sent_per_device(args.devices, args.message),
            "B",
            "2(n−1)M/n",
            inputs,
            A_CH6_RING,
        ),
    ]


def _cost(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--calls",),
        non_negative=(
            "--input-tokens",
            "--output-tokens",
            "--cached-tokens",
            "--cache-write-tokens",
            "--input-price",
            "--output-price",
            "--cache-read-price",
            "--cache-write-price",
        ),
    )
    _check_fraction("--success", args.success, zero_allowed=False)
    per_call = cost.call_cost(
        args.input_tokens,
        args.output_tokens,
        args.input_price,
        args.output_price,
        args.cached_tokens,
        args.cache_read_price,
        args.cache_write_tokens,
        args.cache_write_price,
    )
    tokens = {
        "input": args.input_tokens,
        "output": args.output_tokens,
        "cached": args.cached_tokens,
        "cache_write": args.cache_write_tokens,
    }
    prices = {
        "input_price": args.input_price,
        "output_price": args.output_price,
        "cache_read_price": args.cache_read_price,
        "cache_write_price": args.cache_write_price,
    }
    return [
        _result(
            "cost_per_call",
            per_call,
            "$",
            "(I_u·p_u + I_w·p_w + I_h·p_h + O·p_o) / 10^6",
            tokens | prices,
            A_CH11_CALL,
            notes=(
                "input — только некэшированный вход; хранение кэша и инструменты не входят",
            ),
        ),
        _result(
            "cost_per_accepted_task",
            cost.cost_per_accepted_task(per_call * args.calls, args.success),
            "$",
            "calls × C_call / p_success",
            {"cost_per_call": per_call, "calls": args.calls, "success": args.success},
            A_CH11_TASK,
        ),
    ]


def _edge(args: argparse.Namespace) -> list[Result]:
    _check(
        args, positive=("--up-mbps", "--down-mbps"), non_negative=("--rtt", "--compute")
    )
    up = units.megabits_per_second_to_bytes(args.up_mbps)
    down = units.megabits_per_second_to_bytes(args.down_mbps)
    upload = units.parse_bytes(args.upload)
    download = units.parse_bytes(args.download)
    inputs = {
        "upload": upload,
        "download": download,
        "up_mbps": args.up_mbps,
        "down_mbps": args.down_mbps,
        "rtt": args.rtt,
        "compute": args.compute,
    }
    seconds = edge.serial_seconds(upload, up, args.rtt, args.compute, download, down)
    return [
        _result(
            "serial_seconds",
            seconds,
            "s",
            "S_u/B_u + R + T_c + S_d/B_d",
            inputs,
            A_CH12_EDGE,
        )
    ]


def _units(args: argparse.Namespace) -> list[Result]:
    _check(args, non_negative=("--mbps",))
    if args.size is None and args.mbps is None and args.dtype is None:
        raise ValueError("укажите хотя бы одно: --size, --mbps или --dtype")
    if args.to and args.size is None:
        raise ValueError("--to переводит объём из --size; укажите --size")
    out: list[Result] = []
    if args.size is not None:
        size = units.parse_bytes(args.size)
        out.append(
            _result(
                "size_bytes",
                size,
                "B",
                "число × множитель единицы (GB = 10^9 B, GiB = 2^30 B)",
                {"size_text": args.size},
                A_CH1_UNITS,
            )
        )
        for unit in args.to or ():
            out.append(
                _result(
                    f"size_{unit}",
                    units.to_unit(size, unit),
                    unit,
                    f"size_bytes / {units.UNITS[unit]}",
                    {"size_bytes": size},
                    A_CH1_UNITS,
                )
            )
    if args.mbps is not None:
        out.append(
            _result(
                "link_bytes_per_second",
                units.megabits_per_second_to_bytes(args.mbps),
                "B/s",
                "Mbit/s × 10^6 / 8",
                {"mbps": args.mbps},
                A_CH1_UNITS,
            )
        )
    if args.dtype is not None:
        out.append(
            _result(
                "dtype_bytes",
                units.dtype_bytes(args.dtype),
                "B",
                "байт на один элемент",
                {"dtype": args.dtype},
                A_CH1_UNITS,
            )
        )
    return out


def _pairs(values: Sequence[str] | None, flag: str, example: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for text in values or ():
        name, sep, value = text.partition("=")
        if not sep or not name or not value:
            raise ValueError(
                f"{flag} ожидает имя=значение, например {example}: {text!r}"
            )
        pairs[name] = value
    return pairs


def _classes(values: Sequence[str] | None) -> dict[str, tuple[int, int]]:
    classes: dict[str, tuple[int, int]] = {}
    example = "long_input=8192:256"
    for name, value in _pairs(values, "--class", example).items():
        inputs, sep, outputs = value.partition(":")
        try:
            if not sep:
                raise ValueError(value)
            classes[name] = (int(inputs), int(outputs))
        except ValueError:
            raise ValueError(
                f"--class ожидает имя=входные:выходные токены, например {example}: "
                f"{name}={value!r}"
            ) from None
    return classes


def _rates(values: Sequence[str] | None) -> dict[str, float]:
    rates: dict[str, float] = {}
    for name, value in _pairs(values, "--rate", "long_input=2").items():
        try:
            rates[name] = float(value)
        except ValueError:
            raise ValueError(
                f"--rate ожидает интенсивность в запросах/с: {name}={value!r}"
            ) from None
    return rates


def _queueing(args: argparse.Namespace) -> list[Result]:
    _check(
        args,
        positive=("--decode-capacity", "--prefill-capacity"),
        non_negative=("--arrival-rate", "--time-in-system"),
    )
    classes = _classes(args.classes)
    rates = _rates(args.rates)
    little = {
        "--arrival-rate": args.arrival_rate,
        "--time-in-system": args.time_in_system,
    }
    if not rates and not classes and all(v is None for v in little.values()):
        raise ValueError(
            "укажите --class и --rate (потребность по классам) "
            "или --arrival-rate и --time-in-system (закон Литтла)"
        )
    out: list[Result] = []
    if rates or classes:
        if not rates or not classes:
            raise ValueError("потребность считается по --class и --rate вместе")
        tokens, steps = queueing.demand(rates, classes)
        inputs = {
            "classes": ", ".join(f"{k}={i}:{o}" for k, (i, o) in classes.items()),
            "rates": ", ".join(f"{k}={v:g}" for k, v in rates.items()),
        }
        out += [
            _result(
                "input_tokens_per_second",
                tokens,
                "tok/s",
                "Σ λ_c · I_c",
                inputs,
                A_CH3_QUEUE,
            ),
            _result(
                "decode_steps_per_second",
                steps,
                "step/s",
                "Σ λ_c · (O_c − 1)",
                inputs,
                A_CH3_QUEUE,
                notes=("первый выходной токен даёт prefill",),
            ),
        ]
        overload = "больше 1 — очередь растёт, сколько бы ни длилось окно"
        for name, demand_key, demand, key, capacity in (
            (
                "decode_utilization",
                "decode_steps_per_second",
                steps,
                "decode_capacity",
                args.decode_capacity,
            ),
            (
                "prefill_utilization",
                "input_tokens_per_second",
                tokens,
                "prefill_capacity",
                args.prefill_capacity,
            ),
        ):
            if capacity is not None:
                out.append(
                    _result(
                        name,
                        queueing.utilization(demand, capacity),
                        "",
                        "потребность / мощность",
                        {demand_key: demand, key: capacity},
                        A_CH3_QUEUE,
                        notes=(overload,),
                    )
                )
    elif args.decode_capacity is not None or args.prefill_capacity is not None:
        raise ValueError("для загрузки нужны --class и --rate")
    if any(v is not None for v in little.values()):
        missing = [flag for flag, value in little.items() if value is None]
        if missing:
            raise ValueError(
                "для закона Литтла нужны --arrival-rate и --time-in-system; "
                f"не хватает: {', '.join(missing)}"
            )
        out.append(
            _result(
                "in_system",
                queueing.littles_law_in_system(args.arrival_rate, args.time_in_system),
                "",
                "L = λ · W",
                {
                    "arrival_rate": args.arrival_rate,
                    "time_in_system": args.time_in_system,
                },
                A_CH11_LITTLE,
                notes=("среднее в устойчивом режиме",),
            )
        )
    return out


def _peak_label(peak: dict[str, Any]) -> str:
    fields = ("input_precision", "accumulator_precision", "execution_unit", "sparsity")
    unit = "TFLOP/s" if peak.get("operation_kind") == "floating_point" else "TOPS"
    label = "/".join(str(peak.get(field)) for field in fields)
    return f"{label}: {peak['tera_ops_per_second']} {unit}"


def _device_info(args: argparse.Namespace) -> list[Result]:
    dev = hardware.device(args.device, allow_aggregate=True)
    info = {
        "device": dev.id,
        "name": dev.name,
        "scope": dev.scope,
        "device_count": dev.device_count,
        "shared_with_cpu": dev.shared_with_cpu,
    }
    scope = _aggregate_note(
        dev,
        ["ёмкость памяти", "пропускная способность памяти", "пики"],
        "; на одно устройство снимок их не делит",
    )
    memory_notes = [
        *scope,
        "номинальная ёмкость по этикетке, а не доступный среде выполнения объём",
    ]
    if dev.shared_with_cpu:
        memory_notes.append("память общая с CPU и ОС: модели доступна только её часть")
    if dev.memory_bytes is None:
        memory_notes.append(f"ёмкость не опубликована в снимке {hardware.SNAPSHOT}")
    bandwidth_notes = [*scope]
    if dev.bandwidth is None:
        bandwidth_notes.append(
            f"пропускная способность не опубликована в снимке {hardware.SNAPSHOT}"
        )
    try:
        peak: float | None = hardware.peak_flops(dev)
        peak_notes = [*scope]
    except ValueError as error:
        peak, peak_notes = None, [*scope, str(error)]
    peak_notes += [
        _peak_label(p) for p in dev.peaks if p.get("tera_ops_per_second") is not None
    ]
    return [
        _result(
            "device_count",
            dev.device_count,
            "",
            "spec_scope и gpu_count/npu_count снимка",
            info,
            A_CH1_KEYS,
            notes=scope,
        ),
        _result(
            "memory_bytes",
            dev.memory_bytes,
            "B",
            "nominal_capacity × единица ёмкости",
            info,
            A_CH1_KEYS,
            notes=memory_notes,
        ),
        _result(
            "bandwidth",
            dev.bandwidth,
            "B/s",
            "bandwidth_bytes_per_second",
            info,
            A_CH1_KEYS,
            notes=bandwidth_notes,
        ),
        _result(
            "peak_flops",
            peak,
            "FLOP/s",
            "BF16 / накопление FP32 / tensor / dense; все пики снимка — в примечаниях (вход/накопление/блок/sparsity)",
            info,
            A_CH1_KEYS,
            notes=peak_notes,
        ),
    ]


# --- разбор аргументов ---------------------------------------------------------


class _UsageError(Exception):
    """Ошибка разбора аргументов: сообщение уже переведено."""


_USAGE_MESSAGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"the following arguments are required: (?P<a>.+)"),
        "не заданы обязательные аргументы: {a}",
    ),
    (re.compile(r"unrecognized arguments: (?P<a>.+)"), "неизвестные аргументы: {a}"),
    (
        re.compile(r"ambiguous option: (?P<a>\S+) could match (?P<b>.+)"),
        "неоднозначный флаг {a}: подходят {b}",
    ),
)
_ARGUMENT_MESSAGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"invalid choice: (?P<a>.+?) \(choose from (?P<b>.+)\)"),
        "недопустимое значение {a}; допустимы: {b}",
    ),
    (
        re.compile(r"invalid \w+ value: (?P<a>.+)"),
        "ожидается число, получено {a}",
    ),
    (re.compile(r"expected one argument"), "нужно значение"),
)


def _translate(message: str) -> str:
    """Сообщения argparse — на русском; неизвестные передаются как есть."""
    argument = re.fullmatch(
        r"argument (?P<name>[^:]+): (?P<rest>.+)", message, re.DOTALL
    )
    if argument:
        rest = argument["rest"]
        for pattern, template in _ARGUMENT_MESSAGES:
            match = pattern.fullmatch(rest)
            if match:
                rest = template.format(**match.groupdict())
                break
        return f"аргумент {argument['name']}: {rest}"
    for pattern, template in _USAGE_MESSAGES:
        match = pattern.fullmatch(message)
        if match:
            return template.format(**match.groupdict())
    return f"ошибка аргументов: {message}"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(_translate(message))


def _count(text: str) -> int:
    """Целое число, записанное как угодно: 397e9, 397000000000."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидается число: {text!r}") from None
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise argparse.ArgumentTypeError(f"ожидается целое число не меньше 1: {text!r}")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    fmt = _Parser(add_help=False)
    fmt.add_argument(
        "--format",
        choices=("json", "md"),
        default=argparse.SUPPRESS,
        help="вывод: md (по умолчанию) или json",
    )
    parser = _Parser(
        prog="calc.py",
        description="Калькуляторы по книге «AI Infra in Depth». Каждый результат — "
        "с формулой, входными данными и якорем на книгу.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md"),
        default="md",
        help="вывод: md (по умолчанию) или json",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="команда")

    def command(
        name: str,
        handler: Callable[[argparse.Namespace], list[Result]],
        help_text: str,
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[fmt], help=help_text, description=help_text)
        p.set_defaults(handler=handler)
        return p

    def device_options(p: argparse.ArgumentParser, bandwidth: bool = True) -> None:
        p.add_argument("--device", help=f"id устройства из снимка {hardware.SNAPSHOT}")
        p.add_argument(
            "--allow-aggregate",
            action="store_true",
            help="разрешить стойку, сервер или суперчип: значения суммированы по устройствам",
        )
        p.add_argument("--precision", default="BF16", help="точность входа пика (BF16)")
        p.add_argument(
            "--accumulator",
            default="FP32",
            help="точность накопления пика (FP32, как в книге)",
        )
        p.add_argument("--sparsity", default="dense", choices=("dense", "structured"))
        p.add_argument(
            "--peak-tflops", type=float, help="пик явно, TFLOP/s (вместо снимка)"
        )
        if bandwidth:
            p.add_argument(
                "--bandwidth",
                type=float,
                help="пропускная способность памяти явно, байт/с",
            )

    p = command(
        "model", _model, "параметры, память весов, KV и FLOPs модели по config.json"
    )
    p.add_argument("--config", required=True, help="config.json в формате Hugging Face")
    p.add_argument("--context", type=int, default=0, help="длина контекста, токенов")
    p.add_argument("--weight-dtype", default="bf16")
    p.add_argument("--kv-dtype", default="bf16")
    p.add_argument(
        "--params",
        type=_count,
        help="число параметров из карточки модели, если калькулятор его не считает (гибридные модели)",
    )
    p.add_argument(
        "--quant-overhead-bytes",
        type=float,
        help="байт scale и частей в высокой точности сверх параметров × байты (квантизация)",
    )

    p = command("roofline", _roofline, "нижняя граница времени шага: max(F/Π, R/β)")
    device_options(p)
    p.add_argument("--flops", type=float, required=True, help="FLOPs шага")
    p.add_argument(
        "--bytes", type=float, required=True, help="байт, прочитанных за шаг"
    )

    p = command(
        "serving", _serving, "память, TTFT, TPOT, пропускная способность и цена токена"
    )
    device_options(p)
    p.add_argument("--weights", type=float, required=True, help="байт весов в памяти")
    p.add_argument(
        "--weight-read",
        type=float,
        required=True,
        help="байт весов, читаемых за шаг decode",
    )
    p.add_argument(
        "--decode-flops", type=float, required=True, help="FLOPs шага decode на запрос"
    )
    p.add_argument(
        "--prefill-flops", type=float, help="FLOPs prefill одного запроса (для TTFT)"
    )
    p.add_argument("--kv-per-token", type=float, required=True, help="байт KV на токен")
    p.add_argument(
        "--context", type=int, required=True, help="длина контекста, токенов"
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--memory", type=float, help="байт памяти одного устройства")
    p.add_argument(
        "--reserve", type=float, default=0, help="байт резерва среды выполнения"
    )
    p.add_argument("--price-per-hour", type=float, help="$ за час устройства")

    p = command("training", _training, "FLOPs обучения, состояние ZeRO и срок обучения")
    device_options(p, bandwidth=False)
    p.add_argument("--config", required=True, help="config.json в формате Hugging Face")
    p.add_argument(
        "--tokens", type=int, required=True, help="длина обучающей последовательности"
    )
    p.add_argument("--dp", type=int, help="число GPU в группе шардирования ZeRO")
    p.add_argument("--total-tokens", type=float, help="всего токенов обучения")
    p.add_argument("--devices", type=int, help="число ускорителей")
    p.add_argument("--mfu", type=float, help="доля пика, (0, 1]")

    p = command("checkpoint", _checkpoint, "интервал checkpoint и утилизация конвейера")
    p.add_argument("--checkpoint-bytes", type=float, required=True)
    p.add_argument("--save-bandwidth", type=float, required=True, help="байт/с")
    p.add_argument("--devices", type=int, required=True)
    p.add_argument(
        "--device-mtbf",
        type=float,
        required=True,
        help="секунд между сбоями одного устройства",
    )
    p.add_argument("--recovery", type=float, default=0, help="секунд на восстановление")
    p.add_argument(
        "--interval",
        type=float,
        help="интервал для доли потерь, секунд (по умолчанию — оптимум)",
    )
    p.add_argument("--stages", type=int, help="стадий конвейера")
    p.add_argument("--microbatches", type=int, help="микропакетов на шаг")

    p = command(
        "speculative",
        _speculative,
        "спекулятивное декодирование: E[N] и безубыточный раунд",
    )
    p.add_argument(
        "--acceptance", type=float, required=True, help="доля принятия позиции, [0, 1]"
    )
    p.add_argument(
        "--draft", type=int, required=True, help="токенов черновика за раунд"
    )
    p.add_argument(
        "--plain-step",
        type=float,
        required=True,
        help="секунд на обычный шаг decode",
    )

    p = command("ring", _ring, "кольцевой AllReduce")
    p.add_argument("--devices", type=int, required=True)
    p.add_argument("--message", type=float, required=True, help="байт")
    p.add_argument(
        "--bandwidth", type=float, required=True, help="байт/с в одном направлении"
    )
    p.add_argument("--alpha", type=float, required=True, help="секунд на раунд")

    p = command("cost", _cost, "стоимость вызова и принятой задачи")
    for name in (
        "input-tokens",
        "output-tokens",
        "cached-tokens",
        "cache-write-tokens",
    ):
        p.add_argument(
            f"--{name}", type=float, default=0, help="токенов (можно среднее)"
        )
    for name in (
        "input-price",
        "output-price",
        "cache-read-price",
        "cache-write-price",
    ):
        p.add_argument(f"--{name}", type=float, default=0, help="$ за миллион токенов")
    p.add_argument("--calls", type=float, default=1, help="вызовов на попытку задачи")
    p.add_argument(
        "--success", type=float, default=1.0, help="вероятность приёмки, (0, 1]"
    )

    p = command("edge", _edge, "полное время взаимодействия устройства и облака")
    p.add_argument("--upload", required=True, help="объём с единицей, например '30 MB'")
    p.add_argument(
        "--download", required=True, help="объём с единицей, например '5 MB'"
    )
    p.add_argument("--up-mbps", type=float, required=True)
    p.add_argument("--down-mbps", type=float, required=True)
    p.add_argument("--rtt", type=float, required=True, help="секунд")
    p.add_argument("--compute", type=float, required=True, help="секунд работы модели")

    p = command("units", _units, "перевод объёмов, скоростей канала и типов данных")
    p.add_argument("--size", help="объём с явной единицей, например '141.11 GB'")
    p.add_argument(
        "--to",
        action="append",
        choices=tuple(units.UNITS),
        help="в какую единицу перевести --size (можно несколько раз)",
    )
    p.add_argument("--mbps", type=float, help="скорость канала, Mbit/s")
    p.add_argument("--dtype", help="тип данных: bf16, fp8, int4, …")

    p = command(
        "queueing", _queueing, "потребность по классам запросов, загрузка, закон Литтла"
    )
    p.add_argument(
        "--class",
        dest="classes",
        action="append",
        help="класс запросов имя=входные:выходные токены, например long_input=8192:256",
    )
    p.add_argument(
        "--rate",
        dest="rates",
        action="append",
        help="интенсивность класса имя=запросов/с, например long_input=2",
    )
    p.add_argument("--decode-capacity", type=float, help="шагов decode в секунду")
    p.add_argument("--prefill-capacity", type=float, help="входных токенов в секунду")
    p.add_argument("--arrival-rate", type=float, help="поступлений в секунду")
    p.add_argument("--time-in-system", type=float, help="секунд в системе")

    p = command(
        "device",
        _device_info,
        "запись снимка: память, пропускная способность, пики, охват",
    )
    p.add_argument("--device", required=True)
    return parser


def _error_text(error: Exception) -> str:
    # str(KeyError) заключает сообщение в кавычки
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def _requested_json(argv: Sequence[str]) -> bool:
    """Формат вывода для ошибки разбора, когда argparse ещё не вернул аргументы."""
    for index, item in enumerate(argv):
        if item == "--format=json":
            return True
        if item == "--format" and index + 1 < len(argv) and argv[index + 1] == "json":
            return True
    return False


def _print_error(message: str, as_json: bool, command: str | None) -> None:
    if as_json:
        print(json.dumps({"command": command, "error": message}, ensure_ascii=False))
    else:
        print(f"Ошибка: {message}")


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        args = parser.parse_args(raw)
    except _UsageError as error:
        hint = "" if _requested_json(raw) else ". Справка: calc.py <команда> --help"
        _print_error(f"{error}{hint}", _requested_json(raw), None)
        return 2
    except SystemExit as stop:
        # --help: argparse уже напечатал справку
        return stop.code if isinstance(stop.code, int) else 0
    as_json = args.format == "json"
    try:
        results = args.handler(args)
    except (ValueError, KeyError) as error:
        _print_error(_error_text(error), as_json, args.command)
        return 2
    snapshot = hardware.SNAPSHOT if getattr(args, "device", None) else None
    if as_json:
        payload = {
            "command": args.command,
            "hardware_snapshot": snapshot,
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if snapshot:
            print(f"Снимок железа: {snapshot}\n")
        print("\n\n".join(result.to_markdown() for result in results))
    return 0
