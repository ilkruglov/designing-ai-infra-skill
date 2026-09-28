import io
import json
import shlex
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from infra_calc import cli
from infra_calc.result import Result

ANCHORS = (
    "references/source-book/chapter1.md:152",
    "references/source-book/chapter1.md:189",
    "references/source-book/chapter1.md:263",
    "references/source-book/chapter1.md:450",
    "references/source-book/chapter3.md:75",
    "references/source-book/chapter3.md:384",
    "references/source-book/chapter3.md:442",
    "references/source-book/chapter3.md:611",
    "references/source-book/chapter2.md:542",
    "references/source-book/chapter6.md:131",
    "references/source-book/chapter6.md:391",
    "references/source-book/chapter6.md:473",
    "references/source-book/chapter6.md:705",
    "references/source-book/chapter8.md:52",
    "references/source-book/chapter8.md:268",
    "references/source-book/chapter8.md:536",
    "references/source-book/chapter8.md:598",
    "references/source-book/chapter10.md:151",
    "references/source-book/chapter10.md:322",
    "references/source-book/chapter10.md:553",
    "references/source-book/chapter10.md:600",
    "references/source-book/chapter11.md:62",
    "references/source-book/chapter11.md:515",
    "references/source-book/chapter2.md:236",
    "references/source-book/chapter12.md:11",
    "references/source-book/chapter12.md:156",
    "calculations/results/qwen3-30b-a3b-decode-b1-s8192.json#sha256=fbf0b07f78bd8d7ab3765f6fc9ad5f6992cc95192449c2f8f1ee0fd01b5bc75b",
    "calculations/results/checkpoint-interval-book.json#sha256=e986d06ba47b7d0c13b54a99cc2abde1744d674eb470b26d9cf00e417ac0c6ad",
    "calculations/results/training-pipeline-interleaved-m8.json#sha256=773f52ffe9134ea65961825a000bba925cd1734beb6937b2488ef400710703a3",
    "calculations/results/training-state-book.json#sha256=0cdea3fbd70fac7846e6655282e76624d3f5b14fd0e2627f5393fe178b6f3cd5",
    "calculations/results/tree-qwen3-32b-t1-p8-h100.json#sha256=310710bf34b7d88a7ac7bd35335247b12c1b627549fd067ab52ed37691f07448",
)
SCRIPTS = Path(__file__).resolve().parents[1]
SKILL = SCRIPTS.parent
CONFIGS = SCRIPTS / "tests" / "fixtures" / "configs"
QWEN3_8B = str(CONFIGS / "qwen3-8b.json")
AUTHOR = json.loads(
    (SCRIPTS / "tests" / "fixtures" / "author_results.json").read_text()
)["checkpoint-interval-book"]


def call(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = cli.main(list(argv))
    return code, out.getvalue()


def run(*argv: str) -> dict[str, Any]:
    code, out = call(*argv, "--format", "json")
    assert code == 0, out
    return json.loads(out)


def values(*argv: str) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in run(*argv)["results"]}


def fails(*argv: str) -> str:
    code, out = call(*argv, "--format", "json")
    assert code == 2, out
    return json.loads(out)["error"]


class ModelCommandTest(unittest.TestCase):
    def test_qwen3_8b_reference_numbers(self) -> None:
        v = values("model", "--config", QWEN3_8B, "--context", "2048")
        # chapter3.md:414: «рассмотрим 8 190 735 360 параметров Qwen3-8B»
        self.assertEqual(v["parameters"]["value"], 8_190_735_360)
        # chapter1.md:456: «| Полные веса BF16 | 16.38 GB |»
        self.assertEqual(round(v["weight_bytes"]["value"] / 1e9, 2), 16.38)
        # chapter1.md:457: «Полный KV модели на каждый токен контекста | 144 KiB»
        self.assertEqual(v["kv_bytes_per_token"]["value"], 144 * 2**10)
        # chapter1.md:458: «| KV контекста из 2048 токенов | 288 MiB |»
        self.assertEqual(v["kv_resident_bytes"]["value"], 288 * 2**20)
        # chapter1.md:459: «Матричные операции prefill для 2048 токенов | 29.69 TFLOPs»
        self.assertEqual(round(v["prefill_flops"]["value"] / 1e12, 2), 29.69)
        # chapter1.md:460: «Матричные операции одного шага decode | 16.34 GFLOPs»
        self.assertEqual(round(v["decode_step_flops"]["value"] / 1e9, 2), 16.34)
        # chapter1.md:461: «Основной объём чтения весов за один шаг decode | 15.14 GB»
        self.assertEqual(round(v["decode_weight_read_bytes"]["value"] / 1e9, 2), 15.14)
        for item in v.values():
            self.assertTrue(item["anchor"].startswith("references/source-book/"))

    def test_unsupported_model_exits_with_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "config.json"
            bad.write_text(
                json.dumps({"model_type": "kimi_linear", "linear_attn_config": {}})
            )
            message = fails("model", "--config", str(bad))
        self.assertIn("k3-forward", message)

    def test_missing_config_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            message = fails("model", "--config", str(Path(tmp) / "absent.json"))
        self.assertIn("absent.json", message)

    def test_config_without_required_field_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "config.json"
            bad.write_text(json.dumps({"model_type": "qwen3"}))
            message = fails("model", "--config", str(bad))
        self.assertIn("num_attention_heads", message)

    def test_hybrid_parameters_are_refused_not_guessed(self) -> None:
        config = str(CONFIGS / "qwen3.5-397b-a17b.json")
        v = values("model", "--config", config)
        self.assertIsNone(v["parameters"]["value"])
        notes = " ".join(v["parameters"]["notes"])
        self.assertIn("--params", notes)
        # обёртка qwen3_5_moe над текстовой моделью: энкодеры не учтены
        self.assertIn("qwen3_5_moe", notes)
        self.assertIn("энкодер", notes)
        self.assertNotIn("weight_bytes", v)

    def test_hybrid_parameters_from_user_are_labelled(self) -> None:
        # число из карточки модели передаётся как есть; значение условное
        config = str(CONFIGS / "qwen3.5-397b-a17b.json")
        v = values("model", "--config", config, "--params", "1000")
        self.assertEqual(v["parameters"]["value"], 1000)
        self.assertIn("задано пользователем", v["parameters"]["formula"])
        self.assertEqual(v["weight_bytes"]["value"], 2000)

    def test_params_for_countable_model_is_refused(self) -> None:
        # chapter2-model-comparison.json: Qwen3-8B total_parameters 8 190 735 360
        message = fails("model", "--config", QWEN3_8B, "--params", "1000")
        self.assertIn("получено 8 190 735 360", message)


class RooflineCommandTest(unittest.TestCase):
    def test_roofline_uses_snapshot_and_names_it(self) -> None:
        data = run(
            "roofline", "--device", "h100-sxm", "--flops", "140e9", "--bytes", "70e9"
        )
        v = {item["name"]: item for item in data["results"]}
        # chapter1.md:268: «\frac{70\ \mathrm{GB}}{3350\ \mathrm{GB/s}}\approx20{,}90\ \mathrm{ms}»
        self.assertEqual(round(v["step_lower_bound_seconds"]["value"] * 1e3, 2), 20.90)
        # chapter1.md:268: «\frac{140\ \mathrm{GFLOPs}}{989400\ \mathrm{GFLOP/s}}\approx0{,}1415\ \mathrm{ms}»
        self.assertEqual(round(v["compute_seconds"]["value"] * 1e3, 4), 0.1415)
        self.assertEqual(v["step_lower_bound_seconds"]["bound"], "lower")
        self.assertIn("56ecb425", data["hardware_snapshot"])

    def test_format_before_command(self) -> None:
        code, out = call(
            "--format",
            "json",
            "roofline",
            "--peak-tflops",
            "989.4",
            "--bandwidth",
            "3.35e12",
            "--flops",
            "140e9",
            "--bytes",
            "70e9",
        )
        self.assertEqual(code, 0, out)
        data = json.loads(out)
        # chapter1.md:268: 20,90 ms без снимка — пик и полоса заданы явно
        self.assertEqual(round(data["results"][0]["value"] * 1e3, 2), 20.90)
        self.assertIsNone(data["hardware_snapshot"])

    def test_markdown_marks_lower_bound(self) -> None:
        code, out = call(
            "roofline", "--device", "h100-sxm", "--flops", "140e9", "--bytes", "70e9"
        )
        self.assertEqual(code, 0)
        self.assertIn("(нижняя граница)", out)
        self.assertIn("56ecb425 (2026-09-26)", out)
        self.assertIn("references/source-book/chapter1.md:263", out)
        # входы — в том же виде, что и значения, а не 140000000000.0
        self.assertIn("flops=1.4e+11", out)
        self.assertIn("accumulator=FP32", out)

    def test_accumulator_selects_peak(self) -> None:
        # chapter1.md:184: «Пиковая производительность матриц BF16 (TFLOP/s) | 165.2»
        # у RTX 4090 при накоплении FP32; с накоплением FP16 в снимке 330.3
        # (data/hardware.json, как в test_hardware.test_accumulator_defaults_to_fp32)
        args = ("roofline", "--device", "rtx4090", "--precision", "FP16")
        work = ("--flops", "1e12", "--bytes", "1")
        fp32 = values(*args, *work)["compute_seconds"]
        self.assertEqual(fp32["inputs"]["peak"], 165.2e12)
        self.assertEqual(fp32["inputs"]["accumulator"], "FP32")
        fp16 = values(*args, "--accumulator", "FP16", *work)["compute_seconds"]
        self.assertEqual(fp16["inputs"]["peak"], 330.3e12)

    def test_peak_lookup_failure_names_both_ways_out(self) -> None:
        # у b200-sxm в снимке FP16 с накоплением «unspecified», пика FP16/FP32 нет
        message = fails(
            "roofline",
            "--device",
            "b200-sxm",
            "--precision",
            "FP16",
            "--flops",
            "1",
            "--bytes",
            "1",
        )
        self.assertIn("--accumulator", message)
        self.assertIn("--peak-tflops", message)

    def test_aggregate_is_refused_without_opt_in(self) -> None:
        # chapter6.md:711: «GB200 NVL72 объединяет 72 GPU» — стойка, а не карта
        message = fails(
            "roofline", "--device", "gb200-nvl72", "--flops", "1", "--bytes", "1"
        )
        self.assertIn("агрегат", message)
        self.assertIn("--allow-aggregate", message)

    def test_aggregate_with_opt_in_is_noted(self) -> None:
        v = values(
            "roofline",
            "--device",
            "gb200-nvl72",
            "--allow-aggregate",
            "--flops",
            "1",
            "--bytes",
            "1",
        )
        self.assertIn("72", " ".join(v["step_lower_bound_seconds"]["notes"]))

    def test_rates_without_device_are_required(self) -> None:
        message = fails("roofline", "--flops", "1", "--bytes", "1")
        self.assertIn("--device", message)


class ServingCommandTest(unittest.TestCase):
    RTX = (
        "serving",
        "--device",
        "rtx-pro6000-blackwell-ws",
        "--weights",
        "16.38e9",
        "--weight-read",
        "15.14e9",
        "--decode-flops",
        "16.34e9",
        "--prefill-flops",
        "29.69e12",
        "--kv-per-token",
        "147456",
        "--context",
        "2048",
    )

    def test_rtx_pro_6000_bounds(self) -> None:
        # chapter1.md:464: при 503.8 TFLOP/s «матричные операции prefill занимают около
        # 58.9 ms»; при 1.792 TB/s чтение весов и KV за шаг decode «занимает около 8.62 ms»
        v = values(*self.RTX)
        self.assertEqual(round(v["ttft_lower_bound_seconds"]["value"] * 1e3, 1), 58.9)
        self.assertEqual(round(v["tpot_lower_bound_seconds"]["value"] * 1e3, 2), 8.62)
        self.assertEqual(v["tpot_lower_bound_seconds"]["bound"], "lower")
        self.assertEqual(v["tokens_per_second_upper_bound"]["bound"], "upper")

    def test_aggregate_memory_needs_per_device_value(self) -> None:
        message = fails(
            "serving",
            "--device",
            "gb200-nvl72",
            "--weights",
            "1e9",
            "--weight-read",
            "1e9",
            "--decode-flops",
            "1e9",
            "--kv-per-token",
            "1000",
            "--context",
            "10",
        )
        self.assertIn("--memory", message)
        self.assertIn("--allow-aggregate", message)

    def test_aggregate_memory_with_opt_in_is_noted(self) -> None:
        v = values(
            "serving",
            "--device",
            "gb200-nvl72",
            "--allow-aggregate",
            "--weights",
            "1e9",
            "--weight-read",
            "1e9",
            "--decode-flops",
            "1e9",
            "--kv-per-token",
            "1000",
            "--context",
            "10",
        )
        notes = " ".join(v["max_concurrent_requests"]["notes"])
        self.assertIn("72", notes)


class TrainingCommandTest(unittest.TestCase):
    def test_qwen3_8b_sequence_and_zero_states(self) -> None:
        v = values("training", "--config", QWEN3_8B, "--tokens", "8192", "--dp", "8")
        # chapter3.md:452: «| Прямой и обратный проходы | 431,368 |» TFLOPs
        self.assertEqual(
            round(v["training_flops_per_sequence"]["value"] / 1e12, 3), 431.368
        )
        # chapter3.md:454: «| Оценка $6ND$ по общему числу параметров | 402,591 |»
        self.assertEqual(
            round(v["six_nd_flops_per_sequence"]["value"] / 1e12, 3), 402.591
        )
        # chapter10.md:190-193: 122.1 / 42.0 / 28.6 / 15.3 GiB на GPU при восьми GPU
        gib = [
            round(v[f"zero{stage}_state_bytes_per_gpu"]["value"] / 2**30, 1)
            for stage in range(4)
        ]
        self.assertEqual(gib, [122.1, 42.0, 28.6, 15.3])

    def test_duration_refuses_aggregate_peak_with_device_count(self) -> None:
        # пик агрегата уже суммирован по 72 GPU; --devices 72 умножил бы его ещё раз
        message = fails(
            "training", "--config", QWEN3_8B, "--tokens", "8192", "--total-tokens", "1e9",
            "--devices", "72", "--mfu", "0.4", "--device", "gb200-nvl72", "--allow-aggregate",
        )  # fmt: skip
        self.assertIn("агрегат", message)
        self.assertIn("--peak-tflops", message)
        # пик на одну карту, заданный явно, не зависит от агрегата
        v = values(
            "training", "--config", QWEN3_8B, "--tokens", "8192", "--total-tokens", "16384",
            "--devices", "2", "--mfu", "0.5", "--peak-tflops", "1000",
            "--device", "gb200-nvl72", "--allow-aggregate",
        )  # fmt: skip
        per_sequence = v["training_flops_per_sequence"]["value"]
        self.assertAlmostEqual(
            v["training_seconds"]["value"], per_sequence / (1e15 * 0.5)
        )

    def test_duration_on_whole_aggregate_is_allowed(self) -> None:
        # одна стойка — один пик агрегата: N·Π = Π_агрегата, пик не умножается повторно
        agg = values("device", "--device", "gb200-nvl72")["peak_flops"]["value"]
        v = values(
            "training", "--config", QWEN3_8B, "--tokens", "8192", "--total-tokens", "16384",
            "--devices", "1", "--mfu", "0.4", "--device", "gb200-nvl72", "--allow-aggregate",
        )  # fmt: skip
        item = v["training_seconds"]
        per_sequence = v["training_flops_per_sequence"]["value"]
        self.assertAlmostEqual(item["value"], 2 * per_sequence / (agg * 0.4))
        self.assertIn("72", " ".join(item["notes"]))

    def test_duration_needs_all_its_inputs(self) -> None:
        message = fails(
            "training", "--config", QWEN3_8B, "--tokens", "8192", "--devices", "8"
        )
        self.assertIn("--mfu", message)


class OtherCommandsTest(unittest.TestCase):
    def test_checkpoint_interval(self) -> None:
        v = values(
            "checkpoint",
            "--checkpoint-bytes",
            str(AUTHOR["checkpoint_payload_bytes"]),
            "--save-bandwidth",
            str(AUTHOR["save_bandwidth_bytes_per_second"]),
            "--devices",
            str(AUTHOR["devices"]),
            "--device-mtbf",
            str(AUTHOR["device_mtbf_seconds"]),
            "--recovery",
            str(AUTHOR["recovery_seconds"]),
            "--interval",
            "60",
        )
        # checkpoint-interval-book.json: first_order_optimal_useful_interval_seconds,
        # poisson_optimal_useful_interval_seconds, first_order_loss_at_60;
        # chapter10.md:570: «около 965 s»
        self.assertAlmostEqual(
            v["first_order_optimal_interval_seconds"]["value"],
            AUTHOR["first_order_optimal_useful_interval_seconds"],
            places=6,
        )
        self.assertAlmostEqual(
            v["poisson_optimal_interval_seconds"]["value"],
            AUTHOR["poisson_optimal_useful_interval_seconds"],
            places=6,
        )
        self.assertAlmostEqual(
            v["first_order_loss_at_interval"]["value"],
            AUTHOR["first_order_loss_at_60"],
            places=12,
        )

    def test_checkpoint_rejects_zero_bandwidth(self) -> None:
        message = fails(
            "checkpoint",
            "--checkpoint-bytes",
            "1e9",
            "--save-bandwidth",
            "0",
            "--devices",
            "8",
            "--device-mtbf",
            "1e6",
        )
        self.assertIn("--save-bandwidth", message)

    def test_pipeline_utilization(self) -> None:
        # chapter10.md:336: «При $p=4,m=8$ получаем $u=8/11\approx72.7\%$.»
        v = values(
            "checkpoint",
            "--checkpoint-bytes",
            "1e9",
            "--save-bandwidth",
            "1e9",
            "--devices",
            "8",
            "--device-mtbf",
            "1e6",
            "--stages",
            "4",
            "--microbatches",
            "8",
        )
        self.assertEqual(round(v["pipeline_utilization"]["value"] * 100, 1), 72.7)

    def test_speculative(self) -> None:
        # chapter8.md:552: «Для черновика AAAA значение $a=1/4$, а среднее число
        # выходных токенов за раунд составляет около 1,33»; раунд проверки 26,26 мс
        v = values(
            "speculative",
            "--acceptance",
            "0.25",
            "--draft",
            "4",
            "--plain-step",
            "26.26e-3",
        )
        self.assertEqual(round(v["expected_tokens_per_round"]["value"], 2), 1.33)
        # chapter8.md:554: «на запрос черновика остаётся около 8,7 мс для AAAA»
        budget = v["breakeven_round_seconds"]["value"] - 26.26e-3
        self.assertEqual(round(budget * 1e3, 1), 8.7)

    def test_ring(self) -> None:
        # chapter6.md:499-501: M = 10 KiB, B = 450 GB/s, α = 0,822 μs, TP8:
        # «каждая карта отправляет 17,5 KiB, ... а одна операция — около 11,55 μs»
        v = values(
            "ring",
            "--devices",
            "8",
            "--message",
            "10240",
            "--bandwidth",
            "450e9",
            "--alpha",
            "0.822e-6",
        )
        self.assertEqual(round(v["ring_allreduce_seconds"]["value"] * 1e6, 2), 11.55)
        self.assertEqual(v["bytes_sent_per_device"]["value"] / 1024, 17.5)

    def test_cost_example_11_7(self) -> None:
        # chapter11.md:521-532: A — 1 000 новых и 19 000 из кэша по 1 / 0.1 $, 1 800 + 200
        # выходных по 5 $; «стоимость каждого вызова составляет 0,0129 доллара, а средняя
        # стоимость успешно выполненной задачи — $0.0129/0.8\approx0.0161$ доллара»
        v = values(
            "cost",
            "--input-tokens",
            "1000",
            "--output-tokens",
            "2000",
            "--input-price",
            "1",
            "--output-price",
            "5",
            "--cached-tokens",
            "19000",
            "--cache-read-price",
            "0.1",
            "--success",
            "0.8",
        )
        self.assertAlmostEqual(v["cost_per_call"]["value"], 0.0129)
        self.assertEqual(round(v["cost_per_accepted_task"]["value"], 4), 0.0161)

    def test_cost_accepts_expected_tokens(self) -> None:
        # chapter11.md:538: при h = 50% стоимость успешной задачи B «примерно до 0,0264»;
        # ожидаемые токены: 9 500 из кэша и 1 000 + 9 500 обычных, успех 0.98
        v = values(
            "cost",
            "--input-tokens",
            "10500.0",
            "--output-tokens",
            "300",
            "--input-price",
            "2",
            "--output-price",
            "10",
            "--cached-tokens",
            "9500.0",
            "--cache-read-price",
            "0.2",
            "--success",
            "0.98",
        )
        self.assertEqual(round(v["cost_per_accepted_task"]["value"], 4), 0.0264)

    def test_edge(self) -> None:
        # chapter12.md:33-37: «T_{\mathrm{serial}}=\frac{30\times8}{20}+0.1+0.3+\frac{5\times8}{100} =12.8\ \mathrm{s}»
        v = values(
            "edge",
            "--upload",
            "30 MB",
            "--download",
            "5 MB",
            "--up-mbps",
            "20",
            "--down-mbps",
            "100",
            "--rtt",
            "0.1",
            "--compute",
            "0.3",
        )
        self.assertAlmostEqual(v["serial_seconds"]["value"], 12.8)

    def test_edge_rejects_size_without_unit(self) -> None:
        message = fails(
            "edge",
            "--upload",
            "30",
            "--download",
            "5 MB",
            "--up-mbps",
            "20",
            "--down-mbps",
            "100",
            "--rtt",
            "0.1",
            "--compute",
            "0.3",
        )
        self.assertIn("единиц", message)


class DeviceCommandTest(unittest.TestCase):
    def test_single_device(self) -> None:
        # chapter1.md:182-184: H100 SXM — 80 GB, 3.35 TB/s, 989.4 TFLOP/s BF16
        v = values("device", "--device", "h100-sxm")
        self.assertEqual(v["memory_bytes"]["value"], 80e9)
        self.assertEqual(v["bandwidth"]["value"], 3.35e12)
        self.assertEqual(v["device_count"]["value"], 1)
        self.assertEqual(v["device_count"]["inputs"]["scope"], "single_device")
        self.assertEqual(v["peak_flops"]["value"], 989.4e12)
        notes = " ".join(v["peak_flops"]["notes"])
        self.assertIn("BF16/FP32/tensor/dense: 989.4 TFLOP/s", notes)
        # целочисленные пики снимка показаны как TOPS, а не FLOP/s
        self.assertIn("INT8/INT32/tensor/dense", notes)
        self.assertIn("TOPS", notes)

    def test_aggregate_shows_count_and_scope(self) -> None:
        # chapter6.md:711: «GB200 NVL72 объединяет 72 GPU и 36 CPU Grace»
        v = values("device", "--device", "gb200-nvl72")
        self.assertEqual(v["device_count"]["value"], 72)
        self.assertEqual(v["device_count"]["inputs"]["scope"], "gpu_aggregate")
        self.assertIn("72", " ".join(v["memory_bytes"]["notes"]))

    def test_shared_memory_is_flagged(self) -> None:
        v = values("device", "--device", "m2-max-38gpu-96gb")
        self.assertTrue(v["memory_bytes"]["inputs"]["shared_with_cpu"])
        self.assertIn("CPU", " ".join(v["memory_bytes"]["notes"]))

    def test_unknown_device_suggests_ids(self) -> None:
        self.assertIn("h100-sxm", fails("device", "--device", "h100"))


class ResultTest(unittest.TestCase):
    def test_unknown_bound_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Result("x", 1.0, "s", "F/Π", {}, "a", bound="minimum")

    def test_markdown_of_refused_value(self) -> None:
        text = Result(
            "x", None, "", "не вычисляется", {}, "a", notes=("почему",)
        ).to_markdown()
        self.assertIn("не вычисляется", text)
        self.assertIn("- почему", text)
        self.assertNotIn("входные данные", text)

    def test_markdown_shows_integers_in_full(self) -> None:
        # chapter3.md:414 записывает число параметров как «8 190 735 360»
        text = Result("parameters", 8_190_735_360, "", "Σ", {}, "a").to_markdown()
        self.assertIn("8 190 735 360", text)


class PipelineCommandTest(unittest.TestCase):
    def test_book_bubble_without_checkpoint_arguments(self) -> None:
        # chapter10.md:336: «При $p=4,m=8$ получаем $u=8/11\\approx72.7\\%$.»;
        # chapter10.md:406: доля пузырей 1F1B при восьми micro-batch — 37.5%
        v = values("pipeline", "--stages", "4", "--microbatches", "8")
        self.assertEqual(round(v["pipeline_utilization"]["value"] * 100, 1), 72.7)
        self.assertEqual(v["pipeline_utilization"]["bound"], "upper")
        self.assertEqual(v["pipeline_bubble_ratio"]["value"], 0.375)
        self.assertNotIn("pipeline_step_seconds", v)

    def test_interleaved_with_stage_times(self) -> None:
        # training-pipeline-interleaved-m8.json: bubble_bound_interleaved_seconds 0.045,
        # bubble_bound_interleaved_fraction 0.1875; шаг 285 ms — вывод вручную:
        # 8·30 + 3·30/2, событийная модель автора с передачами — 297.7 ms
        v = values(
            "pipeline", "--stages", "4", "--microbatches", "8",
            "--virtual-stages", "2", "--forward-seconds", "0.01",
            "--backward-seconds", "0.02",
        )  # fmt: skip
        self.assertEqual(v["pipeline_bubble_ratio"]["value"], 0.1875)
        self.assertAlmostEqual(v["pipeline_bubble_seconds"]["value"], 0.045)
        self.assertAlmostEqual(v["pipeline_step_seconds"]["value"], 0.285)
        self.assertEqual(v["pipeline_step_seconds"]["bound"], "lower")
        self.assertEqual(v["pipeline_bubble_seconds"]["bound"], "lower")

    def test_needs_both_stage_times(self) -> None:
        message = fails(
            "pipeline", "--stages", "4", "--microbatches", "8",
            "--forward-seconds", "0.01",
        )  # fmt: skip
        self.assertIn("--backward-seconds", message)

    def test_interleaved_needs_divisible_microbatches(self) -> None:
        message = fails(
            "pipeline", "--stages", "4", "--microbatches", "6", "--virtual-stages", "2"
        )
        self.assertIn("кратн", message)


class TrainingStateCommandTest(unittest.TestCase):
    def test_components_per_device(self) -> None:
        # training-state-book.json, stage 1, 8 участников: weights_bf16 16 381 470 720,
        # gradients 16 381 470 720, master + два момента 3 × 4 095 367 680,
        # persistent_bytes_per_rank 45 049 044 480
        v = values("training-state", "--config", QWEN3_8B, "--dp", "8", "--stage", "1")
        self.assertEqual(v["weight_state_bytes_per_device"]["value"], 16_381_470_720)
        self.assertEqual(v["gradient_state_bytes_per_device"]["value"], 16_381_470_720)
        self.assertEqual(
            v["optimizer_state_bytes_per_device"]["value"], 3 * 4_095_367_680
        )
        self.assertEqual(v["training_state_bytes_per_device"]["value"], 45_049_044_480)

    def test_params_and_fp32_gradients(self) -> None:
        # chapter10.md:190-193: ZeRO-3 на восьми GPU — 15.3 GiB; с градиентами FP32
        # training-state-fp32-gradient.json: stage 3 — 18 429 154 560 байт
        v = values(
            "training-state", "--params", "8190735360", "--dp", "8", "--stage", "3",
            "--grad-bytes", "4",
        )  # fmt: skip
        self.assertEqual(v["training_state_bytes_per_device"]["value"], 18_429_154_560)
        v = values(
            "training-state", "--params", "8190735360", "--dp", "8", "--stage", "3"
        )
        self.assertEqual(
            round(v["training_state_bytes_per_device"]["value"] / 2**30, 1), 15.3
        )

    def test_config_or_params(self) -> None:
        self.assertIn("--params", fails("training-state", "--dp", "8", "--stage", "0"))
        message = fails(
            "training-state", "--config", QWEN3_8B, "--params", "1000", "--dp", "8",
            "--stage", "0",
        )  # fmt: skip
        self.assertIn("--config", message)
        hybrid = str(CONFIGS / "qwen3.5-397b-a17b.json")
        message = fails(
            "training-state", "--config", hybrid, "--dp", "8", "--stage", "3"
        )
        self.assertIn("--params", message)


class CheckpointCommonShockTest(unittest.TestCase):
    BASE = (
        "checkpoint", "--checkpoint-bytes", "114670295040", "--save-bandwidth", "7e9",
        "--devices", "48", "--device-mtbf", "29122560", "--recovery", "120",
    )  # fmt: skip

    def test_loss_spike_book(self) -> None:
        # chapter10.md:623: всплеск раз в семь дней: оптимум «примерно с 4 458 s до
        # 3 150 s», при 1800 s потери «с 1,08% примерно до 1,25%»
        v = values(*self.BASE, "--interval", "1800", "--common-job-mtbf", "604800")
        self.assertEqual(
            round(v["first_order_optimal_interval_seconds"]["value"], -1), 3150
        )
        self.assertEqual(
            round(v["first_order_loss_at_interval"]["value"] * 100, 2), 1.25
        )
        self.assertEqual(
            v["first_order_loss_at_interval"]["anchor"],
            "references/source-book/chapter10.md:600",
        )
        self.assertEqual(
            v["first_order_loss_at_interval"]["inputs"]["common_job_mtbf"], 604800
        )
        plain = values(*self.BASE, "--interval", "1800")
        self.assertEqual(
            round(plain["first_order_loss_at_interval"]["value"] * 100, 2), 1.08
        )

    def test_rejects_non_positive_common_mtbf(self) -> None:
        message = fails(*self.BASE, "--common-job-mtbf", "0")
        self.assertIn("--common-job-mtbf", message)
        self.assertIn("больше нуля", message)


class BatchThresholdCommandTest(unittest.TestCase):
    def test_example_2_2_from_numbers_and_config(self) -> None:
        # chapter2.md:262: «при $H=8192$ минимальный целочисленный размер батча равен 13,
        # а при $H=2048$ — 51»
        raw = (
            "batch-threshold",
            "--weight-read",
            "15136811008",
            "--kv-per-token",
            "147456",
        )
        v = values(*raw, "--context", "8192")
        self.assertEqual(v["kv_read_batch_threshold"]["value"], 13)
        self.assertNotIn("compute_bound_batch_threshold", v)
        v = values("batch-threshold", "--config", QWEN3_8B, "--context", "2048")
        self.assertEqual(v["kv_read_batch_threshold"]["value"], 51)
        self.assertEqual(
            v["kv_read_batch_threshold"]["anchor"],
            "references/source-book/chapter2.md:236",
        )

    def test_compute_bound_book(self) -> None:
        # chapter1.md:307: B_* ≈ 147,7 при b_W = 1, Π = 989,4 TFLOP/s, β = 3,35 TB/s;
        # «После примерно 148 запросов время вычислений превышает время чтения весов»
        v = values(
            "batch-threshold", "--weight-read", "70e9", "--kv-per-token", "0",
            "--context", "1", "--decode-flops", "140e9", "--device", "h100-sxm",
        )  # fmt: skip
        self.assertEqual(v["compute_bound_batch_threshold"]["value"], 148)
        self.assertIn("147.7", " ".join(v["compute_bound_batch_threshold"]["notes"]))
        self.assertIsNone(v["kv_read_batch_threshold"]["value"])

    def test_compute_bound_hand_derived(self) -> None:
        # вывод вручную: B_* = (15e9/3e12) / (16e9/1e15 − 1e6/3e12) ≈ 319.15 → 320;
        # порог примера 2-2 — ceil(15e9 / 1e6) = 15 000
        v = values(
            "batch-threshold", "--weight-read", "15e9", "--kv-per-token", "1e6",
            "--context", "1", "--decode-flops", "16e9", "--peak-tflops", "1000",
            "--bandwidth", "3e12",
        )  # fmt: skip
        self.assertEqual(v["compute_bound_batch_threshold"]["value"], 320)
        self.assertEqual(v["kv_read_batch_threshold"]["value"], 15_000)

    def test_compute_bound_unreachable_for_qwen3_8b(self) -> None:
        # chapter1.md:459-461: 16.34 GFLOPs за шаг при 2048, KV 144 KiB на токен;
        # на запрос F/Π ≈ 16.5 μs < R_KV/β ≈ 90.1 μs — порога нет
        v = values(
            "batch-threshold", "--config", QWEN3_8B, "--context", "2048",
            "--device", "h100-sxm",
        )  # fmt: skip
        item = v["compute_bound_batch_threshold"]
        self.assertIsNone(item["value"])
        self.assertIn("не становится", " ".join(item["notes"]))

    def test_generalisation_is_named(self) -> None:
        # chapter1.md:305: B_* = b_W·Π/(2β) без KV; с R_KV — обобщение, а не формула книги
        v = values(
            "batch-threshold", "--weight-read", "15e9", "--kv-per-token", "1e6",
            "--context", "1", "--decode-flops", "16e9", "--peak-tflops", "1000",
            "--bandwidth", "3e12",
        )  # fmt: skip
        notes = " ".join(v["compute_bound_batch_threshold"]["notes"])
        self.assertIn("обобщ", notes)
        self.assertIn("chapter1.md:305", notes)

    def test_moe_config_is_a_lower_bound(self) -> None:
        v = values(
            "batch-threshold", "--config", str(CONFIGS / "qwen3-30b-a3b.json"),
            "--context", "8192",
        )  # fmt: skip
        item = v["kv_read_batch_threshold"]
        self.assertEqual(item["bound"], "lower")
        self.assertIn("объединение экспертов", " ".join(item["notes"]))

    def test_config_and_numbers_are_exclusive(self) -> None:
        message = fails(
            "batch-threshold", "--config", QWEN3_8B, "--weight-read", "1",
            "--context", "8",
        )  # fmt: skip
        self.assertIn("--config", message)
        self.assertIn("--weight-read", fails("batch-threshold", "--context", "8"))


class SpeculativeRoundsCommandTest(unittest.TestCase):
    def test_mean_time_per_token_book(self) -> None:
        # chapter8.md:544: «два раунда длительностью по 1,5 мс выводят соответственно 1 и
        # 5 токенов. Суммарно получаются 3 мс и 6 токенов, то есть в среднем 0,5 мс на токен»
        v = values("speculative", "--round", "0.0015:1", "--round", "0.0015:5")
        self.assertAlmostEqual(v["mean_time_per_token_seconds"]["value"], 0.5e-3)
        self.assertNotIn("expected_tokens_per_round", v)

    def test_mean_time_hand_derived(self) -> None:
        # вывод вручную: (0.03 + 0.02 + 0.05) / (2 + 1 + 4) = 0.1/7 s
        v = values(
            "speculative", "--round", "0.03:2", "--round", "0.02:1", "--round", "0.05:4"
        )
        self.assertAlmostEqual(v["mean_time_per_token_seconds"]["value"], 0.1 / 7)

    def test_modes_are_checked(self) -> None:
        self.assertIn("--round", fails("speculative"))
        self.assertIn("--plain-step", fails("speculative", "--acceptance", "0.5"))
        self.assertIn("--round", fails("speculative", "--round", "0.0015"))
        self.assertIn("--round", fails("speculative", "--round", "0.0015:0"))


class AllreduceCommandTest(unittest.TestCase):
    def test_ring_and_tree_book(self) -> None:
        # chapter6.md:524: для 8 карт и 10 KiB дерево «около 5,07 μs, что меньше 11,55 μs
        # у кольцевого алгоритма»; «точка пересечения ... примерно на 680 KiB»
        v = values(
            "allreduce", "--devices", "8", "--message", "10240",
            "--bandwidth", "450e9", "--alpha", "0.822e-6",
        )  # fmt: skip
        self.assertEqual(round(v["ring_allreduce_seconds"]["value"] * 1e6, 2), 11.55)
        self.assertEqual(round(v["tree_allreduce_seconds"]["value"] * 1e6, 2), 5.07)
        self.assertLess(v["tree_to_ring_time_ratio"]["value"], 1)
        self.assertEqual(round(v["ring_tree_crossover_bytes"]["value"] / 1024, -1), 680)

    def test_large_message_ring_wins(self) -> None:
        # chapter6.md:524: 80 MiB — кольцо «около 0,34 ms», дерево «около 1,12 ms»
        v = values(
            "allreduce", "--devices", "8", "--message", str(80 * 2**20),
            "--bandwidth", "450e9", "--alpha", "0.822e-6",
        )  # fmt: skip
        self.assertEqual(round(v["tree_allreduce_seconds"]["value"] * 1e3, 2), 1.12)
        self.assertGreater(v["tree_to_ring_time_ratio"]["value"], 1)


class MemoryOnlyRooflineTest(unittest.TestCase):
    def test_phone_bandwidth_book(self) -> None:
        # chapter12.md:165: «\\frac{15.14\\ \\mathrm{GB}}{84.8\\ \\mathrm{GB/s}}\\approx0.179\\ \\mathrm{s}»
        v = values(
            "roofline", "--memory-only", "--bandwidth", "84.8e9",
            "--flops", "0", "--bytes", "15.14e9",
        )  # fmt: skip
        self.assertEqual(round(v["step_lower_bound_seconds"]["value"], 3), 0.179)
        self.assertEqual(v["step_lower_bound_seconds"]["bound"], "lower")
        self.assertIsNone(v["compute_seconds"]["value"])
        self.assertIn("не оценива", " ".join(v["compute_seconds"]["notes"]))
        self.assertIsNone(v["ridge_point"]["value"])

    def test_apple_device_without_peaks(self) -> None:
        # вывод вручную: 1e9 B / 819e9 B/s (m3-ultra-80gpu-256gb в снимке, пиков нет)
        args = (
            "roofline",
            "--device",
            "m3-ultra-80gpu-256gb",
            "--flops",
            "1e12",
            "--bytes",
            "1e9",
        )
        self.assertIn("--memory-only", fails(*args))
        v = values(*args, "--memory-only")
        self.assertAlmostEqual(v["memory_seconds"]["value"], 1e9 / 819e9)
        self.assertAlmostEqual(v["step_lower_bound_seconds"]["value"], 1e9 / 819e9)
        self.assertEqual(v["arithmetic_intensity"]["value"], 1000)

    def test_memory_only_refuses_peak(self) -> None:
        message = fails(
            "roofline", "--memory-only", "--peak-tflops", "1", "--bandwidth", "1e9",
            "--flops", "0", "--bytes", "1",
        )  # fmt: skip
        self.assertIn("--peak-tflops", message)


class ModelParallelCommandTest(unittest.TestCase):
    QWEN3_32B = str(CONFIGS / "qwen3-32b.json")

    def test_kv_per_device_book(self) -> None:
        # chapter6.md:207: «При контексте 128K состояние каждой KV-головы занимает 4 GiB:
        # суммарно 32 GiB на восьми картах и 64 GiB на шестнадцати»
        for tp in ("8", "16"):
            with self.subTest(tp=tp):
                v = values(
                    "model",
                    "--config",
                    self.QWEN3_32B,
                    "--context",
                    "131072",
                    "--tp",
                    tp,
                )
                self.assertEqual(v["kv_resident_bytes_per_device"]["value"], 4 * 2**30)
                self.assertEqual(v["kv_bytes_per_token_per_device"]["value"], 32_768)
        notes = " ".join(v["kv_bytes_per_token_per_device"]["notes"])
        self.assertIn("дублиру", notes)
        # chapter6.md:139: «BF16-веса плотной модели Qwen3-32B занимают около 65,52 GB»
        v = values("model", "--config", self.QWEN3_32B, "--tp", "2")
        self.assertEqual(
            round(v["weight_bytes_per_device"]["value"] * 2 / 1e9, 2), 65.52
        )

    def test_kv_divisibility_is_stated(self) -> None:
        v = values("model", "--config", QWEN3_8B)
        self.assertIn("min(TP, 8)", " ".join(v["kv_bytes_per_token"]["notes"]))
        v = values("model", "--config", str(CONFIGS / "deepseek-v3.json"))
        self.assertIn("не делится", " ".join(v["kv_bytes_per_token"]["notes"]))
        v = values("model", "--config", str(CONFIGS / "deepseek-v3.json"), "--tp", "8")
        self.assertEqual(v["kv_bytes_per_token_per_device"]["value"], 70_272)

    def test_tp_must_divide_heads(self) -> None:
        message = fails("model", "--config", QWEN3_8B, "--tp", "3")
        self.assertIn("голов", message)


class ExpertUnionCommandTest(unittest.TestCase):
    QWEN3_30B = str(CONFIGS / "qwen3-30b-a3b.json")

    def test_uniform_routing_union(self) -> None:
        # шаблон sizing-sheet и формула (6-8), chapter6.md:422: U(4) ≈ 29.123 при E = 128, k = 8
        v = values("model", "--config", self.QWEN3_30B, "--batch", "4")
        union = v["experts_per_layer_at_batch"]
        self.assertEqual(round(union["value"], 3), 29.123)
        self.assertIn("равномерн", " ".join(union["notes"]))
        read = v["decode_weight_read_bytes_at_batch"]
        self.assertEqual(
            read["value"], int(2_459_856_896 + union["value"] * 48 * 9_437_184)
        )

    def test_uniform_read_is_an_estimate_not_a_bound(self) -> None:
        # U(B) по формуле (6-8), chapter6.md:391, — ожидание при равномерной независимой
        # маршрутизации; неравномерная затрагивает меньше экспертов, вплоть до U = k.
        # Нижняя граница чтения шага — decode_weight_read_bytes (U = k)
        v = values("model", "--config", self.QWEN3_30B, "--batch", "16")
        for name in ("experts_per_layer_at_batch", "decode_weight_read_bytes_at_batch"):
            with self.subTest(name=name):
                item = v[name]
                self.assertIsNone(item["bound"])
                notes = " ".join(item["notes"])
                self.assertIn("оценка при равномерной маршрутизации", notes)
        notes = " ".join(v["decode_weight_read_bytes_at_batch"]["notes"])
        self.assertIn("не нижняя граница", notes)
        self.assertIn("decode_weight_read_bytes", notes)
        self.assertIn("serving --weight-read", notes)
        # чтение одного токена (U = k) при batch — нижняя граница чтения шага
        single = " ".join(v["decode_weight_read_bytes"]["notes"])
        self.assertIn("нижняя граница чтения шага", single)
        # квантизация не делает оценку границей
        int8 = values(
            "model", "--config", self.QWEN3_30B, "--batch", "16", "--weight-dtype", "int8",
        )  # fmt: skip
        self.assertIsNone(int8["decode_weight_read_bytes_at_batch"]["bound"])
        # заданное объединение — не оценка: при квантизации чтение — нижняя граница
        given = values(
            "model", "--config", self.QWEN3_30B, "--batch", "16", "--weight-dtype", "int8",
            "--experts-per-layer", "40",
        )  # fmt: skip
        item = given["decode_weight_read_bytes_at_batch"]
        self.assertEqual(item["bound"], "lower")
        self.assertNotIn("оценка при равномерной", " ".join(item["notes"]))

    def test_explicit_union_matches_author(self) -> None:
        # calc.py forward --model qwen3-30b-a3b --batch 4 --history 8191 --tokens 1 --routing balanced
        # (код автора на 56ecb425): expert_union_per_layer 32, weight_read_once_per_operator_bytes
        # 16 955 387 904, из них 4 × 2048 × 2 байта — строки эмбеддингов
        v = values(
            "model", "--config", self.QWEN3_30B, "--batch", "4",
            "--experts-per-layer", "32",
        )  # fmt: skip
        self.assertEqual(v["experts_per_layer_at_batch"]["value"], 32)
        self.assertEqual(
            v["decode_weight_read_bytes_at_batch"]["value"], 16_955_387_904 - 16_384
        )

    def test_refusals(self) -> None:
        self.assertIn("MoE", fails("model", "--config", QWEN3_8B, "--batch", "4"))
        message = fails(
            "model", "--config", self.QWEN3_30B, "--batch", "4",
            "--experts-per-layer", "40",
        )  # fmt: skip
        self.assertIn("32", message)
        message = fails(
            "model", "--config", self.QWEN3_30B, "--experts-per-layer", "32"
        )
        self.assertIn("--batch", message)


class ServingSplitTest(unittest.TestCase):
    RTX_12GIB = (
        "serving", "--device", "rtx-pro6000-blackwell-ws", "--memory", "12884901888",
        "--weights", "0", "--weight-read", "15136811008", "--decode-flops", "16344154112",
        "--kv-per-token", "147456", "--context", "2048",
    )  # fmt: skip

    def test_memory_context_book(self) -> None:
        # chapter8.md:78: «Если учитывать только размер KV на момент завершения prefill, в
        # 12 GiB поместятся 42 коротких запроса»; с выделением на 2304 токена — 37
        # (chapter8.md:75); шаг — при длине входа 2048
        short = values(*self.RTX_12GIB)
        self.assertEqual(short["max_concurrent_requests"]["value"], 42)
        v = values(*self.RTX_12GIB, "--memory-context", "2304")
        memory = v["max_concurrent_requests"]
        self.assertEqual(memory["value"], 37)
        self.assertEqual(memory["inputs"]["memory_context"], 2304)
        step = v["tpot_lower_bound_seconds"]
        self.assertEqual(step["inputs"]["context"], 2048)
        self.assertEqual(step["value"], short["tpot_lower_bound_seconds"]["value"])

    LONG = (
        "serving", "--device", "rtx-pro6000-blackwell-ws", "--memory", "12884901888",
        "--weights", "0", "--weight-read", "15136811008", "--decode-flops", "19968032768",
        "--kv-per-token", "147456", "--context", "8192", "--memory-context", "8448",
    )  # fmt: skip

    def test_shared_prefix_book(self) -> None:
        # chapter8.md:292-298: общий префикс 6144 токена (864 MiB) хранится один раз,
        # на запрос — 8448 − 6144 = 2304 собственных токена (324 MiB); «предел ёмкости
        # увеличивается с 10 независимых запросов до 35»; вывод вручную:
        # ⌊(12 GiB − 6144 × 147 456) / (2304 × 147 456)⌋ = ⌊11 978 932 224 / 339 738 624⌋ = 35
        independent = values(*self.LONG)
        self.assertEqual(independent["max_concurrent_requests"]["value"], 10)
        shared = values(*self.LONG, "--shared-prefix-tokens", "6144")
        capacity = shared["max_concurrent_requests"]
        self.assertEqual(capacity["value"], 35)
        self.assertEqual(capacity["inputs"]["shared_prefix"], 6144)
        self.assertEqual(capacity["input_units"]["shared_prefix"], "tok")
        self.assertEqual(capacity["anchor"], "references/source-book/chapter8.md:268")
        self.assertIn("shared_prefix", capacity["formula"])
        self.assertIn("один раз", " ".join(capacity["notes"]))
        # при batch 1 чтение префикса один раз на batch и каждым запросом совпадает
        self.assertEqual(
            shared["tpot_lower_bound_seconds"]["value"],
            independent["tpot_lower_bound_seconds"]["value"],
        )

    def test_shared_prefix_step_reads_prefix_once_per_batch(self) -> None:
        # вариант B главы 8.6.3 (chapter8.md:666): batch 16, вход 8192, первые 6144 общие.
        # Как ядро читает общий префикс, книга не говорит, поэтому нижняя граница
        # читает его один раз на batch; вывод вручную:
        # (15 136 811 008 + 6144·147 456 + 16·2048·147 456) / 1.792e12 = 11.6488 ms.
        # Без дедупликации — (15 136 811 008 + 16·8192·147 456) / 1.792e12 = 19.2322 ms,
        # как «19.23 ms» при 1792 GB/s для 34,46 GB (chapter8.md:655, 675) и без префикса
        batch = ("--batch", "16")
        independent = values(*self.LONG, *batch)
        shared = values(*self.LONG, *batch, "--shared-prefix-tokens", "6144")
        step = shared["tpot_lower_bound_seconds"]
        self.assertEqual(step["bound"], "lower")
        self.assertAlmostEqual(step["value"] * 1e3, 11.648782857, places=6)
        self.assertEqual(step["inputs"]["context"], 8192)
        self.assertIn("один раз на batch", " ".join(step["notes"]))
        full = shared["tpot_without_prefix_dedup_seconds"]
        self.assertIsNone(full["bound"])
        self.assertEqual(
            full["value"], independent["tpot_lower_bound_seconds"]["value"]
        )
        self.assertEqual(round(full["value"] * 1e3, 2), 19.23)
        self.assertIn("каждым запросом", " ".join(full["notes"]))
        self.assertEqual(step["input_units"]["kv_shared_prefix"], "B")
        self.assertNotIn("kv_shared_prefix", full["inputs"])
        # пропускная способность — от нижней границы шага, поэтому верхняя граница
        self.assertAlmostEqual(
            shared["tokens_per_second_upper_bound"]["value"], 16 / step["value"]
        )
        # без префикса отдельного поля нет
        self.assertNotIn("tpot_without_prefix_dedup_seconds", independent)

    def test_shared_prefix_note_names_where_it_is_stored(self) -> None:
        # «на карту» — только для одиночного устройства и TP; память агрегата — вся запись
        single = values(*self.LONG, "--shared-prefix-tokens", "6144")
        self.assertIn(
            "B на карту", " ".join(single["max_concurrent_requests"]["notes"])
        )
        rack = values(
            "serving", "--device", "gb200-nvl72", "--allow-aggregate",
            "--weights", "0", "--weight-read", "1e9", "--decode-flops", "1e9",
            "--kv-per-token", "147456", "--context", "8192", "--memory-context", "8448",
            "--shared-prefix-tokens", "6144",
        )  # fmt: skip
        notes = " ".join(rack["max_concurrent_requests"]["notes"])
        self.assertIn("B во всём агрегате", notes)
        self.assertNotIn("на карту", notes)

    def test_shared_prefix_limits(self) -> None:
        # префикс — часть входа: не длиннее --context; --memory-context по-прежнему
        # полная длина запроса и не короче --context, с префиксом и без
        message = fails(*self.LONG, "--shared-prefix-tokens", "8193")
        self.assertIn("--shared-prefix-tokens", message)
        self.assertIn("--context", message)
        message = fails(
            *self.RTX_12GIB, "--memory-context", "2304", "--context", "8192",
            "--shared-prefix-tokens", "6144",
        )  # fmt: skip
        self.assertIn("--memory-context", message)
        message = fails(*self.LONG, "--shared-prefix-tokens", "0")
        self.assertIn("--shared-prefix-tokens", message)
        # весь запрос — префикс: собственных токенов нет, ёмкость не определена
        message = fails(*self.RTX_12GIB, "--shared-prefix-tokens", "2048")
        self.assertIn("собственных", message)

    def test_shared_prefix_with_tp_is_per_card(self) -> None:
        # арифметическое тождество: при TP2 у Qwen3-8B (8 голов KV) KV на карту — 73 728 B,
        # префикс на карту — 6144 × 73 728; ⌊(12 GiB − 6144·73 728) / (2304·73 728)⌋ = 73
        v = values(
            "serving", "--peak-tflops", "1", "--bandwidth", "1e12", "--memory", "12884901888",
            "--config", QWEN3_8B, "--tp", "2", "--weights", "0", "--weight-read", "0",
            "--decode-flops", "0", "--kv-per-token", "147456", "--context", "8192",
            "--memory-context", "8448", "--shared-prefix-tokens", "6144",
        )  # fmt: skip
        expected = (12884901888 - 6144 * 73728) // (2304 * 73728)
        self.assertEqual(v["max_concurrent_requests"]["value"], expected)
        self.assertEqual(expected, 73)

    def test_without_memory_only_time_bounds(self) -> None:
        # chapter1.md:464: TPOT 8.62 ms и TTFT 58.9 ms на RTX PRO 6000 — без бюджета памяти
        v = values(
            "serving", "--peak-tflops", "503.8", "--bandwidth", "1.792e12",
            "--weights", "16.38e9", "--weight-read", "15.14e9", "--decode-flops", "16.34e9",
            "--prefill-flops", "29.69e12", "--kv-per-token", "147456", "--context", "2048",
        )  # fmt: skip
        self.assertEqual(round(v["tpot_lower_bound_seconds"]["value"] * 1e3, 2), 8.62)
        self.assertEqual(round(v["ttft_lower_bound_seconds"]["value"] * 1e3, 1), 58.9)
        capacity = v["max_concurrent_requests"]
        self.assertIsNone(capacity["value"])
        self.assertIn("не оценивал", " ".join(capacity["notes"]))

    def test_fixed_state_from_flag_and_config(self) -> None:
        # chapter2.md:422: Qwen3.6 — 20 KiB KV на токен и фиксированное состояние
        # 61,875 MiB; вывод вручную: 12 GiB / (20 KiB × 8192 + 61.875 MiB) =
        # 12288 / 221.875 → 55 запросов (без состояния — 76)
        base = (
            "serving", "--peak-tflops", "1", "--bandwidth", "1e12", "--memory", "12884901888",
            "--weights", "0", "--weight-read", "0", "--decode-flops", "0",
            "--kv-per-token", "20480", "--context", "8192",
        )  # fmt: skip
        self.assertEqual(values(*base)["max_concurrent_requests"]["value"], 76)
        v = values(*base, "--fixed-state-bytes", "64880640")
        self.assertEqual(v["max_concurrent_requests"]["value"], 55)
        auto = values(*base, "--config", str(CONFIGS / "qwen3.6-35b-a3b.json"))
        capacity = auto["max_concurrent_requests"]
        self.assertEqual(capacity["value"], 55)
        self.assertEqual(capacity["inputs"]["fixed_state"], 64_880_640)
        self.assertIn("config", " ".join(capacity["notes"]))
        # состояние читается на каждом шаге (chapter2.md:454): R = B·(KV + S) при нулевых весах
        step = auto["tpot_lower_bound_seconds"]["value"]
        self.assertAlmostEqual(step, (20480 * 8192 + 64_880_640) / 1e12)

    def test_tp_capacity_book_table(self) -> None:
        # chapter6.md:154: «Число запросов 128K/32K на двух H100 | 2/10 | 7/29 | 31/117»,
        # chapter6.md:153: на одной H100 — 0/1, 1/5, 3/11; chapter6.md:139: Qwen3-32B на
        # четырёх картах — семь сессий 128K, на восьми — шестнадцать; резерв 2 GiB на карту
        qwen32 = values("model", "--config", ModelParallelCommandTest.QWEN3_32B)
        cases = (
            (
                "qwen3-32b.json",
                qwen32["weight_bytes"]["value"],
                262144,
                {1: (0, 1), 2: (2, 10), 4: (7, 28), 8: (16, 64)},
            ),
            ("qwen3-30b-a3b.json", 61_064_245_248, 98304, {1: (1, 5), 2: (7, 29)}),
            ("qwen3.6-35b-a3b.json", 69.32e9, 20480, {1: (3, 11), 2: (31, 117)}),
        )
        for config, weights, kv, expected in cases:
            for tp, counts in expected.items():
                for context, count in zip((131072, 32768), counts):
                    with self.subTest(config=config, tp=tp, context=context):
                        v = values(
                            "serving", "--device", "h100-sxm", "--config", str(CONFIGS / config),
                            "--tp", str(tp), "--weights", str(weights), "--weight-read", "0",
                            "--decode-flops", "0", "--kv-per-token", str(kv),
                            "--context", str(context), "--reserve", "2147483648",
                        )  # fmt: skip
                        self.assertEqual(v["max_concurrent_requests"]["value"], count)

    def test_tp_step_local_part_book(self) -> None:
        # chapter6.md:505-508: «Локальный доступ к памяти» 29,35 / 14,68 / 7,34 / 3,67 ms при
        # TP1/2/4/8; chapter6.md:225: 63,97 GB весов и 34,36 GB KV при s = 131 064
        for tp, expected in ((1, 29.35), (2, 14.68), (4, 7.34), (8, 3.67)):
            with self.subTest(tp=tp):
                v = values(
                    "serving", "--device", "h100-sxm", "--config",
                    ModelParallelCommandTest.QWEN3_32B, "--tp", str(tp),
                    "--weights", "65.52e9", "--weight-read", "63.97e9",
                    "--decode-flops", "0.34e12", "--kv-per-token", "262144",
                    "--context", "131064",
                )  # fmt: skip
                step = v["tpot_lower_bound_seconds"]
                self.assertEqual(round(step["value"] * 1e3, 2), expected)
                if tp > 1:
                    self.assertIn("AllReduce", " ".join(step["notes"]))

    def test_tp_price_counts_all_cards(self) -> None:
        # арифметическое тождество: TP2 при том же шаге удваивает цену токена;
        # 999999 байт при 1 GB/s и TP2 — шаг 0.5 ms, 2000 ток/с, 2 × 3.6 $/ч
        v = values(
            "serving", "--peak-tflops", "1", "--bandwidth", "1e9", "--memory", "1e9",
            "--config", QWEN3_8B, "--tp", "2", "--weights", "0", "--weight-read", "999999",
            "--decode-flops", "0", "--kv-per-token", "1", "--context", "1",
            "--price-per-hour", "3.6",
        )  # fmt: skip
        self.assertAlmostEqual(
            v["tokens_per_second_upper_bound"]["value"], 2000, places=0
        )
        self.assertAlmostEqual(
            v["cost_per_million_tokens_lower_bound"]["value"], 1.0, places=3
        )

    def test_tp_refuses_aggregate_device(self) -> None:
        # агрегат уже суммирует ёмкость и полосу всех карт; деление весов и KV на TP
        # поверх него считало бы карты дважды (замечание ревью: 3 118 вместо 388)
        message = fails(
            "serving", "--device", "gb200-nvl72", "--allow-aggregate",
            "--config", ModelParallelCommandTest.QWEN3_32B, "--tp", "8",
            "--weights", "65.52e9", "--weight-read", "63.97e9", "--decode-flops", "0.34e12",
            "--kv-per-token", "262144", "--context", "131072",
        )  # fmt: skip
        self.assertIn("агрегат", message)
        self.assertIn("--tp", message)
        self.assertIn("одиночное", message)

    def test_memory_context_shorter_than_context_is_refused(self) -> None:
        # память считается к концу генерации, шаг — при длине входа; обратное — ошибка
        message = fails(*self.RTX_12GIB, "--memory-context", "100")
        self.assertIn("--memory-context", message)
        self.assertIn("--context", message)

    def test_tp_step_inputs_are_per_device(self) -> None:
        v = values(
            "serving", "--device", "h100-sxm", "--config", str(CONFIGS / "qwen3.6-35b-a3b.json"),
            "--tp", "2", "--weights", "69.32e9", "--weight-read", "0", "--decode-flops", "0",
            "--kv-per-token", "20480", "--context", "8192",
        )  # fmt: skip
        step = v["tpot_lower_bound_seconds"]
        self.assertIn("kv_request_per_device", step["inputs"])
        self.assertNotIn("kv_request", step["inputs"])
        # chapter6.md:1164: ⌊(n·(80 GB − 2 GiB) − W)/S⌋ — состояние S делится на n карт
        notes = " ".join(v["max_concurrent_requests"]["notes"])
        self.assertIn("S/TP", notes)
        self.assertIn("chapter6.md:1164", notes)

    def test_quantized_weights_per_device_keep_lower_bound(self) -> None:
        # chapter1.md:205: 8-битные веса 73.73 GB против 70.55 GB по формуле
        config = str(CONFIGS / "deepseek-r1-distill-llama-70b.json")
        v = values("model", "--config", config, "--weight-dtype", "int8", "--tp", "2")
        item = v["weight_bytes_per_device"]
        self.assertEqual(item["bound"], "lower")
        self.assertIn("73.73", " ".join(item["notes"]))

    def test_mla_note_is_marked_as_derived(self) -> None:
        v = values("model", "--config", str(CONFIGS / "deepseek-v3.json"))
        note = " ".join(v["kv_bytes_per_token"]["notes"])
        self.assertIn("вывод", note)
        self.assertIn("2.3.2", note)

    AGGREGATE = (
        "serving", "--device", "gb200-nvl72", "--allow-aggregate",
        "--weights", "1e9", "--weight-read", "1e9", "--decode-flops", "1e9",
        "--kv-per-token", "1000", "--context", "10", "--batch", "8",
    )  # fmt: skip

    def test_aggregate_price_needs_scope(self) -> None:
        # цена за час карты против цены за час стойки различается в 72 раза
        message = fails(*self.AGGREGATE, "--price-per-hour", "3")
        self.assertIn("--price-scope", message)
        self.assertIn("72", message)

    def test_aggregate_price_scopes(self) -> None:
        # арифметическое тождество: card — цена × 72 карты, aggregate — как задано
        card = values(*self.AGGREGATE, "--price-per-hour", "3", "--price-scope", "card")
        rack = values(
            *self.AGGREGATE, "--price-per-hour", "216", "--price-scope", "aggregate"
        )
        tok_s = card["tokens_per_second_upper_bound"]["value"]
        expected = 3 * 72 / 3600 / tok_s * 1e6
        for v in (card, rack):
            self.assertAlmostEqual(
                v["cost_per_million_tokens_lower_bound"]["value"], expected
            )
        self.assertEqual(
            card["cost_per_million_tokens_lower_bound"]["inputs"]["price_scope"], "card"
        )
        self.assertEqual(
            rack["cost_per_million_tokens_lower_bound"]["inputs"]["price_scope"],
            "aggregate",
        )
        # --reserve на агрегате резервирует память всей стойки
        notes = " ".join(card["max_concurrent_requests"]["notes"])
        self.assertIn("--reserve", notes)

    def test_single_device_price_unchanged(self) -> None:
        # без --price-scope одиночное устройство считает цену за карту, как раньше
        base = (*ServingCommandTest.RTX, "--price-per-hour", "2")
        plain = values(*base)["cost_per_million_tokens_lower_bound"]
        card = values(*base, "--price-scope", "card")[
            "cost_per_million_tokens_lower_bound"
        ]
        self.assertEqual(plain["value"], card["value"])
        tok_s = values(*base)["tokens_per_second_upper_bound"]["value"]
        self.assertAlmostEqual(plain["value"], 2 / 3600 / tok_s * 1e6)

    def test_aggregate_price_follows_where_rates_come_from(self) -> None:
        # замечание ревью 20a-1: все значения на карту заданы явно, а цена умножалась на
        # 72; арифметическое тождество — цена карты как есть: 3 / 3600 / tok_s × 10^6
        user = (
            "serving", "--device", "gb200-nvl72", "--allow-aggregate",
            "--memory", "186e9", "--bandwidth", "8e12", "--peak-tflops", "2500",
            "--weights", "1e9", "--weight-read", "1e9", "--decode-flops", "1e9",
            "--kv-per-token", "1000", "--context", "10", "--batch", "8",
            "--price-per-hour", "3",
        )  # fmt: skip
        for extra in ((), ("--price-scope", "card")):
            with self.subTest(extra=extra):
                v = values(*user, *extra)
                tok_s = v["tokens_per_second_upper_bound"]["value"]
                price = v["cost_per_million_tokens_lower_bound"]
                self.assertAlmostEqual(price["value"], 3 / 3600 / tok_s * 1e6)
                self.assertIn("заданы явно", " ".join(price["notes"]))
                # память задана явно: резерв не «на весь агрегат»
                notes = " ".join(v["max_concurrent_requests"]["notes"])
                self.assertNotIn("на весь агрегат", notes)
        message = fails(*user, "--price-scope", "aggregate")
        self.assertIn("--price-scope aggregate", message)

    def test_reserve_note_only_for_snapshot_memory(self) -> None:
        # ёмкость из снимка — резерв на всю запись; пики из снимка — цена × 72
        v = values(*self.AGGREGATE, "--price-per-hour", "3", "--price-scope", "card")
        self.assertIn(
            "--reserve резервирует память на весь агрегат",
            " ".join(v["max_concurrent_requests"]["notes"]),
        )
        v = values(
            *self.AGGREGATE, "--memory", "186e9", "--price-per-hour", "3",
            "--price-scope", "card",
        )  # fmt: skip
        self.assertNotIn(
            "на весь агрегат", " ".join(v["max_concurrent_requests"]["notes"])
        )
        tok_s = v["tokens_per_second_upper_bound"]["value"]
        self.assertAlmostEqual(
            v["cost_per_million_tokens_lower_bound"]["value"],
            3 * 72 / 3600 / tok_s * 1e6,
        )

    def test_aggregate_rates_are_overridden_together(self) -> None:
        # пик карты с полосой стойки (или наоборот) смешал бы охваты в одной границе
        work = {
            "serving": (
                "--weights", "1e9", "--weight-read", "1e9", "--decode-flops", "1e9",
                "--kv-per-token", "1000", "--context", "10",
            ),
            "roofline": ("--flops", "1", "--bytes", "1"),
            "batch-threshold": (
                "--weight-read", "1e9", "--kv-per-token", "1000", "--context", "10",
                "--decode-flops", "1e9",
            ),
        }  # fmt: skip
        for command, args in work.items():
            for rate in (("--peak-tflops", "2500"), ("--bandwidth", "8e12")):
                with self.subTest(command=command, rate=rate):
                    message = fails(
                        command, "--device", "gb200-nvl72", "--allow-aggregate",
                        *rate, *args,
                    )  # fmt: skip
                    self.assertIn("--peak-tflops", message)
                    self.assertIn("--bandwidth", message)
                    self.assertIn("вместе", message)

    def test_price_scope_aggregate_needs_an_aggregate(self) -> None:
        # одиночная карта: aggregate помечал бы цену агрегатом, а считал бы как карту
        message = fails(
            *ServingCommandTest.RTX,
            "--price-per-hour",
            "2",
            "--price-scope",
            "aggregate",
        )
        self.assertIn("--price-scope aggregate", message)
        self.assertIn("агрегат", message)

    def test_price_scope_needs_a_price(self) -> None:
        message = fails(*ServingCommandTest.RTX, "--price-scope", "card")
        self.assertIn("--price-scope", message)
        self.assertIn("--price-per-hour", message)

    def test_price_help_names_the_aggregate_record(self) -> None:
        # агрегаты снимка — не только стойки: суперчипы, серверы, суперкластеры
        code, out = call("serving", "--help")
        self.assertEqual(code, 0)
        text = " ".join(out.split())
        self.assertIn("всей записи агрегата", text)
        self.assertNotIn("всей стойки", text)

    def test_mla_serving_note_is_marked_as_derived(self) -> None:
        v = values(
            "serving", "--device", "h100-sxm", "--config", str(CONFIGS / "deepseek-v3.json"),
            "--tp", "8", "--weights", "1e9", "--weight-read", "1e9", "--decode-flops", "1e9",
            "--kv-per-token", "70272", "--context", "8192",
        )  # fmt: skip
        notes = " ".join(v["tpot_lower_bound_seconds"]["notes"])
        self.assertIn("вывод из 2.3.2 и 6.2.2", notes)
        message = fails(*self.RTX_12GIB, "--tp", "2")
        self.assertIn("вывод из 2.3.2 и 6.2.2", message)

    def test_tp_needs_config(self) -> None:
        message = fails(*self.RTX_12GIB, "--tp", "2")
        self.assertIn("--config", message)
        message = fails(*self.RTX_12GIB, "--tp", "3", "--config", QWEN3_8B)
        self.assertIn("голов", message)


# Синтетический конфиг формы Mistral-7B-v0.1: sliding_window на всех слоях. Чисел книги
# для окна нет; эталоны ниже выведены вручную из формул serving.
MISTRAL_WINDOW = {
    "model_type": "mistral", "hidden_size": 4096, "intermediate_size": 14336,
    "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8,
    "vocab_size": 32000, "tie_word_embeddings": False, "sliding_window": 4096,
}  # fmt: skip


class ServingWindowTest(unittest.TestCase):
    """serving --config с окном внимания: KV запроса — последние min(длина, окно) токенов."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = Path(tmp.name) / "mistral-7b.json"
        self.config.write_text(json.dumps(MISTRAL_WINDOW))

    def serving(self, *extra: str, config: bool = True) -> dict[str, dict[str, Any]]:
        # веса и чтение весов — model по этому конфигу: 14 483 464 192 и 14 221 320 192 B;
        # KV на токен 2·32·8·128·2 = 131 072 B
        return values(
            "serving", "--device", "h100-sxm",
            *(("--config", str(self.config)) if config else ()),
            "--weights", "14483464192", "--weight-read", "14221320192",
            "--decode-flops", "0", "--kv-per-token", "131072", *extra,
        )  # fmt: skip

    def test_window_caps_step_and_capacity(self) -> None:
        # окно 4096 при контексте 32 768: шаг читает KV 4096 токенов каждого запроса,
        # (14 221 320 192 + 16·131 072·4096) / 3.35e12 = 22 811 254 784 / 3.35e12 =
        # 6.80933 ms (без окна — 82 940 796 928 B, 24.76 ms); ёмкость
        # ⌊(80e9 − 14 483 464 192) / (131 072·4096)⌋ = ⌊122.03⌋ = 122 (без окна — 15)
        v = self.serving("--context", "32768", "--batch", "16")
        step = v["tpot_lower_bound_seconds"]
        self.assertEqual(step["bound"], "lower")
        self.assertAlmostEqual(step["value"] * 1e3, 6.809329786, places=6)
        self.assertEqual(step["inputs"]["window"], 4096)
        self.assertEqual(step["input_units"]["window"], "tok")
        self.assertEqual(step["inputs"]["kv_request"], 131072 * 4096)
        self.assertAlmostEqual(
            v["tokens_per_second_upper_bound"]["value"], 16 / step["value"]
        )
        capacity = v["max_concurrent_requests"]
        self.assertEqual(capacity["value"], 122)
        self.assertEqual(capacity["bound"], "upper")
        self.assertIn("min(memory_context, window)", capacity["formula"])
        for item in (step, capacity):
            self.assertIn("всего контекста", " ".join(item["notes"]))
        full = self.serving("--context", "32768", "--batch", "16", config=False)
        self.assertEqual(full["max_concurrent_requests"]["value"], 15)
        self.assertEqual(
            round(full["tpot_lower_bound_seconds"]["value"] * 1e3, 2), 24.76
        )

    def test_window_applies_to_memory_context(self) -> None:
        # шаг при 2048 < окна — без изменений: (14 221 320 192 + 16·131 072·2048) / 3.35e12
        # = 5.52725 ms; память при 8192 — 4096 токенов на запрос, как выше: 122
        v = self.serving(
            "--context", "2048", "--memory-context", "8192", "--batch", "16"
        )
        self.assertEqual(v["max_concurrent_requests"]["value"], 122)
        step = v["tpot_lower_bound_seconds"]
        self.assertAlmostEqual(step["value"] * 1e3, 5.527249996, places=6)
        self.assertEqual(step["inputs"]["kv_request"], 131072 * 2048)
        self.assertNotIn("всего контекста", " ".join(step["notes"]))

    def test_window_with_shared_prefix(self) -> None:
        # префикс 3072 при длине 6144 и окне 4096: в окне последних 4096 токенов
        # собственных min(6144 − 3072, 4096) = 3072 и 4096 − 3072 = 1024 токена префикса.
        # Ёмкость ⌊(80e9 − 14 483 464 192 − 1024·131 072) / (3072·131 072)⌋ =
        # ⌊65 382 318 080 / 402 653 184⌋ = ⌊162.38⌋ = 162 (без окна — 161).
        # Шаг, batch 16: (14 221 320 192 + 1024·131 072 + 16·3072·131 072) / 3.35e12 =
        # 20 797 988 864 / 3.35e12 = 6.20835 ms; без дедупликации префикса —
        # 16·4096 токенов, 6.80933 ms
        v = self.serving(
            "--context", "6144", "--batch", "16", "--shared-prefix-tokens", "3072"
        )
        capacity = v["max_concurrent_requests"]
        self.assertEqual(capacity["value"], 162)
        self.assertIn("134 217 728 B на карту", " ".join(capacity["notes"]))
        step = v["tpot_lower_bound_seconds"]
        self.assertAlmostEqual(step["value"] * 1e3, 6.208354885, places=6)
        self.assertEqual(step["inputs"]["kv_shared_prefix"], 1024 * 131072)
        self.assertEqual(step["inputs"]["kv_request"], 3072 * 131072)
        full = v["tpot_without_prefix_dedup_seconds"]
        self.assertAlmostEqual(full["value"] * 1e3, 6.809329786, places=6)
        plain = self.serving(
            "--context", "6144", "--batch", "16", "--shared-prefix-tokens", "3072",
            config=False,
        )  # fmt: skip
        self.assertEqual(plain["max_concurrent_requests"]["value"], 161)

    def test_window_longer_than_context_changes_nothing(self) -> None:
        # 2048 < 4096: ⌊(80e9 − 14 483 464 192) / (131 072·2048)⌋ = 244, как без config
        v = self.serving("--context", "2048")
        self.assertEqual(v["max_concurrent_requests"]["value"], 244)
        plain = self.serving("--context", "2048", config=False)
        for name in ("max_concurrent_requests", "tpot_lower_bound_seconds"):
            self.assertEqual(v[name]["value"], plain[name]["value"])
        self.assertNotIn(
            "всего контекста", " ".join(v["max_concurrent_requests"]["notes"])
        )


# Все команды с корректными входами: для проверок якорей и единиц, а не чисел книги
COMMANDS = tuple(
    shlex.split(line)
    for line in (
        f"model --config {QWEN3_8B} --context 2048",
        f"model --config {QWEN3_8B} --context 2048 --weight-dtype int8",
        f"model --config {CONFIGS / 'qwen3.5-397b-a17b.json'} --params 397e9",
        f"model --config {CONFIGS / 'qwen3-30b-a3b.json'} --context 64",
        "roofline --device h100-sxm --flops 1 --bytes 1",
        "roofline --device gb200-nvl72 --allow-aggregate --flops 1 --bytes 1",
        " ".join(ServingCommandTest.RTX) + " --price-per-hour 2",
        (
            f"training --config {QWEN3_8B} --tokens 8192 --dp 8 --total-tokens 1e12"
            " --devices 8 --mfu 0.4 --device h100-sxm"
        ),
        (
            "checkpoint --checkpoint-bytes 1e9 --save-bandwidth 1e9 --devices 8"
            " --device-mtbf 1e6 --stages 4 --microbatches 8"
        ),
        "speculative --acceptance 0.5 --draft 4 --plain-step 0.01",
        (
            "pipeline --stages 4 --microbatches 8 --virtual-stages 2"
            " --forward-seconds 0.01 --backward-seconds 0.02"
        ),
        f"training-state --config {QWEN3_8B} --dp 8 --stage 2",
        (
            "checkpoint --checkpoint-bytes 1e9 --save-bandwidth 1e9 --devices 8"
            " --device-mtbf 1e6 --common-job-mtbf 604800"
        ),
        "ring --devices 8 --message 1 --bandwidth 1 --alpha 0",
        "allreduce --devices 8 --message 10240 --bandwidth 450e9 --alpha 0.822e-6",
        (f"batch-threshold --config {QWEN3_8B} --context 128 --device h100-sxm"),
        "speculative --round 0.0015:1 --round 0.0015:5",
        "roofline --memory-only --device m3-ultra-80gpu-256gb --flops 1 --bytes 1",
        f"model --config {CONFIGS / 'qwen3-30b-a3b.json'} --context 64 --batch 4 --tp 2",
        f"model --config {CONFIGS / 'qwen3.6-35b-a3b.json'} --context 64 --tp 2",
        (
            f"serving --device h100-sxm --config {CONFIGS / 'qwen3.6-35b-a3b.json'} --tp 2"
            " --weights 69.32e9 --weight-read 5.89e9 --decode-flops 1e9 --prefill-flops 1e12"
            " --kv-per-token 20480 --context 4096 --memory-context 4608 --price-per-hour 2"
        ),
        (
            "serving --device rtx-pro6000-blackwell-ws --memory 12884901888 --weights 0"
            " --weight-read 1 --decode-flops 1 --kv-per-token 147456 --context 8192"
            " --memory-context 8448 --shared-prefix-tokens 6144"
        ),
        (
            "serving --peak-tflops 1 --bandwidth 1e12 --weights 1 --weight-read 1"
            " --decode-flops 1 --kv-per-token 1 --context 1"
        ),
        "cost --input-tokens 1 --input-price 1",
        (
            "edge --upload '1 MB' --download '1 MB' --up-mbps 1 --down-mbps 1"
            " --rtt 0 --compute 0"
        ),
        "device --device gb200-nvl72",
        "units --size '141.11 GB' --to GiB --to MiB --mbps 400000 --dtype int8",
        (
            "queueing --class a=8192:256 --class b=1024:2048 --rate a=2 --rate b=2"
            " --decode-capacity 2796 --prefill-capacity 1e5"
            " --arrival-rate 10 --time-in-system 30"
        ),
        (
            "queueing --class a=8192:256 --rate a=2 --decode-capacity 2796"
            " --capacity-upper-bound --anchor ch08"
        ),
    )
)


class AnchorTest(unittest.TestCase):
    def test_every_result_anchor_is_a_heading(self) -> None:
        for argv in COMMANDS:
            for item in run(*argv)["results"]:
                with self.subTest(command=argv[0], result=item["name"]):
                    path, line = item["anchor"].rsplit(":", 1)
                    lines = (SKILL / path).read_text(encoding="utf-8").splitlines()
                    self.assertTrue(lines[int(line) - 1].startswith("#"))


class InputUnitsTest(unittest.TestCase):
    def test_every_input_has_a_unit(self) -> None:
        # безразмерное число допустимо только из явного списка cli.DIMENSIONLESS
        for argv in COMMANDS:
            for item in run(*argv)["results"]:
                with self.subTest(command=argv[0], result=item["name"]):
                    units = item["input_units"]
                    self.assertEqual(set(item["inputs"]), set(units))
                    for key, value in item["inputs"].items():
                        numeric = isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        )
                        if numeric and units[key] == "":
                            self.assertIn(key, cli.DIMENSIONLESS)

    def test_markdown_prints_input_units(self) -> None:
        code, out = call(
            "roofline", "--device", "h100-sxm", "--flops", "140e9", "--bytes", "70e9"
        )
        self.assertEqual(code, 0)
        self.assertIn("peak=9.894e+14 FLOP/s", out)
        self.assertIn("bandwidth=3.35e+12 B/s", out)
        self.assertIn("flops=1.4e+11 FLOP", out)

    def test_result_requires_unit_for_every_input(self) -> None:
        with self.assertRaises(ValueError):
            Result("x", 1.0, "s", "F/Π", {"flops": 1.0}, "a")


class UsageErrorTest(unittest.TestCase):
    def test_missing_argument_json(self) -> None:
        message = fails("roofline", "--flops", "1")
        self.assertIn("--bytes", message)
        self.assertIn("обязательн", message)

    def test_missing_argument_markdown(self) -> None:
        code, out = call("roofline", "--flops", "1")
        self.assertEqual(code, 2)
        self.assertTrue(out.startswith("Ошибка:"), out)
        self.assertIn("--bytes", out)

    def test_invalid_number_and_choice(self) -> None:
        message = fails("roofline", "--flops", "много", "--bytes", "1")
        self.assertIn("--flops", message)
        self.assertIn("много", message)
        message = fails(
            "roofline", "--sparsity", "unspecified", "--flops", "1", "--bytes", "1"
        )
        self.assertIn("--sparsity", message)
        self.assertIn("dense", message)

    def test_unknown_command(self) -> None:
        self.assertIn("model", fails("modle"))

    def test_messages_name_cli_flags(self) -> None:
        # ошибка входа называет флаг CLI, а не имя параметра функции
        serving = " ".join(ServingCommandTest.RTX)
        for line, flag in (
            (f"model --config {QWEN3_8B} --context -1", "--context"),
            ("roofline --device h100-sxm --flops -1 --bytes 1", "--flops"),
            ("roofline --device h100-sxm --flops 1 --bytes -1", "--bytes"),
            (serving.replace("147456", "0"), "--kv-per-token"),
            (serving + " --batch 0", "--batch"),
            (serving + " --memory -1", "--memory"),
            (serving + " --reserve -1", "--reserve"),
            (serving + " --price-per-hour -1", "--price-per-hour"),
            (f"training --config {QWEN3_8B} --tokens 0", "--tokens"),
            (f"training --config {QWEN3_8B} --tokens 8 --dp 0", "--dp"),
            (
                (
                    f"training --config {QWEN3_8B} --tokens 8 --total-tokens 1e9"
                    " --devices 8 --mfu 1.5 --peak-tflops 1"
                ),
                "--mfu",
            ),
            (
                (
                    "checkpoint --checkpoint-bytes 1e9 --save-bandwidth 1e9 --devices 8"
                    " --device-mtbf 1e6 --recovery -1"
                ),
                "--recovery",
            ),
            (
                (
                    "checkpoint --checkpoint-bytes 0 --save-bandwidth 1e9 --devices 8"
                    " --device-mtbf 1e6"
                ),
                "--checkpoint-bytes",
            ),
            ("speculative --acceptance 2 --draft 4 --plain-step 0.01", "--acceptance"),
            ("speculative --acceptance 0.5 --draft -1 --plain-step 0.01", "--draft"),
            ("ring --devices 0 --message 1 --bandwidth 1 --alpha 0", "--devices"),
            ("ring --devices 8 --message 1 --bandwidth 0 --alpha 0", "--bandwidth"),
            ("cost --input-tokens -1 --input-price 1", "--input-tokens"),
            ("cost --input-tokens 1 --input-price nan", "--input-price"),
            ("cost --input-tokens 1 --input-price 1 --success 0", "--success"),
            (
                (
                    "edge --upload '1 MB' --download '1 MB' --up-mbps 0 --down-mbps 1"
                    " --rtt 0 --compute 0"
                ),
                "--up-mbps",
            ),
            ("units --mbps -1", "--mbps"),
            ("queueing --arrival-rate -1 --time-in-system 1", "--arrival-rate"),
            (
                "queueing --class a=1:1 --rate a=1 --decode-capacity 0",
                "--decode-capacity",
            ),
        ):
            with self.subTest(line=line):
                self.assertIn(flag, fails(*shlex.split(line)))


class ReviewFixesTest(unittest.TestCase):
    def test_six_nd_uses_active_parameters_for_moe(self) -> None:
        # chapter3.md:631: «Общий объём весов влияет на ёмкость, а активные параметры
        # дают лишь грубую оценку объёма вычислений»; активные параметры qwen3-30b-a3b —
        # parameters − routed_expert_total + routed_expert_active_per_token
        # (qwen3-30b-a3b-decode-b1-s8192.json, как в test_accounting)
        active = 30_532_122_624 - 28_991_029_248 + 1_811_939_328
        v = values(
            "training", "--config", str(CONFIGS / "qwen3-30b-a3b.json"),
            "--tokens", "4096", "--dp", "8",
        )  # fmt: skip
        six_nd = v["six_nd_flops_per_sequence"]
        self.assertEqual(six_nd["value"], 6 * active * 4096)
        self.assertEqual(six_nd["inputs"]["active_parameters"], active)
        self.assertIn("ёмкост", " ".join(six_nd["notes"]))
        # состояние ZeRO хранит все параметры, а не активные
        self.assertEqual(v["zero0_state_bytes_per_gpu"]["value"], 16 * 30_532_122_624)

    def test_moe_note_groups_digits(self) -> None:
        # qwen3-30b-a3b-decode-b1-s8192.json: 30 532 122 624 параметров, из них активных
        # 30 532 122 624 − 28 991 029 248 + 1 811 939 328 = 3 353 032 704; целые в
        # примечаниях — с разрядами через пробел, как в значениях результатов
        v = values(
            "training", "--config", str(CONFIGS / "qwen3-30b-a3b.json"),
            "--tokens", "4096", "--dp", "8",
        )  # fmt: skip
        notes = " ".join(v["six_nd_flops_per_sequence"]["notes"])
        self.assertIn("3 353 032 704 активных параметров из 30 532 122 624", notes)

    def test_fixed_state_notes_group_digits(self) -> None:
        # chapter2.md:422: фиксированное состояние Qwen3.6 — 64 880 640 байт на запрос
        base = (
            "serving", "--peak-tflops", "1", "--bandwidth", "1e12", "--memory", "12884901888",
            "--weights", "0", "--weight-read", "0", "--decode-flops", "0",
            "--kv-per-token", "20480", "--context", "8192",
            "--config", str(CONFIGS / "qwen3.6-35b-a3b.json"),
        )  # fmt: skip
        auto = " ".join(values(*base)["max_concurrent_requests"]["notes"])
        self.assertIn("64 880 640 B на запрос", auto)
        given = values(*base, "--fixed-state-bytes", "1000")
        self.assertIn(
            "64 880 640 B на запрос",
            " ".join(given["max_concurrent_requests"]["notes"]),
        )

    def test_concurrency_is_an_upper_bound(self) -> None:
        v = values(*ServingCommandTest.RTX)
        self.assertEqual(v["max_concurrent_requests"]["bound"], "upper")

    def test_aggregate_note_names_only_snapshot_quantities(self) -> None:
        base = ("roofline", "--device", "gb200-nvl72", "--allow-aggregate")
        work = ("--flops", "1", "--bytes", "1")
        # пик задан явно, полоса из снимка — отказ (test_aggregate_rates_are_overridden_together);
        # оба заданы явно — ни одна величина не из агрегата, примечания нет
        v = values(*base, "--peak-tflops", "2500", "--bandwidth", "8e12", *work)
        self.assertEqual(v["step_lower_bound_seconds"]["notes"], [])

    def test_aggregate_memory_note_reads_correctly(self) -> None:
        v = values(
            "serving", "--device", "gb200-nvl72", "--allow-aggregate",
            "--weights", "1e9", "--weight-read", "1e9", "--decode-flops", "1e9",
            "--kv-per-token", "1000", "--context", "10",
        )  # fmt: skip
        notes = " ".join(v["max_concurrent_requests"]["notes"])
        self.assertIn("сумма по 72 устройствам", notes)
        self.assertIn("ёмкость памяти", notes)
        self.assertNotIn("суммированы", notes)

    def test_user_parameters_of_wrapper_are_not_annotated_as_text_only(self) -> None:
        config = str(CONFIGS / "qwen3.5-397b-a17b.json")
        v = values("model", "--config", config, "--params", "1000")
        notes = " ".join(v["parameters"]["notes"])
        self.assertNotIn("не учтены", notes)
        self.assertNotIn("передайте", notes)
        self.assertIn("--params", notes)
        self.assertIn("карточк", notes)

    def test_quantized_weights_are_a_lower_bound(self) -> None:
        # chapter1.md:205: «в 8-битной схеме занимают около 73.73 GB»; chapter1.md:209:
        # BF16 «141.11 GB», половина — «около 70.55 GB»: параметры × 1 байт не учитывают
        # scale и части модели в BF16
        config = str(CONFIGS / "deepseek-r1-distill-llama-70b.json")
        v = values("model", "--config", config, "--weight-dtype", "int8")
        weights = v["weight_bytes"]
        self.assertEqual(round(weights["value"] / 1e9, 2), 70.55)
        self.assertEqual(weights["bound"], "lower")
        self.assertIn("73.73", " ".join(weights["notes"]))
        self.assertEqual(v["decode_weight_read_bytes"]["bound"], "lower")
        bf16 = values("model", "--config", config)["weight_bytes"]
        self.assertIsNone(bf16["bound"])
        # накладные расходы квантизации, известные пользователю, прибавляются явно
        v = values(
            "model", "--config", config, "--weight-dtype", "int8",
            "--quant-overhead-bytes", "3.18e9",
        )  # fmt: skip
        self.assertEqual(round(v["weight_bytes"]["value"] / 1e9, 2), 73.73)
        self.assertIsNone(v["weight_bytes"]["bound"])
        self.assertIn(
            "--quant-overhead-bytes", fails("model", "--config", config,
            "--quant-overhead-bytes", "1")
        )  # fmt: skip

    def test_training_seconds_identity(self) -> None:
        # Арифметическое тождество, не число книги: две последовательности на двух
        # устройствах при 1000 TFLOP/s и MFU 0.5 — F_seq / (1e15 · 0.5)
        v = values(
            "training", "--config", QWEN3_8B, "--tokens", "8192",
            "--total-tokens", "16384", "--devices", "2",
            "--peak-tflops", "1000", "--mfu", "0.5",
        )  # fmt: skip
        per_sequence = v["training_flops_per_sequence"]["value"]
        self.assertAlmostEqual(
            v["training_seconds"]["value"], per_sequence / (1e15 * 0.5)
        )

    def test_cost_per_million_tokens_identity(self) -> None:
        # Арифметическое тождество, не число книги: шаг 1 ms (10^6 байт при 1 GB/s)
        # даёт 1000 токенов/с, 3.6 $/ч = 0.001 $/с — миллион токенов за 1 $
        v = values(
            "serving", "--peak-tflops", "1", "--bandwidth", "1e9", "--memory", "1e9",
            "--weights", "0", "--weight-read", "999999", "--decode-flops", "0",
            "--kv-per-token", "1", "--context", "1", "--price-per-hour", "3.6",
        )  # fmt: skip
        self.assertAlmostEqual(v["tokens_per_second_upper_bound"]["value"], 1000)
        price = v["cost_per_million_tokens_lower_bound"]
        self.assertAlmostEqual(price["value"], 1.0)
        self.assertEqual(price["bound"], "lower")


class UnitsCommandTest(unittest.TestCase):
    def test_book_conversions(self) -> None:
        v = values(
            "units", "--size", "141.11 GB", "--to", "GiB", "--mbps", "400000",
            "--dtype", "int8",
        )  # fmt: skip
        self.assertEqual(v["size_bytes"]["value"], 141.11e9)
        # chapter1.md:199: «141.11 GB — это примерно 131.42 GiB»
        self.assertEqual(round(v["size_GiB"]["value"], 2), 131.42)
        # chapter1.md:221: «ConnectX-7 с пропускной способностью 400 Gbit/s ... равна 50 GB/s»
        self.assertEqual(v["link_bytes_per_second"]["value"], 50e9)
        self.assertEqual(v["dtype_bytes"]["value"], 1)

    def test_size_without_unit_is_refused(self) -> None:
        self.assertIn("единиц", fails("units", "--size", "141.11"))

    def test_something_to_convert_is_required(self) -> None:
        self.assertIn("--size", fails("units"))


class QueueingCommandTest(unittest.TestCase):
    CLASSES = ("--class", "long_input=8192:256", "--class", "long_output=1024:2048")

    def test_example_3_1(self) -> None:
        # chapter3.md:89: «| Равномерная смесь | 18,432 | 4,604 |»
        v = values(
            "queueing", *self.CLASSES, "--rate", "long_input=2",
            "--rate", "long_output=2", "--decode-capacity", "2796",
        )  # fmt: skip
        self.assertEqual(v["input_tokens_per_second"]["value"], 18_432)
        self.assertEqual(v["decode_steps_per_second"]["value"], 4_604)
        # chapter3.md:137: «одна карта может выполнять не более приблизительно 2,796 шага
        # decode в секунду, что ниже средней потребности равномерной смеси в 4,604 шага»
        self.assertGreater(v["decode_utilization"]["value"], 1)

    def test_second_minute_of_changing_mix(self) -> None:
        # chapter3.md:91: «| Изменяющаяся во времени смесь, вторая минута | 6,963.2 | 7,471.2 |»
        v = values(
            "queueing", *self.CLASSES, "--rate", "long_input=0.4",
            "--rate", "long_output=3.6",
        )  # fmt: skip
        self.assertAlmostEqual(v["input_tokens_per_second"]["value"], 6_963.2)
        self.assertAlmostEqual(v["decode_steps_per_second"]["value"], 7_471.2)

    def test_littles_law(self) -> None:
        # chapter11.md:76: «каждую секунду использовать среду начинают 10 задач»,
        # «Каждая задача занимает среду на 30 секунд» — в среднем 300 сред
        v = values("queueing", "--arrival-rate", "10", "--time-in-system", "30")
        self.assertEqual(v["in_system"]["value"], 300)

    def test_capacity_upper_bound_marks_utilization(self) -> None:
        # chapter3.md:137: 2,796 шага decode в секунду — предел одной карты из границы
        # времени шага, то есть верхняя граница мощности; загрузка 4604/2796 ≥ 1.65 —
        # нижняя граница
        v = values(
            "queueing", *self.CLASSES, "--rate", "long_input=2",
            "--rate", "long_output=2", "--decode-capacity", "2796",
            "--capacity-upper-bound",
        )  # fmt: skip
        item = v["decode_utilization"]
        self.assertEqual(item["bound"], "lower")
        self.assertAlmostEqual(item["value"], 4604 / 2796)
        self.assertIn("верхняя граница", " ".join(item["notes"]))
        plain = values(
            "queueing", *self.CLASSES, "--rate", "long_input=2",
            "--rate", "long_output=2", "--decode-capacity", "2796",
        )  # fmt: skip
        self.assertIsNone(plain["decode_utilization"]["bound"])

    def test_anchor_follows_use(self) -> None:
        args = (*self.CLASSES, "--rate", "long_input=2", "--rate", "long_output=2")
        little = ("--arrival-rate", "4", "--time-in-system", "0.5")
        default = values("queueing", *args, *little)
        self.assertEqual(
            default["decode_steps_per_second"]["anchor"],
            "references/source-book/chapter3.md:75",
        )
        self.assertEqual(
            default["in_system"]["anchor"], "references/source-book/chapter11.md:62"
        )
        for choice, anchor in (
            ("ch08", "references/source-book/chapter8.md:598"),
            ("ch11", "references/source-book/chapter11.md:62"),
            ("ch03", "references/source-book/chapter3.md:75"),
        ):
            with self.subTest(choice=choice):
                v = values("queueing", *args, *little, "--anchor", choice)
                self.assertEqual({item["anchor"] for item in v.values()}, {anchor})

    def test_capacity_upper_bound_without_capacity_is_refused(self) -> None:
        message = fails(
            "queueing", *self.CLASSES, "--rate", "long_input=2",
            "--rate", "long_output=2", "--capacity-upper-bound",
        )  # fmt: skip
        self.assertIn("--capacity-upper-bound", message)
        self.assertIn("--decode-capacity", message)

    def test_class_errors_agree_in_gender(self) -> None:
        message = fails("queueing", "--class", "a=1:1", "--rate", "a=-1")
        self.assertIn("не может быть отрицательной", message)
        message = fails("queueing", "--class", "a=1:0", "--rate", "a=1")
        self.assertIn("должны быть не меньше 1", message)

    def test_malformed_inputs(self) -> None:
        self.assertIn("long_input", fails("queueing", *self.CLASSES, "--rate", "x=1"))
        self.assertIn(
            "--class", fails("queueing", "--class", "a=8192", "--rate", "a=1")
        )
        self.assertIn(
            "--rate", fails("queueing", *self.CLASSES, "--rate", "long_input")
        )
        self.assertIn("--time-in-system", fails("queueing", "--arrival-rate", "1"))


if __name__ == "__main__":
    unittest.main()
