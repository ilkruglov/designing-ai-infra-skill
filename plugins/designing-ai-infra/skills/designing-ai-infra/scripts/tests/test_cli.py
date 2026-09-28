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
    "references/source-book/chapter6.md:473",
    "references/source-book/chapter6.md:705",
    "references/source-book/chapter8.md:536",
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
        message = fails("model", "--config", QWEN3_8B, "--params", "1000")
        self.assertIn("8190735360", message)


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

    def test_concurrency_is_an_upper_bound(self) -> None:
        v = values(*ServingCommandTest.RTX)
        self.assertEqual(v["max_concurrent_requests"]["bound"], "upper")

    def test_aggregate_note_names_only_snapshot_quantities(self) -> None:
        base = ("roofline", "--device", "gb200-nvl72", "--allow-aggregate")
        work = ("--flops", "1", "--bytes", "1")
        v = values(*base, "--peak-tflops", "2500", *work)
        note = " ".join(v["step_lower_bound_seconds"]["notes"])
        self.assertIn("пропускная способность", note)
        self.assertNotIn("пик", note)
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
