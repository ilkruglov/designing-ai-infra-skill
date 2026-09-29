from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIRECTORY = ROOT / "plugins" / "designing-ai-infra"
TRIGGER_SET = PLUGIN_DIRECTORY / "evals" / "trigger-evals.json"
BENCHMARK = PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
PLAYBOOKS = (
    PLUGIN_DIRECTORY / "skills" / "designing-ai-infra" / "references" / "playbooks"
)
REQUIRED_KEYS = {"id", "query", "should_trigger", "reason"}
# Префиксы id: p — целевой запрос, w — агентная нагрузка на инфраструктуру,
# x — логика агента (уходит в developing-ai-agents), n — похожий словарь
# или посторонняя задача.
FALSE_CATEGORIES = ("near-miss:", "unrelated:")
# Доля общих слов (Jaccard), выше которой запрос считается пересказом промпта
# benchmark. Самые близкие пары набора дают около 0,16; пересказ с заменой
# пары слов даёт больше 0,8. Порог 0,5 ловит пересказ и не мешает общей лексике.
MAX_BENCHMARK_OVERLAP = 0.5
# Доля слов запроса, взятых из промпта benchmark (|q ∩ p| / |q|). Jaccard не
# ловит короткий запрос, вырезанный из длинного промпта: объединение велико.
# Самая близкая пара набора даёт около 0,37; порог 0,6.
MAX_BENCHMARK_CONTAINMENT = 0.6


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def overlap(left: str, right: str) -> float:
    a, b = words(left), words(right)
    return len(a & b) / len(a | b)


def containment(query: str, prompt: str) -> float:
    q = words(query)
    return len(q & words(prompt)) / len(q)


def load_queries() -> list[dict]:
    payload = json.loads(TRIGGER_SET.read_text(encoding="utf-8"))
    return payload["queries"]


class TriggerSetTests(unittest.TestCase):
    def test_skill_name_matches_measurement_target(self) -> None:
        payload = json.loads(TRIGGER_SET.read_text(encoding="utf-8"))
        self.assertEqual("designing-ai-infra", payload["skill_name"])

    def test_every_query_has_required_keys_and_types(self) -> None:
        for query in load_queries():
            with self.subTest(query=query.get("id")):
                self.assertEqual(REQUIRED_KEYS, set(query))
                self.assertIsInstance(query["should_trigger"], bool)
                for key in ("id", "query", "reason"):
                    self.assertIsInstance(query[key], str)
                    self.assertTrue(query[key].strip())
                self.assertNotIn("\n", query["reason"])

    def test_ids_are_unique(self) -> None:
        ids = [query["id"] for query in load_queries()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_set_has_18_true_and_19_false(self) -> None:
        # 18 целевых и 19 посторонних: x07 добавлен в пару к целевому p19 набора
        # developing-ai-agents (стоимость задачи агента растёт от контекста)
        queries = load_queries()
        positive = [q for q in queries if q["should_trigger"]]
        negative = [q for q in queries if not q["should_trigger"]]
        self.assertEqual(18, len(positive))
        self.assertEqual(19, len(negative))

    def test_id_prefix_matches_expected_outcome(self) -> None:
        expected = {"p": True, "w": True, "x": False, "n": False}
        for query in load_queries():
            with self.subTest(query=query["id"]):
                self.assertIn(query["id"][0], expected)
                self.assertEqual(expected[query["id"][0]], query["should_trigger"])

    def test_seven_cross_skill_cases_route_to_agent_skill(self) -> None:
        cross = [q for q in load_queries() if q["id"].startswith("x")]
        self.assertEqual(7, len(cross))
        self.assertIn("x07", {q["id"] for q in cross})
        for query in cross:
            with self.subTest(query=query["id"]):
                self.assertTrue(query["reason"].startswith("developing-ai-agents:"))

    def test_three_agent_workload_cases_trigger(self) -> None:
        workload = [q for q in load_queries() if q["id"].startswith("w")]
        self.assertEqual(3, len(workload))

    def test_other_negatives_name_their_category(self) -> None:
        for query in load_queries():
            if not query["id"].startswith("n"):
                continue
            with self.subTest(query=query["id"]):
                self.assertTrue(query["reason"].startswith(FALSE_CATEGORIES))

    def test_positives_cover_every_playbook(self) -> None:
        playbooks = {path.stem for path in PLAYBOOKS.glob("*.md")}
        covered = set()
        for query in load_queries():
            if not query["should_trigger"]:
                continue
            with self.subTest(query=query["id"]):
                name = query["reason"].split(":", 1)[0]
                self.assertIn(name, playbooks)
                covered.add(name)
        self.assertEqual(playbooks, covered)

    def test_queries_do_not_name_the_skill_or_reuse_benchmark_prompts(self) -> None:
        benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        prompts = {scenario["prompt"].strip() for scenario in benchmark["evals"]}
        for query in load_queries():
            with self.subTest(query=query["id"]):
                text = query["query"]
                self.assertNotIn("designing-ai-infra", text)
                self.assertNotIn("developing-ai-agents", text)
                self.assertNotIn(text.strip(), prompts)
                for prompt in prompts:
                    self.assertLess(overlap(text, prompt), MAX_BENCHMARK_OVERLAP)
                    self.assertLess(
                        containment(text, prompt), MAX_BENCHMARK_CONTAINMENT
                    )

    def test_containment_catches_a_query_cut_from_a_prompt(self) -> None:
        # первые слова длинного промпта: Jaccard мал, но весь запрос взят из промпта
        benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        prompt = max((scenario["prompt"] for scenario in benchmark["evals"]), key=len)
        cut = " ".join(re.findall(r"\w+", prompt)[:12])
        self.assertLess(overlap(cut, prompt), MAX_BENCHMARK_OVERLAP)
        self.assertGreaterEqual(containment(cut, prompt), MAX_BENCHMARK_CONTAINMENT)


if __name__ == "__main__":
    unittest.main()
