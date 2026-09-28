from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = ROOT / ".tmp" / "tests"
PLUGIN_DIRECTORY = Path("plugins") / "designing-ai-infra"
SKILL_DIRECTORY = PLUGIN_DIRECTORY / "skills" / "designing-ai-infra"


def run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "validate.py"), str(root)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def repository_copy() -> Iterator[Path]:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
        copied_root = Path(temporary_directory) / "repo"
        shutil.copytree(
            ROOT,
            copied_root,
            ignore=shutil.ignore_patterns(".git", ".tmp", "__pycache__"),
        )
        yield copied_root


class ValidateRepositoryTests(unittest.TestCase):
    def test_repository_validates_offline(self) -> None:
        result = run_validator(ROOT)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_rejects_missing_plugin_manifest(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"
            )
            manifest_path.unlink(missing_ok=True)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn(".codex-plugin/plugin.json", result.stdout)

    def test_rejects_missing_marketplace_manifest(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".agents" / "plugins" / "marketplace.json"
            marketplace_path.unlink()

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn(".agents/plugins/marketplace.json", result.stdout)

    def test_rejects_missing_claude_marketplace_manifest(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".claude-plugin" / "marketplace.json"
            marketplace_path.unlink(missing_ok=True)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn(".claude-plugin/marketplace.json", result.stdout)

    def test_rejects_missing_claude_plugin_manifest(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json"
            )
            manifest_path.unlink(missing_ok=True)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn(".claude-plugin/plugin.json", result.stdout)

    def test_rejects_marketplace_source_that_does_not_resolve(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".agents" / "plugins" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            payload["plugins"][0]["source"]["path"] = "./plugins/missing"
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid marketplace plugin source", result.stdout)

    def test_rejects_mismatched_marketplace_name(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".agents" / "plugins" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            payload["name"] = "wrong-marketplace"
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid marketplace name", result.stdout)

    def test_rejects_invalid_marketplace_contract(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".agents" / "plugins" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            del payload["plugins"][0]["policy"]["installation"]
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid marketplace contract", result.stdout)

    def test_rejects_invalid_claude_marketplace_source(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".claude-plugin" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            payload["plugins"][0]["source"] = "./plugins/missing"
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid Claude marketplace source", result.stdout)

    def test_rejects_invalid_claude_marketplace_owner(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".claude-plugin" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            payload["owner"] = {}
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid Claude marketplace contract", result.stdout)

    def test_rejects_missing_bundled_skill(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.unlink(missing_ok=True)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn("skills/designing-ai-infra/SKILL.md", result.stdout)

    def test_rejects_missing_bundled_eval_suite(self) -> None:
        with repository_copy() as copied_root:
            eval_path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            eval_path.unlink(missing_ok=True)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing required file", result.stdout)
        self.assertIn("evals/benchmark-v1.json", result.stdout)

    def test_rejects_plugin_without_legal_source_files(self) -> None:
        for relative_path in ("LICENSE", "NOTICE", "SOURCE.json"):
            with (
                self.subTest(relative_path=relative_path),
                repository_copy() as copied_root,
            ):
                target_path = copied_root / PLUGIN_DIRECTORY / relative_path
                target_path.unlink(missing_ok=True)

                result = run_validator(copied_root)

                self.assertNotEqual(0, result.returncode)
                self.assertIn("missing required file", result.stdout)
                self.assertIn(
                    str(PLUGIN_DIRECTORY / relative_path),
                    result.stdout,
                )

    def test_rejects_mismatched_plugin_manifest(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["name"] = "wrong-plugin"
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid plugin manifest", result.stdout)

    def test_rejects_invalid_plugin_manifest_contract(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["version"] = "v0.2"
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid plugin manifest", result.stdout)

    def test_rejects_unsupported_plugin_manifest_field(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["hooks"] = "./hooks.json"
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid plugin manifest", result.stdout)

    def test_rejects_invalid_claude_plugin_manifest_contract(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["version"] = "v0.2"
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid Claude plugin manifest", result.stdout)

    def test_rejects_unsynchronized_plugin_versions(self) -> None:
        with repository_copy() as copied_root:
            manifest_path = (
                copied_root / PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json"
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["version"] = "0.2.1"
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("plugin versions differ", result.stdout)

    def test_rejects_unsynchronized_claude_marketplace_version(self) -> None:
        with repository_copy() as copied_root:
            marketplace_path = copied_root / ".claude-plugin" / "marketplace.json"
            payload = json.loads(marketplace_path.read_text(encoding="utf-8"))
            payload["plugins"][0]["version"] = "0.2.1"
            marketplace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("plugin versions differ", result.stdout)

    def test_rejects_non_local_book_anchor(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\nLegacy source: `book/chapter1.md:1`.\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("non-local source anchor", result.stdout)

    def test_rejects_non_local_book_anchor_in_eval_json(self) -> None:
        with repository_copy() as copied_root:
            eval_path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            payload["legacy_anchor"] = "book/*.md:line"
            eval_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("non-local source anchor", result.stdout)

    def test_rejects_out_of_range_local_anchor(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\nBroken source: `references/source-book/chapter1.md:999999`.\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("source anchor out of range", result.stdout)

    def test_rejects_broken_markdown_link(self) -> None:
        with repository_copy() as copied_root:
            readme_path = copied_root / "README.md"
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8")
                + "\n[Broken local link](missing-file.md)\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("broken Markdown link", result.stdout)

    def test_ignores_markdown_under_tmp(self) -> None:
        with repository_copy() as copied_root:
            temporary_markdown = copied_root / ".tmp" / "race.md"
            temporary_markdown.parent.mkdir(parents=True)
            temporary_markdown.write_text(
                "[Transient broken link](missing-file.md)\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_rejects_missing_author_attribution(self) -> None:
        with repository_copy() as copied_root:
            readme_path = copied_root / "README.md"
            readme_path.write_text(
                readme_path.read_text(encoding="utf-8").replace(
                    "https://github.com/bojieli",
                    "https://example.invalid/author",
                ),
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing attribution", result.stdout)

    def test_rejects_invalid_json(self) -> None:
        with repository_copy() as copied_root:
            (copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json").write_text(
                "{\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid JSON", result.stdout)

    def test_rejects_invalid_jsonl(self) -> None:
        # Этот репозиторий пока не бандлит fixtures с трейсами (появятся вместе
        # с evals в фазе бенчмарков), поэтому файл создаётся фикстурой теста,
        # а не портится в уже существующем — проверяется тот же общий разбор
        # JSONL в validate_repository.
        with repository_copy() as copied_root:
            trace_path = copied_root / PLUGIN_DIRECTORY / "evals" / "sample-trace.jsonl"
            trace_path.write_text("{\n", encoding="utf-8")

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid JSONL", result.stdout)

    def test_rejects_invalid_skill_frontmatter(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8").replace(
                    "name: designing-ai-infra",
                    "name: wrong-skill-name",
                    1,
                ),
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid SKILL.md frontmatter", result.stdout)


class SourceLockTests(unittest.TestCase):
    @staticmethod
    def lock_path(root: Path) -> Path:
        return root / SKILL_DIRECTORY / "references" / "source-map.lock.json"

    def write_lock(self, root: Path, lock: dict) -> None:
        self.lock_path(root).write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def add_anchor_and_rebuild_lock(root: Path) -> None:
        """Ни один конспект пока не цитирует книгу (references/chapters ещё не
        создан), поэтому в закоммиченном локе нет анкоров; фикстура добавляет
        свою собственную ссылку на существующий заголовок и пересобирает лок,
        чтобы протестировать сам механизм проверки, а не будущий контент."""
        skill_path = root / SKILL_DIRECTORY / "SKILL.md"
        skill_path.write_text(
            skill_path.read_text(encoding="utf-8")
            + "\n\nПроверка: `references/source-book/chapter1.md:11`.\n",
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, str(root / "scripts" / "build_source_lock.py")],
            cwd=root,
            check=True,
            capture_output=True,
        )

    def test_rejects_anchor_missing_from_lock(self) -> None:
        with repository_copy() as copied_root:
            self.add_anchor_and_rebuild_lock(copied_root)
            lock = json.loads(self.lock_path(copied_root).read_text(encoding="utf-8"))
            lock["anchors"].pop("chapter1.md:11")
            self.write_lock(copied_root, lock)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("anchor missing from lock", result.stdout)
        self.assertIn("chapter1.md:11", result.stdout)

    def test_rejects_anchor_drift(self) -> None:
        with repository_copy() as copied_root:
            self.add_anchor_and_rebuild_lock(copied_root)
            lock = json.loads(self.lock_path(copied_root).read_text(encoding="utf-8"))
            lock["anchors"]["chapter1.md:11"]["line_sha256"] = "0" * 64
            self.write_lock(copied_root, lock)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("anchor drift", result.stdout)

    def test_rejects_missing_lock_file(self) -> None:
        with repository_copy() as copied_root:
            self.lock_path(copied_root).unlink()

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("source-map.lock.json", result.stdout)

    def test_rejects_inline_anchor_without_allowlist(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\n\nПроверка: `references/source-book/chapter1.md:15`.\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, str(copied_root / "scripts" / "build_source_lock.py")],
                cwd=copied_root,
                check=True,
                capture_output=True,
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("anchor is not a heading", result.stdout)
        self.assertIn("chapter1.md:15", result.stdout)

    def test_accepts_inline_anchor_listed_in_allowlist(self) -> None:
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\n\nПроверка: `references/source-book/chapter1.md:15`.\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, str(copied_root / "scripts" / "build_source_lock.py")],
                cwd=copied_root,
                check=True,
                capture_output=True,
            )
            lock = json.loads(self.lock_path(copied_root).read_text(encoding="utf-8"))
            lock["allowed_inline"] = [
                {"anchor": "chapter1.md:15", "reason": "формула вводится в абзаце"}
            ]
            self.write_lock(copied_root, lock)

            result = run_validator(copied_root)

        self.assertNotIn("anchor is not a heading", result.stdout)

    def test_rejects_anchor_range_beyond_file(self) -> None:
        # references/patterns.md ещё не существует в этом репозитории; носителем
        # анкора служит SKILL.md — он точно есть и не участвует в других
        # проверках этого теста.
        with repository_copy() as copied_root:
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\n\nИсточник: `references/source-book/chapter1.md:11-99999`.\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("source anchor out of range", result.stdout)


class ChapterQuoteTests(unittest.TestCase):
    @staticmethod
    def chapter_path(root: Path) -> Path:
        """Конспект-фикстура: references/chapters ещё не существует в этом
        репозитории (главы 12-15 фазы B создадут реальные конспекты), поэтому
        тест создаёт свой файл и регистрирует его в SKILL.md, чтобы не
        нарушить проверку маршрутизации (validate_skill_routing)."""
        path = (
            root / SKILL_DIRECTORY / "references" / "chapters" / "ch01-test-summary.md"
        )
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Глава 1 (тестовый конспект)\n\n"
                "Вспомогательный файл для теста источников; не публикуется.\n",
                encoding="utf-8",
            )
            skill_path = root / SKILL_DIRECTORY / "SKILL.md"
            relative = path.relative_to(root / SKILL_DIRECTORY).as_posix()
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + f"\n- [{path.name}]({relative})\n",
                encoding="utf-8",
            )
        return path

    @staticmethod
    def book_lines(root: Path) -> list[str]:
        book_path = (
            root / SKILL_DIRECTORY / "references" / "source-book" / "chapter1.md"
        )
        return book_path.read_text(encoding="utf-8").splitlines()

    def test_rejects_quote_absent_from_anchor_section(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n> «этой фразы нет ни в одной секции книги» — "
                "`references/source-book/chapter1.md:11`\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("quote not found in anchor section", result.stdout)

    def test_accepts_quote_present_in_anchor_section(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            quote = self.book_lines(copied_root)[12].strip()[:60]
            path.write_text(
                path.read_text(encoding="utf-8")
                + f"\n> «{quote}» — `references/source-book/chapter1.md:11`\n",
                encoding="utf-8",
            )
            # Новый анкор ещё не в закоммиченном локе (он пуст на этой стадии
            # проекта), поэтому лок пересобирается, иначе validate_source_lock
            # сообщит "anchor missing from lock" и до нужной проверки не дойдёт.
            subprocess.run(
                [sys.executable, str(copied_root / "scripts" / "build_source_lock.py")],
                cwd=copied_root,
                check=True,
                capture_output=True,
            )

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_checks_quote_containing_nested_guillemets(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n> «эта цитата содержит «вложенные» кавычки и в книге "
                "отсутствует» — `references/source-book/chapter1.md:11`\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("quote not found in anchor section", result.stdout)

    def test_rejects_chapter_summary_without_quotes(self) -> None:
        with repository_copy() as copied_root:
            path = (
                copied_root
                / SKILL_DIRECTORY
                / "references"
                / "chapters"
                / "ch02-empty-summary.md"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Пустой конспект\n\nБез цитат.\n", encoding="utf-8")

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("chapter summary without verified quotes", result.stdout)

    def test_rejects_quote_line_that_does_not_parse(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n> «цитата без ссылки на источник»\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("unparsed chapter quote", result.stdout)

    def test_rejects_quote_taken_from_a_neighbouring_section(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            lines = self.book_lines(copied_root)
            foreign = next(
                line.strip()
                for line in lines[150:250]
                if len(line.strip()) > 40 and not line.startswith("#")
            )
            path.write_text(
                path.read_text(encoding="utf-8")
                + f"\n> «{foreign[:60]}» — `references/source-book/chapter1.md:11`\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("quote not found in anchor section", result.stdout)


class SkillRoutingTests(unittest.TestCase):
    @staticmethod
    def skill_path(root: Path) -> Path:
        return root / SKILL_DIRECTORY / "SKILL.md"

    def test_rejects_reference_file_missing_from_skill(self) -> None:
        with repository_copy() as copied_root:
            orphan_path = (
                copied_root / SKILL_DIRECTORY / "references" / "orphan-note.md"
            )
            orphan_path.write_text("# Осиротевший файл\n", encoding="utf-8")

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("reference file not listed in SKILL.md", result.stdout)
        self.assertIn("orphan-note.md", result.stdout)

    def test_rejects_skill_listing_missing_reference(self) -> None:
        with repository_copy() as copied_root:
            path = self.skill_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n- `references/playbooks/nonexistent.md`\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("SKILL.md lists missing reference file", result.stdout)

    def test_rejects_oversized_skill_document(self) -> None:
        with repository_copy() as copied_root:
            path = self.skill_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8") + "строка\n" * 400,
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("SKILL.md exceeds", result.stdout)


class BenchmarkCoverageTests(unittest.TestCase):
    def test_rejects_playbook_without_benchmark_coverage(self) -> None:
        with repository_copy() as copied_root:
            new_playbook = (
                copied_root
                / SKILL_DIRECTORY
                / "references"
                / "playbooks"
                / "uncovered.md"
            )
            new_playbook.parent.mkdir(parents=True, exist_ok=True)
            new_playbook.write_text("# Новый playbook\n", encoding="utf-8")
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\n- [uncovered](references/playbooks/uncovered.md)\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("insufficient benchmark coverage", result.stdout)

    def test_rejects_benchmark_pointing_at_missing_file(self) -> None:
        # benchmark-v1.json пока пустой (evals: []), поэтому сценарий
        # добавляется фикстурой, а не правится в существующем элементе списка.
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["evals"].append(
                {"name": "ghost-check", "covers": ["references/playbooks/ghost.md"]}
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("benchmark covers a missing file", result.stdout)


if __name__ == "__main__":
    unittest.main()
