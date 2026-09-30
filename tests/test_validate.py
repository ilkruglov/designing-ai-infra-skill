from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
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


def rebuild_lock(root: Path) -> None:
    """Пересобрать lock копии: новый якорь фикстуры иначе дал бы вторую ошибку,
    «anchor missing from lock», и тест проверял бы не одну причину отказа."""
    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_source_lock.py")],
        cwd=root,
        check=True,
        capture_output=True,
    )


def error_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith("ERROR:")]


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

    def test_excluded_paths_are_pinned(self) -> None:
        # is_excluded_path: рабочие каталоги SDD (docs/superpowers, .superpowers) и
        # служебные .git/.tmp на любой глубине не проверяются; соседний docs/other —
        # обычная часть репозитория, и её ошибки сообщаются
        ignored = (
            Path("docs") / "superpowers",
            Path(".superpowers"),
            Path(".tmp"),
            Path("plugins") / ".tmp",
            PLUGIN_DIRECTORY / ".git" / "objects",
        )
        reported = Path("docs") / "other"
        with repository_copy() as copied_root:
            for directory in (*ignored, reported):
                target = copied_root / directory
                target.mkdir(parents=True, exist_ok=True)
                (target / "broken.json").write_text("{\n", encoding="utf-8")
                (target / "broken.md").write_text(
                    "[Broken local link](missing-file.md)\n", encoding="utf-8"
                )

            result = run_validator(copied_root)

        errors = [
            line for line in result.stdout.splitlines() if line.startswith("ERROR:")
        ]
        self.assertEqual(2, len(errors), result.stdout)
        self.assertTrue(
            errors[0].startswith(f"ERROR: invalid JSON in {reported / 'broken.json'}:"),
            result.stdout,
        )
        self.assertEqual(
            f"ERROR: broken Markdown link in {reported / 'broken.md'}: missing-file.md",
            errors[1],
        )

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
            # к разрешениям репозитория добавляется своё, а не заменяет их
            lock["allowed_inline"].append(
                {"anchor": "chapter1.md:15", "reason": "формула вводится в абзаце"}
            )
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
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("source anchor out of range", errors[0])


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
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("quote not found in anchor section", errors[0])

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
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("quote not found in anchor section", errors[0])

    def test_rejects_absent_quote_outside_chapter_summaries(self) -> None:
        # SKILL.md и все references/**, кроме текста книги, проверяются так же, как
        # конспекты глав: цитата с якорем должна быть дословной в секции якоря
        for relative in (
            SKILL_DIRECTORY / "SKILL.md",
            SKILL_DIRECTORY / "references" / "cheatsheet.md",
            SKILL_DIRECTORY / "references" / "playbooks" / "size-inference.md",
        ):
            with self.subTest(document=relative.as_posix()):
                with repository_copy() as copied_root:
                    path = copied_root / relative
                    path.write_text(
                        path.read_text(encoding="utf-8")
                        + "\n> «этой фразы нет ни в одной секции книги» — "
                        "`references/source-book/chapter1.md:11`\n",
                        encoding="utf-8",
                    )
                    rebuild_lock(copied_root)

                    result = run_validator(copied_root)

                errors = error_lines(result)
                self.assertEqual(1, len(errors), result.stdout)
                self.assertIn("quote not found in anchor section", errors[0])
                self.assertIn(relative.as_posix(), errors[0])

    def test_rejects_unparsed_quote_outside_chapter_summaries(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / SKILL_DIRECTORY / "references" / "cheatsheet.md"
            path.write_text(
                path.read_text(encoding="utf-8") + "\n> «цитата без якоря на книгу»\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("unparsed", errors[0])
        self.assertIn("cheatsheet.md", errors[0])

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
            # конспект указан в SKILL.md, иначе сработала бы и проверка маршрутизации
            skill_path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + f"\n- [{path.name}](references/chapters/{path.name})\n",
                encoding="utf-8",
            )
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("chapter summary without verified quotes", errors[0])

    def test_rejects_quote_line_that_does_not_parse(self) -> None:
        with repository_copy() as copied_root:
            path = self.chapter_path(copied_root)
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\n> «цитата без ссылки на источник»\n",
                encoding="utf-8",
            )
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("unparsed chapter quote", errors[0])

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
            rebuild_lock(copied_root)

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("quote not found in anchor section", errors[0])


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
        # Сценарий добавляется отдельным элементом без id, а не правится в
        # существующем: заодно проверяется, что валидатор не падает на нём.
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

    def test_rejects_scenario_with_missing_input_file(self) -> None:
        # Сценарий с несуществующим вложением прогоняется «вслепую»: модель
        # не видит данных, и критерии проверяют уже не то, что задумано.
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["evals"][0]["files"] = ["evals/fixtures/ghost-brief.yaml"]
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("benchmark scenario file missing", result.stdout)
        self.assertIn("evals/fixtures/ghost-brief.yaml", result.stdout)

    def test_rejects_calculation_scenario_without_golden(self) -> None:
        # Расчётный сценарий (id 3xx) без эталона нельзя оценить по допуску.
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["evals"].append(
                {"id": 399, "name": "calc-without-golden", "files": [], "covers": []}
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("calculation scenario without golden: 399", result.stdout)

    def test_rejects_golden_without_calculation_scenario(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "calc-goldens.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["goldens"].append(
                {
                    "id": 398,
                    "quantity": "orphan",
                    "value": 1,
                    "unit": "s",
                    "relative_tolerance": 0.01,
                    "derivation": "вручную",
                }
            )
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("golden without calculation scenario: 398", result.stdout)

    def _run_with_first_scenario_files(self, files: object) -> str:
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["evals"][0]["files"] = files
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            result = run_validator(copied_root)
        self.assertNotEqual(0, result.returncode)
        self.assertNotIn("Traceback", result.stderr)
        return result.stdout

    def test_rejects_scenario_file_outside_plugin(self) -> None:
        # Вложение вне плагина не попадает в установленный плагин: у
        # прогона оно есть, у пользователя — нет. «../» и абсолютный путь
        # проходят проверку существования, поэтому нужна отдельная граница.
        for outside in ("../../README.md", "/etc/hostname"):
            with self.subTest(path=outside):
                stdout = self._run_with_first_scenario_files([outside])
                self.assertIn("benchmark scenario file outside plugin", stdout)

    def test_rejects_non_string_scenario_file(self) -> None:
        stdout = self._run_with_first_scenario_files([None])
        self.assertIn("invalid benchmark scenario file", stdout)

    def test_rejects_scenario_files_that_are_not_a_list(self) -> None:
        # Строка вместо списка иначе перебиралась бы по символам.
        stdout = self._run_with_first_scenario_files(
            "evals/fixtures/chat-rag-brief.yaml"
        )
        self.assertIn("benchmark scenario files must be a list", stdout)

    def test_rejects_duplicate_golden_id(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / PLUGIN_DIRECTORY / "evals" / "calc-goldens.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["goldens"].append(dict(payload["goldens"][0]))
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("duplicate golden id: 301", result.stdout)


CALCULATOR_DIRECTORY = SKILL_DIRECTORY / "scripts" / "infra_calc"
CALCULATOR_TESTS_DIRECTORY = SKILL_DIRECTORY / "scripts" / "tests"
# chapter1.md:263 — заголовок «### 1.3.2 …», строка 265 — абзац под ним
HEADING_ANCHOR = "references/source-book/chapter1.md:263"
PARAGRAPH_ANCHOR = "references/source-book/chapter1.md:265"
SAMPLE_CALL = "\n\ndef test_sample() -> None:\n    sample.public_fn()\n"
AUTHOR_RESULTS = Path(".tmp") / "upcalc" / "calculations" / "results"
PIN = "d0cc188b68f49584fd21e05518a5d0f0db79aaf5"
AUTHOR_RESULT_NAME = re.compile(r"calculations/results/([\w.-]+\.json)#sha256=")


class CalculatorCoverageTests(unittest.TestCase):
    @staticmethod
    def add_calculator(root: Path, test_text: str) -> None:
        (root / CALCULATOR_DIRECTORY / "sample.py").write_text(
            "def public_fn() -> int:\n    return 1\n\n\n"
            "def _helper() -> int:\n    return 2\n",
            encoding="utf-8",
        )
        (root / CALCULATOR_TESTS_DIRECTORY / "test_sample.py").write_text(
            "from infra_calc import sample\n\n" + test_text, encoding="utf-8"
        )

    @staticmethod
    def add_author_result(root: Path, content: bytes) -> tuple[str, bool]:
        """Клон оригинала в копии с sample.json; вернуть хеш и полноту клона.

        Если рядом есть настоящий клон, в копию переносятся и результаты, на
        которые ссылаются настоящие тесты калькуляторов: клон полный, и валидатор
        не сообщает ни о чём, кроме причины теста. Без настоящего клона (чистый
        checkout, CI) клон синтетический — только sample.json, и тесты проверяют
        лишь ошибки о test_sample.py.
        """
        results = root / AUTHOR_RESULTS
        results.mkdir(parents=True, exist_ok=True)
        real = ROOT / AUTHOR_RESULTS
        complete = real.is_dir()
        if complete:
            for test_file in (ROOT / CALCULATOR_TESTS_DIRECTORY).glob("test_*.py"):
                for name in AUTHOR_RESULT_NAME.findall(test_file.read_text("utf-8")):
                    shutil.copy2(real / name, results / name)
        (results / "sample.json").write_bytes(content)
        return hashlib.sha256(content).hexdigest(), complete

    def assert_accepted(
        self, result: subprocess.CompletedProcess[str], complete: bool
    ) -> None:
        """Якорь test_sample.py принят; с полным клоном — и весь репозиторий."""
        own = [line for line in error_lines(result) if "test_sample.py" in line]
        self.assertEqual([], own, result.stdout)
        self.assertNotIn("local author clone is not at pin", result.stdout)
        if complete:
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_rejects_public_function_without_test(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(copied_root, f'ANCHORS = ("{HEADING_ANCHOR}",)\n')

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "calculator function without anchored test: sample.public_fn",
            result.stdout,
        )
        self.assertNotIn("sample._helper", result.stdout)

    def test_rejects_function_tested_without_anchors(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(copied_root, "ANCHORS = ()\n" + SAMPLE_CALL)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "calculator function without anchored test: sample.public_fn",
            result.stdout,
        )

    def test_rejects_call_that_is_only_a_comment(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root,
                f'ANCHORS = ("{HEADING_ANCHOR}",)\n# sample.public_fn(\n',
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "calculator function without anchored test: sample.public_fn",
            result.stdout,
        )

    def test_rejects_call_that_is_only_a_string(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root,
                f'ANCHORS = ("{HEADING_ANCHOR}",)\n\n\n'
                'def test_sample() -> None:\n    text = "sample.public_fn(1)"\n',
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "calculator function without anchored test: sample.public_fn",
            result.stdout,
        )

    def test_rejects_call_outside_test_functions(self) -> None:
        # вызов на уровне модуля не проверяется ни одним assert теста
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root,
                f'ANCHORS = ("{HEADING_ANCHOR}",)\nVALUE = sample.public_fn()\n',
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "calculator function without anchored test: sample.public_fn",
            result.stdout,
        )

    def test_accepts_call_inside_test_method(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root,
                f'import unittest\n\nANCHORS = ("{HEADING_ANCHOR}",)\n\n\n'
                "class SampleTest(unittest.TestCase):\n"
                "    def test_value(self) -> None:\n"
                "        self.assertEqual(sample.public_fn(), 1)\n",
            )

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_rejects_anchor_that_is_not_a_heading(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root, f'ANCHORS = ("{PARAGRAPH_ANCHOR}",)\n' + SAMPLE_CALL
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"calculator test anchor is not a heading: test_sample.py: {PARAGRAPH_ANCHOR}",
            result.stdout,
        )

    def test_rejects_anchor_of_unknown_form(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root, 'ANCHORS = ("book/chapter1.md:263",)\n' + SAMPLE_CALL
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("calculator test anchor has unknown form", result.stdout)

    def test_accepts_function_with_anchored_test(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root, f'ANCHORS = ("{HEADING_ANCHOR}",)\n' + SAMPLE_CALL
            )

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_reads_annotated_anchors(self) -> None:
        with repository_copy() as copied_root:
            self.add_calculator(
                copied_root,
                f'ANCHORS: tuple[str, ...] = ("{HEADING_ANCHOR}",)\n' + SAMPLE_CALL,
            )

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_rejects_author_result_with_wrong_hash(self) -> None:
        with repository_copy() as copied_root:
            self.add_author_result(copied_root, b'{"value": 1}\n')
            wrong = hashlib.sha256(b'{"value": 2}\n').hexdigest()
            anchor = f"calculations/results/sample.json#sha256={wrong}"
            self.add_calculator(copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"calculator test anchor hash mismatch: test_sample.py: {anchor}",
            result.stdout,
        )

    def test_accepts_author_result_with_matching_hash(self) -> None:
        with repository_copy() as copied_root:
            digest, complete = self.add_author_result(copied_root, b'{"value": 1}\n')
            anchor = f"calculations/results/sample.json#sha256={digest}"
            self.add_calculator(copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL)

            result = run_validator(copied_root)

        self.assert_accepted(result, complete)

    def test_rejects_author_result_missing_from_local_clone(self) -> None:
        # клон на месте, а файла в нём нет: якорь указывает на несуществующий результат
        with repository_copy() as copied_root:
            self.add_author_result(copied_root, b'{"value": 1}\n')
            anchor = f"calculations/results/absent.json#sha256={'0' * 64}"
            self.add_calculator(copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"calculator test anchor missing from local author clone: "
            f"test_sample.py: {anchor}",
            result.stdout,
        )

    def test_rejects_local_clone_off_the_pin(self) -> None:
        # хеши сверяются с клоном на пине d0cc188b; клон на другом коммите — ошибка
        with repository_copy() as copied_root:
            digest, complete = self.add_author_result(copied_root, b'{"value": 1}\n')
            anchor = f"calculations/results/sample.json#sha256={digest}"
            self.add_calculator(copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL)
            head = copied_root / ".tmp" / "upcalc" / ".git" / "HEAD"
            head.parent.mkdir(parents=True)
            head.write_text("0" * 40 + "\n", encoding="utf-8")
            off_pin = run_validator(copied_root)
            head.write_text(PIN + "\n", encoding="utf-8")
            on_pin = run_validator(copied_root)

        self.assertNotEqual(0, off_pin.returncode)
        self.assertIn("local author clone is not at pin d0cc188b", off_pin.stdout)
        self.assert_accepted(on_pin, complete)

    def test_resolves_branch_head_of_local_clone(self) -> None:
        # клон на ветке: HEAD — «ref: refs/heads/…», sha лежит в .git/refs или в
        # packed-refs; на пине ошибки нет, на другом коммите — есть
        cases = (
            ("loose ref on pin", PIN, False, True),
            ("packed ref on pin", PIN, True, True),
            ("loose ref off pin", "1" * 40, False, False),
            ("packed ref off pin", "1" * 40, True, False),
        )
        for label, sha, packed, on_pin in cases:
            with self.subTest(label), repository_copy() as copied_root:
                digest, complete = self.add_author_result(
                    copied_root, b'{"value": 1}\n'
                )
                anchor = f"calculations/results/sample.json#sha256={digest}"
                self.add_calculator(
                    copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL
                )
                git = copied_root / ".tmp" / "upcalc" / ".git"
                git.mkdir(parents=True)
                (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
                if packed:
                    (git / "packed-refs").write_text(
                        "# pack-refs with: peeled fully-peeled sorted\n"
                        f"{'2' * 40} refs/heads/other\n"
                        f"{sha} refs/heads/main\n",
                        encoding="utf-8",
                    )
                else:
                    (git / "refs" / "heads").mkdir(parents=True)
                    (git / "refs" / "heads" / "main").write_text(
                        sha + "\n", encoding="utf-8"
                    )

                result = run_validator(copied_root)

                if on_pin:
                    self.assert_accepted(result, complete)
                else:
                    self.assertIn(
                        "local author clone is not at pin d0cc188b", result.stdout
                    )
                    self.assertIn(sha, result.stdout)

    def test_skips_hash_check_without_local_clone(self) -> None:
        # копия репозитория не содержит .tmp/upcalc: форма проверяется, хеш — нет
        with repository_copy() as copied_root:
            anchor = f"calculations/results/absent.json#sha256={'0' * 64}"
            self.add_calculator(copied_root, f'ANCHORS = ("{anchor}",)\n' + SAMPLE_CALL)

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


class DataIntegrityTests(unittest.TestCase):
    @staticmethod
    def edit_source(root: Path, relative: Path, change: Callable[[dict], None]) -> None:
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def test_rejects_hardware_snapshot_changed_after_recording(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / SKILL_DIRECTORY / "data" / "hardware.json"
            path.write_bytes(path.read_bytes() + b"\n")

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("hardware.json sha256 mismatch with SOURCE.json", result.stdout)
        self.assertIn(
            f"hardware.json sha256 mismatch with {PLUGIN_DIRECTORY / 'SOURCE.json'}",
            result.stdout,
        )

    def test_rejects_source_record_without_hash(self) -> None:
        def drop_hash(payload: dict) -> None:
            del payload["data"]["hardware_json"]["sha256"]

        with repository_copy() as copied_root:
            self.edit_source(copied_root, Path("SOURCE.json"), drop_hash)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "SOURCE.json lacks data.hardware_json.sha256: SOURCE.json", result.stdout
        )

    def test_rejects_root_source_path_that_does_not_resolve(self) -> None:
        def move(payload: dict) -> None:
            payload["data"]["hardware_json"]["path"] = "data/hardware.json"

        with repository_copy() as copied_root:
            self.edit_source(copied_root, Path("SOURCE.json"), move)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "SOURCE.json path does not resolve: SOURCE.json "
            "data.hardware_json.path=data/hardware.json",
            result.stdout,
        )

    def test_rejects_plugin_source_path_relative_to_repository_root(self) -> None:
        # путь в SOURCE.json плагина отсчитывается от каталога плагина,
        # поэтому путь от корня репозитория в нём не разрешается
        rooted = (SKILL_DIRECTORY / "references" / "source-book").as_posix()

        def reroot(payload: dict) -> None:
            payload["bundled_sources"]["directory"] = rooted

        with repository_copy() as copied_root:
            self.edit_source(copied_root, PLUGIN_DIRECTORY / "SOURCE.json", reroot)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"SOURCE.json path does not resolve: {PLUGIN_DIRECTORY / 'SOURCE.json'} "
            f"bundled_sources.directory={rooted}",
            result.stdout,
        )

    def test_rejects_book_text_changed_outside_anchors(self) -> None:
        # README, SKILL.md и NOTICE обещают побайтную копию перевода; строка
        # chapter1.md:176 — не якорь, и lock её не хеширует
        with repository_copy() as copied_root:
            path = copied_root / SKILL_DIRECTORY / "references" / "source-book"
            path = path / "chapter1.md"
            lines = path.read_text(encoding="utf-8").split("\n")
            self.assertIn("80 GB", lines[175])
            lines[175] = lines[175].replace("80 GB", "40 GB", 1)
            path.write_text("\n".join(lines), encoding="utf-8")

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(2, len(errors), result.stdout)
        for source in (Path("SOURCE.json"), PLUGIN_DIRECTORY / "SOURCE.json"):
            self.assertIn(
                f"bundled source sha256 mismatch with {source}: chapter1.md",
                result.stdout,
            )

    def test_rejects_source_record_without_book_hash(self) -> None:
        def drop_hash(payload: dict) -> None:
            del payload["bundled_sources"]["sha256"]["chapter1.md"]

        with repository_copy() as copied_root:
            self.edit_source(copied_root, Path("SOURCE.json"), drop_hash)

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "SOURCE.json lacks bundled_sources.sha256 for chapter1.md: SOURCE.json",
            result.stdout,
        )

    def test_rejects_cjk_artifact_in_skill_document(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / SKILL_DIRECTORY / "SKILL.md"
            text = path.read_text(encoding="utf-8")
            path.write_text(text + "Номер 对\n", encoding="utf-8")
            line = len(text.splitlines()) + 1

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"CJK artifact: {SKILL_DIRECTORY / 'SKILL.md'}:{line}", result.stdout
        )

    def test_rejects_cjk_artifact_in_references(self) -> None:
        with repository_copy() as copied_root:
            path = copied_root / SKILL_DIRECTORY / "references" / "chapters" / "x.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Глава\n\nНомер 对 из перевода\n", encoding="utf-8")

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            f"CJK artifact: {SKILL_DIRECTORY / 'references' / 'chapters' / 'x.md'}:3",
            result.stdout,
        )


class PinTests(unittest.TestCase):
    """Пины оригинала и перевода записаны в нескольких местах; источник истины —
    константы scripts/build_source_lock.py, остальные записи с ними сверяются."""

    UPSTREAM = "d0cc188b68f49584fd21e05518a5d0f0db79aaf5"
    TRANSLATION = "ec343c9d23a69dca5a4922b242c26b77b1025e6d"
    OTHER = "0123456789abcdef0123456789abcdef01234567"

    def mutate(self, relative: Path, old: str, new: str) -> list[str]:
        with repository_copy() as copied_root:
            path = copied_root / relative
            text = path.read_text(encoding="utf-8")
            self.assertIn(old, text)
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            result = run_validator(copied_root)
        return error_lines(result)

    def test_rejects_pin_mismatch(self) -> None:
        cases = (
            (PLUGIN_DIRECTORY / "SOURCE.json", self.TRANSLATION),
            (Path("SOURCE.json"), self.UPSTREAM),
            (SKILL_DIRECTORY / "references" / "source-map.lock.json", self.UPSTREAM),
            (SKILL_DIRECTORY / "SKILL.md", self.TRANSLATION),
            (SKILL_DIRECTORY / "references" / "source-map.md", self.UPSTREAM),
            (Path("README.md"), self.TRANSLATION),
        )
        for relative, pin in cases:
            with self.subTest(document=relative.as_posix()):
                errors = self.mutate(relative, pin, self.OTHER)
                self.assertEqual(1, len(errors), errors)
                self.assertIn("pin mismatch", errors[0])
                self.assertIn(relative.as_posix(), errors[0])

    def test_rejects_short_upstream_pin_mismatch(self) -> None:
        errors = self.mutate(
            SKILL_DIRECTORY / "SKILL.md",
            "ai-infra-book@d0cc188b",
            "ai-infra-book@deadbeef",
        )
        self.assertEqual(1, len(errors), errors)
        self.assertIn("pin mismatch", errors[0])
        self.assertIn("deadbeef", errors[0])

    def test_rejects_short_pin_mismatch_in_any_skill_document(self) -> None:
        # короткая ссылка на оригинал есть и в шаблонах, шпаргалке и главах, а не
        # только в документах, которые называют пины
        references = SKILL_DIRECTORY / "references"
        for relative in (
            references / "templates" / "sizing-sheet.md",
            references / "templates" / "training-plan.md",
            references / "cheatsheet.md",
            references / "chapters" / "ch04-accelerators.md",
        ):
            with self.subTest(document=relative.as_posix()):
                errors = self.mutate(
                    relative,
                    "bojieli/ai-infra-book@d0cc188b",
                    "bojieli/ai-infra-book@56ecb426",
                )
                self.assertEqual(1, len(errors), errors)
                self.assertIn("pin mismatch", errors[0])
                self.assertIn(relative.as_posix(), errors[0])
                self.assertIn("56ecb426", errors[0])

    def append_line(self, relative: Path, line: str) -> list[str]:
        with repository_copy() as copied_root:
            path = copied_root / relative
            text = path.read_text(encoding="utf-8")
            path.write_text(f"{text}\n{line}\n", encoding="utf-8")
            result = run_validator(copied_root)
        return error_lines(result)

    def test_rejects_translation_and_full_pin_mismatch_in_skill_documents(
        self,
    ) -> None:
        template = SKILL_DIRECTORY / "references" / "templates" / "sizing-sheet.md"
        cases = (
            ("Перевод: ilkruglov/ai-infra-book@cb502e1d.", "cb502e1d"),
            (f"Коммит оригинала `{self.OTHER}`.", self.OTHER),
        )
        for line, wrong in cases:
            with self.subTest(line=line):
                errors = self.append_line(template, line)
                self.assertEqual(1, len(errors), errors)
                self.assertIn("pin mismatch", errors[0])
                self.assertIn(template.as_posix(), errors[0])
                self.assertIn(wrong, errors[0])
        # верные короткие ссылки на оба пина ошибок не дают
        self.assertEqual(
            [],
            self.append_line(
                template,
                "ilkruglov/ai-infra-book@ec343c9d и bojieli/ai-infra-book@d0cc188b68f4",
            ),
        )

    def test_rejects_bare_short_pin_mismatch(self) -> None:
        # короткий sha без имени репозитория: «на `d0cc188b`», «на коммите `d0cc188b`»
        references = SKILL_DIRECTORY / "references"
        cases = (
            (
                references / "chapters" / "ch00-preface.md",
                "Снимок `hardware.json` взят из репозитория оригинала на `d0cc188b`",
            ),
            (
                references / "source-map.md",
                "снимок оригинала на коммите `d0cc188b`",
            ),
            (
                references / "playbooks" / "compare-model-architectures.md",
                "в репозитории оригинала на коммите `d0cc188b`",
            ),
        )
        for relative, old in cases:
            with self.subTest(document=relative.as_posix()):
                errors = self.mutate(relative, old, old.replace("d0cc188b", "56ecb426"))
                self.assertEqual(1, len(errors), errors)
                self.assertIn("pin mismatch", errors[0])
                self.assertIn(relative.as_posix(), errors[0])
                self.assertIn("56ecb426", errors[0])

    def test_bare_short_pin_check_ignores_numbers_and_hashes(self) -> None:
        # числа вида 16345e6, sha256 и blob id без слова-якоря ошибок не дают;
        # верные короткие sha обоих пинов после якоря — тоже
        template = SKILL_DIRECTORY / "references" / "templates" / "sizing-sheet.md"
        line = (
            "Чтение на 16345e6 байт, пин 1.0195e12 B/s, blob 0123abcd0123abcd, "
            "sha256 `0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef`; "
            "оригинал на коммите `d0cc188b68f`, перевод на `ec343c9d`, commit ec343c9."
        )
        self.assertEqual([], self.append_line(template, line))
        errors = self.append_line(template, "Перевод на коммите `cb502e1d`.")
        self.assertEqual(1, len(errors), errors)
        self.assertIn("cb502e1d", errors[0])

    def test_bare_short_pin_check_ignores_decimals_and_emails(self) -> None:
        # sha пина содержит буквы a–f: десятичное число после якоря — не sha;
        # @ внутри слова (почта, repo@ref чужого репозитория) — не якорь
        template = SKILL_DIRECTORY / "references" / "templates" / "sizing-sheet.md"
        line = (
            "Буфер на 1048576 байт, at 1048576 B, @1234567, "
            "почта email@deadbeef1 и other/repo@deadbeef2."
        )
        self.assertEqual([], self.append_line(template, line))
        # якорный @ в начале слова по-прежнему проверяется
        errors = self.append_line(template, "Снимок @56ecb426.")
        self.assertEqual(1, len(errors), errors)
        self.assertIn("56ecb426", errors[0])

    def test_pins_are_the_lock_builder_constants(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import build_source_lock
        finally:
            sys.path.pop(0)
        self.assertEqual(build_source_lock.UPSTREAM_COMMIT, self.UPSTREAM)
        self.assertEqual(build_source_lock.TRANSLATION_COMMIT, self.TRANSLATION)


class NumbersAnchorTests(unittest.TestCase):
    HEADER = (
        "# Числа книги\n\n"
        "| Величина | Значение | Модель или железо | Как использовать | Источник |\n"
        "|---|---|---|---|---|\n"
    )

    @staticmethod
    def numbers_path(root: Path) -> Path:
        return root / SKILL_DIRECTORY / "references" / "numbers.md"

    def test_rejects_numbers_row_without_anchor(self) -> None:
        with repository_copy() as copied_root:
            self.numbers_path(copied_root).write_text(
                self.HEADER + "| Вес | 1 GB | модель | для примера | глава 1 |\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("numbers.md row without anchor", result.stdout)
        self.assertIn("numbers.md:5", result.stdout)

    def test_treats_aligned_separator_as_table_structure(self) -> None:
        # выравнивание столбцов «---:» не делает разделитель строкой данных,
        # а строка перед ним остаётся заголовком
        with repository_copy() as copied_root:
            self.numbers_path(copied_root).write_text(
                "# Числа книги\n\n"
                "| Величина | Значение | Модель или железо | Как использовать | Источник |\n"
                "| :--- | ---: | :---: | --- | --- |\n"
                "| Вес | 1 GB | модель | для примера | "
                "`references/source-book/chapter1.md:3` |\n"
                "| Вес | 2 GB | модель | без якоря | глава 1 |\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotIn("numbers.md:3", result.stdout)
        self.assertNotIn("numbers.md:4", result.stdout)
        self.assertNotIn("numbers.md:5", result.stdout)
        self.assertIn("numbers.md row without anchor", result.stdout)
        self.assertIn("numbers.md:6", result.stdout)

    def test_accepts_numbers_rows_with_anchor(self) -> None:
        with repository_copy() as copied_root:
            self.numbers_path(copied_root).write_text(
                self.HEADER + "| Вес | 1 GB | модель | для примера | "
                "`references/source-book/chapter1.md:3` |\n",
                encoding="utf-8",
            )

            result = run_validator(copied_root)

        self.assertNotIn("numbers.md row without anchor", result.stdout)


class CodeCommentQuoteTests(unittest.TestCase):
    """Цитата книги в комментарии кода сверяется со строкой, на которую он
    ссылается: после сдвига текста книги такой номер иначе молча устаревает."""

    # Ссылка ищется в самом тесте: номер строки меняется при обновлении книги
    COMMENT = re.compile(
        r"chapter6\.md:(?P<line>\d+): «Четыре карты сначала выполняют попарную"
    )

    @staticmethod
    def collectives_test(root: Path) -> Path:
        return root / SKILL_DIRECTORY / "scripts" / "tests" / "test_collectives.py"

    def shifted_comment(self, text: str) -> tuple[str, str, str]:
        """Комментарий с верной ссылкой, он же со ссылкой на строку выше и
        неверная ссылка."""
        match = self.COMMENT.search(text)
        self.assertIsNotNone(match)
        assert match is not None
        wrong = f"chapter6.md:{int(match.group('line')) - 1}"
        shifted = match.group(0).replace(match.group(0).split(": ")[0], wrong)
        return match.group(0), shifted, wrong

    def test_rejects_comment_quote_absent_from_referenced_line(self) -> None:
        with repository_copy() as copied_root:
            path = self.collectives_test(copied_root)
            text = path.read_text(encoding="utf-8")
            original, shifted, wrong = self.shifted_comment(text)
            path.write_text(text.replace(original, shifted), encoding="utf-8")

            result = run_validator(copied_root)

        errors = error_lines(result)
        self.assertEqual(1, len(errors), result.stdout)
        self.assertIn("code comment quote not found at referenced line", errors[0])
        self.assertIn("test_collectives.py:", errors[0])
        self.assertIn(wrong, errors[0])

    def test_ignores_quotes_outside_comments(self) -> None:
        with repository_copy() as copied_root:
            path = self.collectives_test(copied_root)
            text = path.read_text(encoding="utf-8")
            _, shifted, _ = self.shifted_comment(text)
            path.write_text(text + f'\nNOTE = "{shifted}"\n', encoding="utf-8")

            result = run_validator(copied_root)

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    PHRASE = "Четыре карты сначала выполняют попарную редукцию"

    @staticmethod
    def phrase_line(root: Path) -> int:
        """Номер строки chapter6.md с фразой PHRASE в копии книги."""
        book = root / SKILL_DIRECTORY / "references" / "source-book" / "chapter6.md"
        for number, line in enumerate(
            book.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if CodeCommentQuoteTests.PHRASE in line:
                return number
        raise AssertionError("phrase missing from chapter6.md")

    @staticmethod
    def comment_errors(root: Path, comment: str) -> list[str]:
        """Ошибки проверки цитат для модуля, состоящего из комментария."""
        sample = root / SKILL_DIRECTORY / "scripts" / "comment_sample.py"
        sample.write_text(comment, encoding="utf-8")
        result = run_validator(root)
        return [line for line in error_lines(result) if "comment_sample.py" in line]

    def test_checks_quote_on_following_comment_line(self) -> None:
        with repository_copy() as copied_root:
            line = self.phrase_line(copied_root)
            wrong = self.comment_errors(
                copied_root,
                f"# chapter6.md:{line - 1}, пример:\n#     «{self.PHRASE}»\n",
            )
            right = self.comment_errors(
                copied_root,
                f"# chapter6.md:{line}, пример:\n#     «{self.PHRASE}»\n",
            )

        self.assertEqual(1, len(wrong), wrong)
        self.assertIn("code comment quote not found at referenced line", wrong[0])
        self.assertIn(f"chapter6.md:{line - 1}", wrong[0])
        self.assertEqual([], right)

    def test_quote_on_following_line_stops_at_next_anchor(self) -> None:
        # цитата после второй ссылки не приписывается первой
        with repository_copy() as copied_root:
            line = self.phrase_line(copied_root)
            errors = self.comment_errors(
                copied_root,
                f"# chapter6.md:{line - 1}: без цитаты;\n"
                f"# chapter6.md:{line}: «{self.PHRASE}»\n",
            )

        self.assertEqual([], errors)

    def test_checks_quote_against_line_range(self) -> None:
        with repository_copy() as copied_root:
            line = self.phrase_line(copied_root)
            inside = self.comment_errors(
                copied_root, f"# chapter6.md:{line - 2}-{line}: «{self.PHRASE}»\n"
            )
            outside = self.comment_errors(
                copied_root,
                f"# chapter6.md:{line + 1}-{line + 3}: «{self.PHRASE}»\n",
            )

        self.assertEqual([], inside)
        self.assertEqual(1, len(outside), outside)
        self.assertIn(f"chapter6.md:{line + 1}-{line + 3}", outside[0])

    def test_checks_only_text_before_ellipsis(self) -> None:
        with repository_copy() as copied_root:
            line = self.phrase_line(copied_root)
            errors = [
                self.comment_errors(
                    copied_root,
                    f"# chapter6.md:{line}: «Четыре карты сначала{mark} нет в книге»\n",
                )
                for mark in (" ...", " …")
            ]
            wrong = self.comment_errors(
                copied_root, f"# chapter6.md:{line}: «Четыре карты потом ... »\n"
            )

        self.assertEqual([[], []], errors)
        self.assertEqual(1, len(wrong), wrong)

    def test_rejects_anchor_beyond_end_of_book_file(self) -> None:
        with repository_copy() as copied_root:
            errors = self.comment_errors(
                copied_root, f"# chapter6.md:999999: «{self.PHRASE}»\n"
            )

        self.assertEqual(1, len(errors), errors)
        self.assertIn("code comment anchor out of range", errors[0])
        self.assertIn("chapter6.md:999999", errors[0])


if __name__ == "__main__":
    unittest.main()
