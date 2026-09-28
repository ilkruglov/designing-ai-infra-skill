from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_source_lock

TEMP_ROOT = ROOT / ".tmp" / "tests"


class BuildLockTests(unittest.TestCase):
    def test_lock_contains_known_anchor_with_heading_text(self) -> None:
        # На этой стадии проекта ни один конспект ещё не цитирует книгу
        # (references/chapters ещё не существует), поэтому у build_lock нет
        # анкора, взятого из реального содержимого репозитория; фикстура
        # добавляет собственную ссылку на существующий заголовок книги, чтобы
        # проверить именно механику сборки лока, а не будущий контент.
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
            copied_root = Path(temporary_directory) / "repo"
            shutil.copytree(
                ROOT,
                copied_root,
                ignore=shutil.ignore_patterns(".git", ".tmp", "__pycache__"),
            )
            skill_path = copied_root / build_source_lock.SKILL_DIRECTORY / "SKILL.md"
            skill_path.write_text(
                skill_path.read_text(encoding="utf-8")
                + "\n\nПроверка: `references/source-book/chapter1.md:11`.\n",
                encoding="utf-8",
            )

            lock = build_source_lock.build_lock(copied_root)

        entry = lock["anchors"]["chapter1.md:11"]

        self.assertEqual("heading", entry["kind"])
        self.assertTrue(entry["line_text"].startswith("##"))
        self.assertEqual(64, len(entry["line_sha256"]))

    def test_lock_pins_book_commits(self) -> None:
        lock = build_source_lock.build_lock(ROOT)

        self.assertEqual(
            "56ecb425b07ea6d16e891cba87bf7db416927d09",
            lock["book"]["upstream_commit"],
        )

    def test_committed_lock_matches_generated_lock(self) -> None:
        generated = build_source_lock.build_lock(ROOT)
        committed = json.loads(
            (ROOT / build_source_lock.LOCK_RELATIVE_PATH).read_text(encoding="utf-8")
        )

        self.assertEqual(generated, committed)


if __name__ == "__main__":
    unittest.main()
