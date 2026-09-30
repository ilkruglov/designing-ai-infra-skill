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
            "d0cc188b68f49584fd21e05518a5d0f0db79aaf5",
            lock["book"]["upstream_commit"],
        )
        self.assertEqual(
            "ec343c9d23a69dca5a4922b242c26b77b1025e6d",
            lock["book"]["translation_commit"],
        )

    def test_committed_lock_matches_generated_lock(self) -> None:
        generated = build_source_lock.build_lock(ROOT)
        committed = json.loads(
            (ROOT / build_source_lock.LOCK_RELATIVE_PATH).read_text(encoding="utf-8")
        )

        self.assertEqual(generated, committed)

    @staticmethod
    def copy_with_inline_anchor(temporary_directory: str) -> Path:
        """Копия репозитория, где SKILL.md ссылается на абзац chapter1.md:15,
        а lock разрешает этот якорь в allowed_inline."""
        copied_root = Path(temporary_directory) / "repo"
        shutil.copytree(
            ROOT,
            copied_root,
            ignore=shutil.ignore_patterns(".git", ".tmp", "__pycache__"),
        )
        skill_path = copied_root / build_source_lock.SKILL_DIRECTORY / "SKILL.md"
        skill_path.write_text(
            skill_path.read_text(encoding="utf-8")
            + "\n\nПроверка: `references/source-book/chapter1.md:15`.\n",
            encoding="utf-8",
        )
        lock_path = copied_root / build_source_lock.LOCK_RELATIVE_PATH
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["allowed_inline"] = [
            {"anchor": "chapter1.md:15", "reason": "формула вводится в абзаце"},
            {"anchor": "chapter1.md:99999", "reason": "ссылки больше нет"},
        ]
        lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
        return copied_root

    def test_rebuild_keeps_allowlist_of_referenced_inline_anchors(self) -> None:
        # Разрешение на якорь-абзац — решение ревьюера с причиной; пересборка
        # lock после сдвига книги не должна его стирать, а разрешение для
        # якоря, на который больше никто не ссылается, не должно оставаться
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
            copied_root = self.copy_with_inline_anchor(temporary_directory)

            lock = build_source_lock.build_lock(copied_root)

        self.assertEqual(
            [{"anchor": "chapter1.md:15", "reason": "формула вводится в абзаце"}],
            lock["allowed_inline"],
        )
        self.assertEqual("inline", lock["anchors"]["chapter1.md:15"]["kind"])


if __name__ == "__main__":
    unittest.main()
