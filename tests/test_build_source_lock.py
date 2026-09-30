from __future__ import annotations

import hashlib
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
    def line_sha256(root: Path, name: str, number: int) -> str:
        book = root / build_source_lock.SKILL_DIRECTORY / "references" / "source-book"
        line = (book / name).read_text(encoding="utf-8").splitlines()[number - 1]
        return hashlib.sha256(line.encode("utf-8")).hexdigest()

    @staticmethod
    def copy_with_inline_anchor(temporary_directory: str, digest: str | None) -> Path:
        """Копия репозитория, где SKILL.md ссылается на абзац chapter1.md:15,
        а lock разрешает этот якорь в allowed_inline с хешем digest (None —
        хеш настоящей строки) и держит разрешение для исчезнувшей ссылки."""
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
        if digest is None:
            digest = BuildLockTests.line_sha256(copied_root, "chapter1.md", 15)
        lock_path = copied_root / build_source_lock.LOCK_RELATIVE_PATH
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["allowed_inline"] = [
            {
                "anchor": "chapter1.md:15",
                "line_sha256": digest,
                "reason": "формула вводится в абзаце",
            },
            {
                "anchor": "chapter1.md:99999",
                "line_sha256": digest,
                "reason": "ссылки больше нет",
            },
        ]
        lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
        return copied_root

    def test_rebuild_keeps_allowlist_of_referenced_inline_anchors(self) -> None:
        # Разрешение на якорь-абзац — решение ревьюера с причиной; пересборка
        # lock не должна его стирать, пока строка под номером та же, а
        # разрешение для якоря, на который никто не ссылается, не остаётся
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
            copied_root = self.copy_with_inline_anchor(temporary_directory, None)
            digest = self.line_sha256(copied_root, "chapter1.md", 15)

            lock = build_source_lock.build_lock(copied_root)

        self.assertEqual(
            [
                {
                    "anchor": "chapter1.md:15",
                    "line_sha256": digest,
                    "reason": "формула вводится в абзаце",
                }
            ],
            lock["allowed_inline"],
        )
        self.assertEqual("inline", lock["anchors"]["chapter1.md:15"]["kind"])

    def test_rebuild_drops_allowlist_entry_when_line_changed(self) -> None:
        # под тем же номером теперь другая строка: разрешение ревьюер давал
        # не ей, и перенос молча разрешил бы чужой абзац
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
            copied_root = self.copy_with_inline_anchor(temporary_directory, "0" * 64)

            lock = build_source_lock.build_lock(copied_root)

        self.assertEqual([], lock["allowed_inline"])

    def test_rebuild_drops_allowlist_entry_without_hash(self) -> None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary_directory:
            copied_root = self.copy_with_inline_anchor(temporary_directory, None)
            lock_path = copied_root / build_source_lock.LOCK_RELATIVE_PATH
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            for item in lock["allowed_inline"]:
                del item["line_sha256"]
            lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")

            lock = build_source_lock.build_lock(copied_root)

        self.assertEqual([], lock["allowed_inline"])


if __name__ == "__main__":
    unittest.main()
