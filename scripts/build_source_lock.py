#!/usr/bin/env python3
"""Собрать lock-файл якорей на текст книги.

Валидатор не обновляет lock самостоятельно: расхождение — ошибка. Обновление
выполняется только этим скриптом и попадает в diff отдельным изменением,
поэтому сдвиг текста книги нельзя «залечить» незаметно для ревьюера.
Список allowed_inline (якоря-абзацы, разрешённые с причиной и хешем строки)
переносится из текущего lock, только пока строка под номером не изменилась.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from source_anchors import (
    HEADING,
    LOCAL_SOURCE_ANCHOR,
    LOCK_RELATIVE_PATH,
    SKILL_DIRECTORY,
    anchor_key,
    iter_skill_documents,
)

UPSTREAM_COMMIT = "3bdcb4fcab73010eeaccf31cc10a8242a895d82e"
TRANSLATION_COMMIT = "c9f4ec00ac658416c29857b1632a9ddbfe2ace3c"
LINE_TEXT_LIMIT = 200


def build_lock(root: Path) -> dict:
    anchors: dict[str, dict[str, str]] = {}
    line_cache: dict[Path, list[str]] = {}
    for document in iter_skill_documents(root):
        text = document.read_text(encoding="utf-8")
        for match in LOCAL_SOURCE_ANCHOR.finditer(text):
            source_path = root / SKILL_DIRECTORY / match.group("path")
            if not source_path.is_file():
                continue
            if source_path not in line_cache:
                line_cache[source_path] = source_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            lines = line_cache[source_path]
            start = int(match.group("start"))
            if start < 1 or start > len(lines):
                continue
            line = lines[start - 1]
            anchors[anchor_key(match.group("path"), start)] = {
                "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "line_text": line[:LINE_TEXT_LIMIT],
                "kind": "heading" if HEADING.match(line) else "inline",
            }
    return {
        "schema_version": 1,
        "book": {
            "upstream_commit": UPSTREAM_COMMIT,
            "translation_commit": TRANSLATION_COMMIT,
        },
        "anchors": dict(sorted(anchors.items())),
        "allowed_inline": _kept_allowlist(root, anchors),
    }


def _kept_allowlist(root: Path, anchors: dict[str, dict[str, str]]) -> list[dict]:
    """Разрешения allowed_inline из текущего lock, которые ещё относятся к той
    же строке книги.

    Разрешение на якорь-абзац с причиной вносит человек и привязывает к хешу
    строки (line_sha256). Пересборка переносит запись, только если документы
    по-прежнему ссылаются на этот номер и строка под ним не изменилась. После
    сдвига книги под номером оказывается другой абзац: запись отбрасывается, и
    валидатор снова требует заголовок или новое разрешение. Запись без хеша
    тоже отбрасывается — сверить её не с чем.
    """
    lock_path = root / LOCK_RELATIVE_PATH
    if not lock_path.is_file():
        return []
    try:
        current = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    allowlist = current.get("allowed_inline") if isinstance(current, dict) else None
    if not isinstance(allowlist, list):
        return []
    kept: list[dict] = []
    for item in allowlist:
        if not isinstance(item, dict):
            continue
        anchor = anchors.get(item.get("anchor"))
        if anchor is not None and item.get("line_sha256") == anchor["line_sha256"]:
            kept.append(item)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the source anchor lock file")
    parser.add_argument("root", nargs="?", default=".", type=Path)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    lock = build_lock(root)
    lock_path = root / LOCK_RELATIVE_PATH
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(lock['anchors'])} anchors to {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
