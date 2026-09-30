"""Команды калькулятора из документов скилла выполняются и дают показанные числа.

Проверяются SKILL.md и references/**/*.md, кроме полного текста книги
(references/source-book/), и README.md репозитория. Команды README тоже
запускаются из каталога скилла.

1. Блоки кода. Команды — строки `python3 …` блока, в котором есть `calc.py`,
   с продолжениями через `\\` и многострочными кавычками. Каждая команда
   запускается из каталога скилла через `bash -o pipefail` с тайм-аутом;
   ненулевой код возврата — ошибка. Если сразу за блоком (через пустые
   строки) идёт блок `text`, это показанный вывод: каждая его строка с
   числами должна найтись в фактическом выводе блока по порядку — с тем же
   началом строки до первого числа и с теми же числами в том же порядке с
   точностью до округления, показанного в документе. Строки без чисел не
   сравниваются: это формулы и пояснения; исключение — строки результата
   `**name**: …` без чисел («не вычисляется»), они сверяются целиком.
2. Инлайн-команды. Спан `python3 … calc.py …` вне блоков — полная команда:
   она тоже запускается, сверяется только код возврата, так как вывода
   рядом нет. Шаблоны с подстановкой (`<id>`) запустить нельзя; они
   перечислены в TEMPLATES с обоснованием, и каждый элемент списка обязан
   существовать в документах.
3. Фрагменты. Спан вида `calc.py model` или `calc.py training --dp` без
   `python3` — ссылка на команду в тексте, а не вызов. Её не запускают, но
   команда обязана существовать, а флаги — быть в её `--help`: так
   переименование команды или флага не пройдёт незамеченным. Требовать
   полные команды в прозе и в глоссарии значило бы раздувать текст ради
   теста.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins" / "designing-ai-infra" / "skills" / "designing-ai-infra"
TIMEOUT_SECONDS = 60
WORKERS = 8

# Инлайн-вызовы, которые нельзя выполнить как есть. Список держится коротким:
# каждый элемент — шаблон с подстановкой, конкретные вызовы той же команды
# запускаются из других документов.
TEMPLATES: dict[tuple[str, str], str] = {
    (
        "references/source-map.md",
        "python3 scripts/calc.py device --device <id>",
    ): "шаблон drift gate: <id> — любое устройство снимка; `device --device h100-sxm` "
    "и другие запускаются из глав и SKILL.md",
}

FENCE_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
INLINE = re.compile(r"(?<!`)`(?P<code>[^`\n]+)`(?!`)")
FRAGMENT = re.compile(
    r"(?<![\w/])(?:scripts/)?calc\.py\s+(?P<command>[a-z][\w-]*)(?P<rest>[^`]*)"
)
FLAG = re.compile(r"(?<![\w-])--[a-z][\w-]*")
NAMED = re.compile(r"^\*\*[\w-]+\*\*")
# число вне идентификатора: 8 190 735 360, 0.0086154, 1.7652e+24, -1
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?: \d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


@dataclass
class Block:
    path: str
    line: int
    info: str
    lines: list[str]
    end: int  # номер строки закрывающего забора


@dataclass
class CommandBlock:
    path: str
    line: int
    commands: list[str]
    expected: list[str] | None = None
    expected_line: int | None = None
    outputs: list[str] = field(default_factory=list)


def documents() -> list[Path]:
    references = [
        path
        for path in sorted((SKILL / "references").rglob("*.md"))
        if "source-book" not in path.relative_to(SKILL).parts
    ]
    return [SKILL / "SKILL.md", *references, ROOT / "README.md"]


def label(path: Path) -> str:
    """Путь документа в сообщениях: от каталога скилла, README — от корня."""
    base = SKILL if path.is_relative_to(SKILL) else ROOT
    return path.relative_to(base).as_posix()


def parse_blocks(path: Path) -> tuple[list[Block], set[int]]:
    """Блоки кода по CommonMark и номера строк внутри них (с заборами)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    relative = label(path)
    blocks: list[Block] = []
    inside: set[int] = set()
    index = 0
    while index < len(lines):
        match = FENCE_OPEN.match(lines[index])
        if not match:
            index += 1
            continue
        fence = match.group("fence")
        closing = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")
        end = index + 1
        while end < len(lines) and not closing.match(lines[end]):
            end += 1
        blocks.append(
            Block(
                relative,
                index + 1,
                match.group("info").strip(),
                lines[index + 1 : end],
                end + 1,
            )
        )
        inside.update(range(index + 1, end + 2))
        index = end + 1
    return blocks, inside


def split_commands(lines: list[str]) -> list[str]:
    """Команды блока: продолжения через `\\` и многострочные кавычки склеиваются."""
    commands: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not buffer and (not line.strip() or line.lstrip().startswith("#")):
            continue
        buffer.append(line)
        text = "\n".join(buffer)
        if text.endswith("\\"):
            continue
        try:
            shlex.split(text.replace("\\\n", " "))
        except ValueError:  # кавычка не закрыта: команда продолжается
            continue
        commands.append(text)
        buffer = []
    if buffer:
        commands.append("\n".join(buffer))
    return commands


def command_blocks() -> list[CommandBlock]:
    found: list[CommandBlock] = []
    for path in documents():
        blocks, _ = parse_blocks(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, block in enumerate(blocks):
            commands = split_commands(block.lines)
            if not any(
                command.startswith("python3 ") and "calc.py" in command
                for command in commands
            ):
                continue
            item = CommandBlock(block.path, block.line, commands)
            if index + 1 < len(blocks):
                following = blocks[index + 1]
                between = lines[block.end : following.line - 1]
                if following.info == "text" and not any(s.strip() for s in between):
                    item.expected = following.lines
                    item.expected_line = following.line
            found.append(item)
    return found


def inline_spans() -> Iterator[tuple[str, int, str]]:
    for path in documents():
        _, inside = parse_blocks(path)
        relative = label(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if number in inside:
                continue
            for match in INLINE.finditer(line):
                yield relative, number, match.group("code").strip()


def run(command: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["bash", "-o", "pipefail", "-c", command],
            cwd=SKILL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command, 124, "", f"тайм-аут {TIMEOUT_SECONDS} s"
        )


def run_all(commands: list[str]) -> dict[str, subprocess.CompletedProcess[str]]:
    unique = list(dict.fromkeys(commands))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return dict(zip(unique, pool.map(run, unique)))


def numbers(text: str) -> list[tuple[Decimal, Decimal]]:
    """Числа строки и половина единицы последнего показанного разряда."""
    found = []
    for match in NUMBER.finditer(text):
        token = match.group(0).replace(" ", "")
        mantissa, _, exponent = token.lower().partition("e")
        decimals = len(mantissa.partition(".")[2])
        power = int(exponent) if exponent else 0
        found.append(
            (Decimal(token), Decimal(5) * Decimal(10) ** (power - decimals - 1))
        )
    return found


def same_numbers(expected: str, actual: str) -> bool:
    """Числа ожидаемой строки — по порядку среди чисел фактической."""
    remaining = iter(numbers(actual))
    for value, tolerance in numbers(expected):
        for other, other_tolerance in remaining:
            # половина единицы более грубого из двух показанных разрядов
            if abs(value - other) <= max(tolerance, other_tolerance):
                break
        else:
            return False
    return True


def missing_lines(expected: list[str], actual: list[str]) -> list[str]:
    """Строки показанного вывода с числами, не найденные по порядку в фактическом."""
    missing = []
    position = 0
    for line in expected:
        first = NUMBER.search(line)
        if not line.strip():
            continue
        if first is None:
            # строка результата без чисел («**name**: не вычисляется») сверяется
            # целиком: вместо неё в выводе не должно оказаться числа
            if NAMED.match(line.strip()):
                for index in range(position, len(actual)):
                    if actual[index].strip() == line.strip():
                        position = index + 1
                        break
                else:
                    missing.append(line)
            continue
        prefix = line[: first.start()].strip()
        for index in range(position, len(actual)):
            candidate = actual[index].strip()
            if candidate.startswith(prefix) and same_numbers(line, candidate):
                position = index + 1
                break
        else:
            missing.append(line)
    return missing


class DocCommandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.blocks = command_blocks()
        spans = list(inline_spans())
        cls.inline = [
            (path, line, code)
            for path, line, code in spans
            if code.startswith("python3 ") and "calc.py" in code
        ]
        cls.fragments = [
            (path, line, code)
            for path, line, code in spans
            if not code.startswith("python3 ") and FRAGMENT.search(code)
        ]
        runnable = [c for block in cls.blocks for c in block.commands]
        runnable += [
            code for path, _, code in cls.inline if (path, code) not in TEMPLATES
        ]
        cls.results = run_all(runnable)

    def test_documents_quote_commands(self) -> None:
        # страховка от пустой проверки: разбор нашёл блоки и инлайн-вызовы
        self.assertGreater(len(self.blocks), 100)
        self.assertGreater(len(self.inline), 50)

    def test_readme_commands_are_checked(self) -> None:
        # README репозитория показывает команды калькулятора пользователю;
        # они запускаются из каталога скилла, как команды SKILL.md
        readme = [
            command
            for block in self.blocks
            if block.path == "README.md"
            for command in block.commands
        ]
        quoted = [
            line.strip()
            for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("python3 scripts/calc.py")
        ]
        self.assertTrue(quoted)
        self.assertEqual(readme, quoted)

    def test_block_commands_are_calculator_calls(self) -> None:
        for block in self.blocks:
            for command in block.commands:
                with self.subTest(block=f"{block.path}:{block.line}", command=command):
                    self.assertTrue(command.startswith("python3 "), command)

    def test_block_commands_run(self) -> None:
        for block in self.blocks:
            for command in block.commands:
                result = self.results[command]
                with self.subTest(block=f"{block.path}:{block.line}", command=command):
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_shown_output_matches(self) -> None:
        for block in self.blocks:
            if block.expected is None:
                continue
            actual = "\n".join(self.results[c].stdout for c in block.commands)
            with self.subTest(block=f"{block.path}:{block.line}"):
                missing = missing_lines(block.expected, actual.splitlines())
                self.assertEqual(
                    missing,
                    [],
                    f"вывод в {block.path}:{block.expected_line} расходится с "
                    f"фактическим:\n{actual}",
                )

    def test_inline_commands_run(self) -> None:
        for path, line, code in self.inline:
            if (path, code) in TEMPLATES:
                continue
            result = self.results[code]
            with self.subTest(place=f"{path}:{line}", command=code):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_templates_are_still_quoted(self) -> None:
        quoted = {(path, code) for path, _, code in self.inline}
        for template, reason in TEMPLATES.items():
            with self.subTest(template=template):
                self.assertTrue(reason)
                self.assertIn(template, quoted)

    def test_fragments_name_existing_commands_and_flags(self) -> None:
        helps: dict[str, subprocess.CompletedProcess[str]] = {}
        for path, line, code in self.fragments:
            match = FRAGMENT.search(code)
            assert match is not None  # отобрано по FRAGMENT в setUpClass
            command = match.group("command")
            if command not in helps:
                helps[command] = subprocess.run(
                    [sys.executable, "scripts/calc.py", command, "--help"],
                    cwd=SKILL,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                    check=False,
                )
            result = helps[command]
            with self.subTest(place=f"{path}:{line}", fragment=code):
                self.assertEqual(result.returncode, 0, result.stdout)
                for flag in FLAG.findall(match.group("rest")):
                    exact = rf"(?<![\w-]){re.escape(flag)}(?![\w-])"
                    self.assertRegex(result.stdout, exact)


class ComparisonTest(unittest.TestCase):
    """Правила сверки чисел, на которых держится DocCommandsTest."""

    def test_rounding_shown_in_document(self) -> None:
        self.assertTrue(same_numbers("**x**: 12.8 s", "**x**: 12.8034 s"))
        self.assertFalse(same_numbers("**x**: 12.8 s", "**x**: 12.9 s"))
        self.assertTrue(same_numbers("**x**: 8 190 735 360", "**x**: 8 190 735 360"))
        self.assertFalse(same_numbers("**x**: 8 190 735 360", "**x**: 8 190 735 361"))
        self.assertTrue(same_numbers("**x**: 1.66015e+06 s", "**x**: 1660150.4 s"))

    def test_identifiers_are_not_numbers(self) -> None:
        self.assertEqual(numbers("qwen3-8b FP16 chapter8.md d0cc188b"), [])

    def test_order_and_prefix(self) -> None:
        actual = ["**a**: 1 B", "**b**: 2 B", "**a**: 3 B"]
        self.assertEqual(missing_lines(["**a**: 1 B", "**a**: 3 B"], actual), [])
        self.assertEqual(
            missing_lines(["**a**: 3 B", "**b**: 2 B"], actual), ["**b**: 2 B"]
        )
        self.assertEqual(missing_lines(["**c**: 1 B"], actual), ["**c**: 1 B"])

    def test_named_lines_without_numbers(self) -> None:
        # «не вычисляется» в документе, а в выводе число — расхождение
        actual = ["**x** (нижняя граница): 5 s", "**y**: не вычисляется"]
        self.assertEqual(
            missing_lines(["**x** (нижняя граница): не вычисляется"], actual),
            ["**x** (нижняя граница): не вычисляется"],
        )
        self.assertEqual(missing_lines(["**y**: не вычисляется"], actual), [])

    def test_commands_are_split_like_bash(self) -> None:
        lines = [
            "python3 scripts/calc.py a \\",
            "  --x 1 | python3 -c 'import sys",
            "print(1)'",
            "# комментарий",
            "python3 scripts/calc.py b",
        ]
        self.assertEqual(len(split_commands(lines)), 2)


if __name__ == "__main__":
    unittest.main()
