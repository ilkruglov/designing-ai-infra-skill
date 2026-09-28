#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_anchors import (
    HEADING,
    LOCAL_SOURCE_ANCHOR,
    LOCK_RELATIVE_PATH,
    anchor_key,
    iter_skill_documents,
    normalize,
    section_text,
)

PLUGIN_NAME = "designing-ai-infra"
MARKETPLACE_NAME = "designing-ai-infra-skill"
PLUGIN_DIRECTORY = Path("plugins") / PLUGIN_NAME
PLUGIN_SOURCE = f"./{PLUGIN_DIRECTORY.as_posix()}"
SKILL_DIRECTORY = PLUGIN_DIRECTORY / "skills" / PLUGIN_NAME

BOOK_FILES = ("preface.md",) + tuple(f"chapter{n}.md" for n in range(1, 13))
REQUIRED_PATHS = (
    ".agents/plugins/marketplace.json",
    ".claude-plugin/marketplace.json",
    str(PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"),
    str(PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json"),
    str(PLUGIN_DIRECTORY / "LICENSE"),
    str(PLUGIN_DIRECTORY / "NOTICE"),
    str(PLUGIN_DIRECTORY / "SOURCE.json"),
    str(PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"),
    str(SKILL_DIRECTORY / "SKILL.md"),
    str(SKILL_DIRECTORY / "agents" / "openai.yaml"),
    str(SKILL_DIRECTORY / "data" / "hardware.json"),
    "README.md",
    "LICENSE",
    "NOTICE",
    "SOURCE.json",
) + tuple(
    str(SKILL_DIRECTORY / "references" / "source-book" / name) for name in BOOK_FILES
)
REQUIRED_ATTRIBUTIONS = {
    "README.md": (
        "https://github.com/bojieli",
        "https://github.com/bojieli/ai-infra-book",
        "https://github.com/ilkruglov/ai-infra-book-ru",
    ),
    "NOTICE": (
        "Bojie Li",
        "https://github.com/bojieli",
        "https://github.com/bojieli/ai-infra-book",
        "https://github.com/ilkruglov/ai-infra-book-ru",
    ),
    "SOURCE.json": (
        "Bojie Li",
        "https://github.com/bojieli",
        "https://github.com/bojieli/ai-infra-book",
        "https://github.com/ilkruglov/ai-infra-book-ru",
    ),
    str(PLUGIN_DIRECTORY / "NOTICE"): (
        "Bojie Li",
        "https://github.com/bojieli",
        "https://github.com/bojieli/ai-infra-book",
        "https://github.com/ilkruglov/ai-infra-book-ru",
    ),
    str(PLUGIN_DIRECTORY / "SOURCE.json"): (
        "Bojie Li",
        "https://github.com/bojieli",
        "https://github.com/bojieli/ai-infra-book",
        "https://github.com/ilkruglov/ai-infra-book-ru",
    ),
}
NON_LOCAL_SOURCE_PATH = re.compile(r"(?<![-\w/])book/")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")
# Цитата закрывается последней кавычкой перед ссылкой, а не первой встреченной:
# книга часто цитирует сама себя, и вложенные «...» иначе обрывали бы совпадение,
# из-за чего такая строка молча переставала считаться цитатой и не проверялась.
CHAPTER_QUOTE = re.compile(
    r"^>\s*«(?P<quote>.{3,200})»\s*[—-]\s*`"
    r"(?P<path>references/source-book/[A-Za-z0-9._/-]+\.md):(?P<start>\d+)`",
    re.MULTILINE,
)
CHAPTER_QUOTE_MARKER = re.compile(r"^>\s*«", re.MULTILINE)
CHAPTERS_DIRECTORY = SKILL_DIRECTORY / "references" / "chapters"
SKILL_LINE_LIMIT = 300
CALC_DIRECTORY = SKILL_DIRECTORY / "scripts" / "infra_calc"
CALC_TESTS_DIRECTORY = SKILL_DIRECTORY / "scripts" / "tests"
# cli/result — обвязка, checks — проверки входов: формул книги в них нет
CALC_EXCLUDED = {"__init__.py", "checks.py", "cli.py", "result.py"}
AUTHOR_RESULT = re.compile(
    r"^calculations/results/(?P<name>[\w.-]+\.json)#sha256=(?P<sha256>[0-9a-f]{64})$"
)
# Разреженный клон оригинала на пине; в git не входит и может отсутствовать
AUTHOR_RESULTS_CLONE = Path(".tmp") / "upcalc" / "calculations" / "results"
SOURCE_BOOK_DIRECTORY_NAME = "source-book"
CJK = re.compile(r"[\u3000-\u9fff]")
REFERENCE_PATH = re.compile(r"references/[A-Za-z0-9._/-]+\.md")
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\."
    r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
PLUGIN_MANIFEST_FIELDS = {
    "id",
    "name",
    "version",
    "description",
    "skills",
    "apps",
    "mcpServers",
    "interface",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
}
PLUGIN_INTERFACE_FIELDS = {
    "displayName",
    "shortDescription",
    "longDescription",
    "developerName",
    "category",
    "capabilities",
    "websiteURL",
    "privacyPolicyURL",
    "termsOfServiceURL",
    "brandColor",
    "composerIcon",
    "logo",
    "logoDark",
    "screenshots",
    "defaultPrompt",
    "default_prompt",
}
CLAUDE_MARKETPLACE_SCHEMA = "https://anthropic.com/claude-code/marketplace.schema.json"
CLAUDE_PLUGIN_SCHEMA = "https://json.schemastore.org/claude-code-plugin-manifest.json"
CLAUDE_MARKETPLACE_FIELDS = {
    "$schema",
    "name",
    "description",
    "owner",
    "plugins",
}
CLAUDE_MARKETPLACE_PLUGIN_FIELDS = {
    "name",
    "source",
    "description",
    "version",
    "category",
}
CLAUDE_PLUGIN_MANIFEST_FIELDS = {
    "$schema",
    "name",
    "displayName",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
}
MARKETPLACE_INSTALLATION_POLICIES = {
    "NOT_AVAILABLE",
    "AVAILABLE",
    "INSTALLED_BY_DEFAULT",
}
MARKETPLACE_AUTHENTICATION_POLICIES = {"ON_INSTALL", "ON_USE"}


def parse_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    return unquote(target.split("#", maxsplit=1)[0])


def parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}

    values: dict[str, str] = {}
    for line in lines[1:]:
        if line == "---":
            return values
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = value.strip()
    return {}


def is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_https_url(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def add_contract_error(errors: list[str], contract: str, detail: str) -> None:
    errors.append(f"invalid {contract}: {detail}")


# .superpowers/ и docs/superpowers/ — рабочие материалы SDD-процесса разработки
# этого репозитория (см. .gitignore); в публикуемый репозиторий не попадают,
# поэтому валидатор должен их игнорировать так же, как .git и .tmp.
EXCLUDED_TOP_LEVEL_DIRECTORIES = (".git", ".tmp", ".superpowers")
EXCLUDED_NESTED_DIRECTORIES = (("docs", "superpowers"),)


def is_excluded_path(relative_parts: tuple[str, ...]) -> bool:
    if relative_parts and relative_parts[0] in EXCLUDED_TOP_LEVEL_DIRECTORIES:
        return True
    return any(
        relative_parts[: len(prefix)] == prefix
        for prefix in EXCLUDED_NESTED_DIRECTORIES
    )


def iter_markdown_documents(root: Path) -> list[Path]:
    source_root = root / SKILL_DIRECTORY / "references" / "source-book"
    return [
        path
        for path in root.rglob("*.md")
        if source_root not in path.parents
        and not is_excluded_path(path.relative_to(root).parts)
    ]


def iter_source_anchor_documents(root: Path) -> list[Path]:
    documents = iter_markdown_documents(root)
    evals_root = root / PLUGIN_DIRECTORY / "evals"
    if evals_root.is_dir():
        documents.extend(evals_root.rglob("*.json"))
        documents.extend(evals_root.rglob("*.jsonl"))
    return sorted(set(documents))


def validate_marketplace(root: Path, errors: list[str]) -> None:
    marketplace_path = root / ".agents" / "plugins" / "marketplace.json"
    if not marketplace_path.is_file():
        return
    try:
        marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(marketplace, dict):
        add_contract_error(errors, "marketplace contract", "root must be an object")
        return
    if marketplace.get("name") != MARKETPLACE_NAME:
        errors.append(f"invalid marketplace name: expected {MARKETPLACE_NAME}")

    unknown_fields = set(marketplace) - {"name", "interface", "plugins"}
    if unknown_fields:
        add_contract_error(
            errors,
            "marketplace contract",
            f"unsupported fields: {', '.join(sorted(unknown_fields))}",
        )

    interface = marketplace.get("interface")
    if interface is not None and not isinstance(interface, dict):
        add_contract_error(
            errors,
            "marketplace contract",
            "interface must be an object",
        )
    elif isinstance(interface, dict):
        interface_fields = set(interface)
        if interface_fields - {"displayName"}:
            add_contract_error(
                errors,
                "marketplace contract",
                "interface contains unsupported fields",
            )
        if "displayName" in interface and not is_non_empty_string(
            interface["displayName"]
        ):
            add_contract_error(
                errors,
                "marketplace contract",
                "interface.displayName must be a non-empty string",
            )

    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        errors.append("invalid marketplace plugin source: plugins must be an array")
        return

    entry = next(
        (
            value
            for value in plugins
            if isinstance(value, dict) and value.get("name") == PLUGIN_NAME
        ),
        None,
    )
    if entry is None:
        add_contract_error(
            errors,
            "marketplace contract",
            f"missing plugin entry {PLUGIN_NAME}",
        )

    source = entry.get("source") if isinstance(entry, dict) else None
    source_path = source.get("path") if isinstance(source, dict) else None
    resolved_source = root / str(source_path).removeprefix("./")
    if source_path != PLUGIN_SOURCE or not resolved_source.is_dir():
        errors.append(
            "invalid marketplace plugin source: expected "
            f"{PLUGIN_NAME} at {PLUGIN_SOURCE}"
        )

    if not isinstance(entry, dict):
        return
    unknown_entry_fields = set(entry) - {"name", "source", "policy", "category"}
    if unknown_entry_fields:
        add_contract_error(
            errors,
            "marketplace contract",
            f"plugin entry contains unsupported fields: "
            f"{', '.join(sorted(unknown_entry_fields))}",
        )

    if not isinstance(source, dict) or source.get("source") != "local":
        add_contract_error(
            errors,
            "marketplace contract",
            "plugin source.source must be local",
        )
    elif set(source) - {"source", "path"}:
        add_contract_error(
            errors,
            "marketplace contract",
            "plugin source contains unsupported fields",
        )

    policy = entry.get("policy")
    if not isinstance(policy, dict):
        add_contract_error(
            errors,
            "marketplace contract",
            "plugin policy must be an object",
        )
    else:
        unknown_policy_fields = set(policy) - {
            "installation",
            "authentication",
            "products",
        }
        if unknown_policy_fields:
            add_contract_error(
                errors,
                "marketplace contract",
                "plugin policy contains unsupported fields",
            )
        if policy.get("installation") not in MARKETPLACE_INSTALLATION_POLICIES:
            add_contract_error(
                errors,
                "marketplace contract",
                "plugin policy.installation is invalid",
            )
        if policy.get("authentication") not in MARKETPLACE_AUTHENTICATION_POLICIES:
            add_contract_error(
                errors,
                "marketplace contract",
                "plugin policy.authentication is invalid",
            )
        products = policy.get("products")
        if products is not None and (
            not isinstance(products, list)
            or not all(is_non_empty_string(product) for product in products)
        ):
            add_contract_error(
                errors,
                "marketplace contract",
                "plugin policy.products must be an array of strings",
            )

    if not is_non_empty_string(entry.get("category")):
        add_contract_error(
            errors,
            "marketplace contract",
            "plugin category must be a non-empty string",
        )


def validate_plugin_manifest(root: Path, errors: list[str]) -> None:
    manifest_path = root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        add_contract_error(errors, "plugin manifest", "root must be an object")
        return

    unknown_fields = set(manifest) - PLUGIN_MANIFEST_FIELDS
    if unknown_fields:
        add_contract_error(
            errors,
            "plugin manifest",
            f"unsupported fields: {', '.join(sorted(unknown_fields))}",
        )

    if manifest.get("name") != PLUGIN_NAME:
        add_contract_error(
            errors,
            "plugin manifest",
            f"name must be {PLUGIN_NAME}",
        )
    version = manifest.get("version")
    if not isinstance(version, str) or SEMVER.fullmatch(version) is None:
        add_contract_error(errors, "plugin manifest", "version must be strict semver")
    if not is_non_empty_string(manifest.get("description")):
        add_contract_error(
            errors,
            "plugin manifest",
            "description must be a non-empty string",
        )
    if manifest.get("skills") != "./skills/":
        add_contract_error(
            errors,
            "plugin manifest",
            "skills path must be ./skills/",
        )

    author = manifest.get("author")
    if not isinstance(author, dict):
        add_contract_error(errors, "plugin manifest", "author must be an object")
    else:
        if set(author) - {"name", "email", "url"}:
            add_contract_error(
                errors,
                "plugin manifest",
                "author contains unsupported fields",
            )
        if not is_non_empty_string(author.get("name")):
            add_contract_error(
                errors,
                "plugin manifest",
                "author.name must be a non-empty string",
            )
        if "email" in author and not is_non_empty_string(author["email"]):
            add_contract_error(
                errors,
                "plugin manifest",
                "author.email must be a non-empty string",
            )
        if "url" in author and not is_https_url(author["url"]):
            add_contract_error(
                errors,
                "plugin manifest",
                "author.url must be an absolute HTTPS URL",
            )

    for field in ("homepage", "repository"):
        if field in manifest and not is_https_url(manifest[field]):
            add_contract_error(
                errors,
                "plugin manifest",
                f"{field} must be an absolute HTTPS URL",
            )
    if "license" in manifest and not is_non_empty_string(manifest["license"]):
        add_contract_error(
            errors,
            "plugin manifest",
            "license must be a non-empty string",
        )
    keywords = manifest.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list)
        or not all(is_non_empty_string(keyword) for keyword in keywords)
    ):
        add_contract_error(
            errors,
            "plugin manifest",
            "keywords must be an array of strings",
        )

    interface = manifest.get("interface")
    if not isinstance(interface, dict):
        add_contract_error(errors, "plugin manifest", "interface must be an object")
        return
    unknown_interface_fields = set(interface) - PLUGIN_INTERFACE_FIELDS
    if unknown_interface_fields:
        add_contract_error(
            errors,
            "plugin manifest",
            "interface contains unsupported fields: "
            f"{', '.join(sorted(unknown_interface_fields))}",
        )
    for field in (
        "displayName",
        "shortDescription",
        "longDescription",
        "developerName",
        "category",
    ):
        if not is_non_empty_string(interface.get(field)):
            add_contract_error(
                errors,
                "plugin manifest",
                f"interface.{field} must be a non-empty string",
            )
    capabilities = interface.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        is_non_empty_string(capability) for capability in capabilities
    ):
        add_contract_error(
            errors,
            "plugin manifest",
            "interface.capabilities must be an array of strings",
        )
    default_prompt = interface.get(
        "defaultPrompt",
        interface.get("default_prompt"),
    )
    if not is_non_empty_string(default_prompt) and (
        not isinstance(default_prompt, list)
        or not default_prompt
        or not all(is_non_empty_string(prompt) for prompt in default_prompt)
    ):
        add_contract_error(
            errors,
            "plugin manifest",
            "interface.defaultPrompt must be a string or an array of strings",
        )
    for field in ("websiteURL", "privacyPolicyURL", "termsOfServiceURL"):
        if field in interface and not is_https_url(interface[field]):
            add_contract_error(
                errors,
                "plugin manifest",
                f"interface.{field} must be an absolute HTTPS URL",
            )


def validate_claude_marketplace(root: Path, errors: list[str]) -> None:
    marketplace_path = root / ".claude-plugin" / "marketplace.json"
    if not marketplace_path.is_file():
        return
    try:
        marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(marketplace, dict):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "root must be an object",
        )
        return

    unknown_fields = set(marketplace) - CLAUDE_MARKETPLACE_FIELDS
    if unknown_fields:
        add_contract_error(
            errors,
            "Claude marketplace contract",
            f"unsupported fields: {', '.join(sorted(unknown_fields))}",
        )
    if marketplace.get("$schema") != CLAUDE_MARKETPLACE_SCHEMA:
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "unexpected $schema",
        )
    if marketplace.get("name") != MARKETPLACE_NAME:
        add_contract_error(
            errors,
            "Claude marketplace contract",
            f"name must be {MARKETPLACE_NAME}",
        )
    if not is_non_empty_string(marketplace.get("description")):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "description must be a non-empty string",
        )

    owner = marketplace.get("owner")
    if not isinstance(owner, dict):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "owner must be an object",
        )
    else:
        if set(owner) - {"name", "email"}:
            add_contract_error(
                errors,
                "Claude marketplace contract",
                "owner contains unsupported fields",
            )
        if not is_non_empty_string(owner.get("name")):
            add_contract_error(
                errors,
                "Claude marketplace contract",
                "owner.name must be a non-empty string",
            )
        if "email" in owner and not is_non_empty_string(owner["email"]):
            add_contract_error(
                errors,
                "Claude marketplace contract",
                "owner.email must be a non-empty string",
            )

    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "plugins must be an array",
        )
        return
    entry = next(
        (
            value
            for value in plugins
            if isinstance(value, dict) and value.get("name") == PLUGIN_NAME
        ),
        None,
    )
    if not isinstance(entry, dict):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            f"missing plugin entry {PLUGIN_NAME}",
        )
        return

    unknown_entry_fields = set(entry) - CLAUDE_MARKETPLACE_PLUGIN_FIELDS
    if unknown_entry_fields:
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "plugin entry contains unsupported fields: "
            f"{', '.join(sorted(unknown_entry_fields))}",
        )
    source = entry.get("source")
    resolved_source = root / str(source).removeprefix("./")
    if source != PLUGIN_SOURCE or not resolved_source.is_dir():
        errors.append(
            "invalid Claude marketplace source: expected "
            f"{PLUGIN_NAME} at {PLUGIN_SOURCE}"
        )
    if not is_non_empty_string(entry.get("description")):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "plugin description must be a non-empty string",
        )
    version = entry.get("version")
    if not isinstance(version, str) or SEMVER.fullmatch(version) is None:
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "plugin version must be strict semver",
        )
    if not is_non_empty_string(entry.get("category")):
        add_contract_error(
            errors,
            "Claude marketplace contract",
            "plugin category must be a non-empty string",
        )


def validate_claude_plugin_manifest(root: Path, errors: list[str]) -> None:
    manifest_path = root / PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        add_contract_error(
            errors,
            "Claude plugin manifest",
            "root must be an object",
        )
        return

    unknown_fields = set(manifest) - CLAUDE_PLUGIN_MANIFEST_FIELDS
    if unknown_fields:
        add_contract_error(
            errors,
            "Claude plugin manifest",
            f"unsupported fields: {', '.join(sorted(unknown_fields))}",
        )
    if manifest.get("$schema") != CLAUDE_PLUGIN_SCHEMA:
        add_contract_error(
            errors,
            "Claude plugin manifest",
            "unexpected $schema",
        )
    if manifest.get("name") != PLUGIN_NAME:
        add_contract_error(
            errors,
            "Claude plugin manifest",
            f"name must be {PLUGIN_NAME}",
        )
    for field in ("displayName", "description", "license"):
        if not is_non_empty_string(manifest.get(field)):
            add_contract_error(
                errors,
                "Claude plugin manifest",
                f"{field} must be a non-empty string",
            )
    version = manifest.get("version")
    if not isinstance(version, str) or SEMVER.fullmatch(version) is None:
        add_contract_error(
            errors,
            "Claude plugin manifest",
            "version must be strict semver",
        )

    author = manifest.get("author")
    if not isinstance(author, dict):
        add_contract_error(
            errors,
            "Claude plugin manifest",
            "author must be an object",
        )
    else:
        if set(author) - {"name", "email", "url"}:
            add_contract_error(
                errors,
                "Claude plugin manifest",
                "author contains unsupported fields",
            )
        if not is_non_empty_string(author.get("name")):
            add_contract_error(
                errors,
                "Claude plugin manifest",
                "author.name must be a non-empty string",
            )
        if "email" in author and not is_non_empty_string(author["email"]):
            add_contract_error(
                errors,
                "Claude plugin manifest",
                "author.email must be a non-empty string",
            )
        if "url" in author and not is_https_url(author["url"]):
            add_contract_error(
                errors,
                "Claude plugin manifest",
                "author.url must be an absolute HTTPS URL",
            )

    for field in ("homepage", "repository"):
        if not is_https_url(manifest.get(field)):
            add_contract_error(
                errors,
                "Claude plugin manifest",
                f"{field} must be an absolute HTTPS URL",
            )
    keywords = manifest.get("keywords")
    if not isinstance(keywords, list) or not all(
        is_non_empty_string(keyword) for keyword in keywords
    ):
        add_contract_error(
            errors,
            "Claude plugin manifest",
            "keywords must be an array of strings",
        )


def validate_plugin_versions(root: Path, errors: list[str]) -> None:
    paths = {
        "Codex": root / PLUGIN_DIRECTORY / ".codex-plugin" / "plugin.json",
        "Claude": root / PLUGIN_DIRECTORY / ".claude-plugin" / "plugin.json",
        "Claude marketplace": root / ".claude-plugin" / "marketplace.json",
    }
    versions: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if label == "Claude marketplace":
            plugins = payload.get("plugins")
            if not isinstance(plugins, list):
                return
            entry = next(
                (
                    value
                    for value in plugins
                    if isinstance(value, dict) and value.get("name") == PLUGIN_NAME
                ),
                None,
            )
            version = entry.get("version") if isinstance(entry, dict) else None
        else:
            version = payload.get("version")
        if not isinstance(version, str):
            return
        versions[label] = version

    if len(set(versions.values())) != 1:
        details = ", ".join(f"{label}={version}" for label, version in versions.items())
        errors.append(f"plugin versions differ: {details}")


def load_lock(root: Path, errors: list[str]) -> dict:
    lock_path = root / LOCK_RELATIVE_PATH
    if not lock_path.is_file():
        errors.append(f"missing required file: {LOCK_RELATIVE_PATH.as_posix()}")
        return {}
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(f"invalid JSON in {LOCK_RELATIVE_PATH.as_posix()}: {error}")
        return {}


def validate_source_lock(root: Path, lock: dict, errors: list[str]) -> None:
    anchors = lock.get("anchors")
    if not isinstance(anchors, dict):
        if lock:
            errors.append("invalid lock: anchors must be an object")
        return

    allowed_inline = lock.get("allowed_inline")
    allowed_keys = (
        {item.get("anchor") for item in allowed_inline if isinstance(item, dict)}
        if isinstance(allowed_inline, list)
        else set()
    )

    line_cache: dict[Path, list[str]] = {}
    for document in iter_skill_documents(root):
        relative_path = document.relative_to(root)
        text = document.read_text(encoding="utf-8")
        for match in LOCAL_SOURCE_ANCHOR.finditer(text):
            source_path = root / SKILL_DIRECTORY / match.group("path")
            if not source_path.is_file():
                continue
            start = int(match.group("start"))
            key = anchor_key(match.group("path"), start)
            entry = anchors.get(key)
            if entry is None:
                errors.append(
                    f"anchor missing from lock: {key} (referenced in {relative_path})"
                )
                continue
            if entry.get("kind") != "heading" and key not in allowed_keys:
                errors.append(
                    f"anchor is not a heading: {key} (referenced in {relative_path}); "
                    "point at a section heading or add it to allowed_inline "
                    "with a reason"
                )
            if source_path not in line_cache:
                line_cache[source_path] = source_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            lines = line_cache[source_path]
            if start > len(lines):
                continue
            digest = hashlib.sha256(lines[start - 1].encode("utf-8")).hexdigest()
            if digest != entry.get("line_sha256"):
                errors.append(
                    f"anchor drift: {key} no longer matches the locked line; "
                    "re-run scripts/build_source_lock.py and review the diff"
                )


def validate_chapter_quotes(root: Path, errors: list[str]) -> None:
    chapters_root = root / CHAPTERS_DIRECTORY
    if not chapters_root.is_dir():
        return

    line_cache: dict[Path, list[str]] = {}
    for chapter_path in sorted(chapters_root.glob("*.md")):
        relative_path = chapter_path.relative_to(root)
        text = chapter_path.read_text(encoding="utf-8")

        # Нераспознанная цитата опаснее неверной: она не проверяется и при этом
        # выглядит как подтверждённая ссылка на книгу.
        started = len(CHAPTER_QUOTE_MARKER.findall(text))
        parsed = len(CHAPTER_QUOTE.findall(text))
        if started == 0:
            errors.append(
                f"chapter summary without verified quotes: {relative_path}; "
                "a summary that retells the book must cite it"
            )
        if started != parsed:
            errors.append(
                f"unparsed chapter quote in {relative_path}: "
                f"{started} quote lines, {parsed} parsed; "
                "expected format: > «текст» — `references/source-book/chapterN.md:LINE`"
            )

        for match in CHAPTER_QUOTE.finditer(text):
            source_path = root / SKILL_DIRECTORY / match.group("path")
            if not source_path.is_file():
                errors.append(f"source anchor file missing: {match.group('path')}")
                continue
            if source_path not in line_cache:
                line_cache[source_path] = source_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            start = int(match.group("start"))
            haystack = section_text(line_cache[source_path], start)
            needle = normalize(match.group("quote"))
            if needle not in haystack:
                errors.append(
                    "quote not found in anchor section: "
                    f"{relative_path} -> {match.group('path')}:{start}"
                )


def validate_benchmark_coverage(root: Path, errors: list[str]) -> None:
    """Каждый playbook и шаблон обязан быть покрыт сценариями бенчмарка.

    Без этой проверки новый playbook добавляется без eval, и его качество
    остаётся непроверенным до первого реального запроса.
    """
    benchmark_path = root / PLUGIN_DIRECTORY / "evals" / "benchmark-v1.json"
    if not benchmark_path.is_file():
        errors.append(f"missing required file: {benchmark_path.relative_to(root)}")
        return
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return  # об ошибке уже сообщила общая проверка JSON

    scenarios = payload.get("evals")
    if not isinstance(scenarios, list):
        errors.append("invalid benchmark: evals must be a list")
        return

    covered: dict[str, int] = {}
    for scenario in scenarios:
        for target in scenario.get("covers", []):
            covered[target] = covered.get(target, 0) + 1
            if not (root / SKILL_DIRECTORY / target).is_file():
                errors.append(
                    f"benchmark covers a missing file: {target} "
                    f"(scenario {scenario.get('name', '?')})"
                )

    for group, minimum in (("playbooks", 2), ("templates", 1)):
        group_root = root / SKILL_DIRECTORY / "references" / group
        if not group_root.is_dir():
            continue
        for path in sorted(group_root.glob("*.md")):
            key = f"references/{group}/{path.name}"
            if covered.get(key, 0) < minimum:
                errors.append(
                    f"insufficient benchmark coverage: {key} "
                    f"covered by {covered.get(key, 0)} scenarios, need {minimum}"
                )


def validate_skill_routing(root: Path, errors: list[str]) -> None:
    skill_path = root / SKILL_DIRECTORY / "SKILL.md"
    if not skill_path.is_file():
        return

    text = skill_path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    if line_count > SKILL_LINE_LIMIT:
        errors.append(
            f"SKILL.md exceeds {SKILL_LINE_LIMIT} lines: {line_count}; "
            "move detail into references/"
        )

    listed = {
        match.group(0)
        for match in REFERENCE_PATH.finditer(text)
        if "source-book/" not in match.group(0)
    }
    references_root = root / SKILL_DIRECTORY / "references"
    present = {
        path.relative_to(root / SKILL_DIRECTORY).as_posix()
        for path in references_root.rglob("*.md")
        if "source-book" not in path.parts
    }
    for missing in sorted(present - listed):
        errors.append(f"reference file not listed in SKILL.md: {missing}")
    for dangling in sorted(listed - present):
        errors.append(f"SKILL.md lists missing reference file: {dangling}")


def _test_anchors(tree: ast.Module) -> list[str]:
    anchors: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "ANCHORS" for t in targets):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List)):
            anchors = [
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    return anchors


def _check_test_anchor(
    root: Path, test_name: str, anchor: str, errors: list[str]
) -> None:
    match = LOCAL_SOURCE_ANCHOR.fullmatch(anchor)
    if match:
        path = root / SKILL_DIRECTORY / match.group("path")
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        start = int(match.group("start"))
        if not (1 <= start <= len(lines)) or not HEADING.match(lines[start - 1]):
            errors.append(
                f"calculator test anchor is not a heading: {test_name}: {anchor}"
            )
        return
    author = AUTHOR_RESULT.match(anchor)
    if not author:
        errors.append(f"calculator test anchor has unknown form: {test_name}: {anchor}")
        return
    local = root / AUTHOR_RESULTS_CLONE / author.group("name")
    if local.is_file():
        actual = hashlib.sha256(local.read_bytes()).hexdigest()
        if actual != author.group("sha256"):
            errors.append(
                f"calculator test anchor hash mismatch: {test_name}: {anchor} "
                f"(local clone has {actual})"
            )


def validate_calculator_coverage(root: Path, errors: list[str]) -> None:
    """Каждая публичная функция калькулятора вызывается в тесте с эталоном.

    Эталон — якорь на заголовок книги или хешированный результат автора в
    кортеже ANCHORS тестового модуля. Функция без такого теста выдаёт числа,
    которые никто не сверял с книгой.
    """
    calc_dir = root / CALC_DIRECTORY
    tests_dir = root / CALC_TESTS_DIRECTORY
    if not calc_dir.is_dir():
        return
    anchored_text: list[str] = []
    for test_file in sorted(tests_dir.glob("test_*.py")):
        text = test_file.read_text(encoding="utf-8")
        anchors = _test_anchors(ast.parse(text))
        for anchor in anchors:
            _check_test_anchor(root, test_file.name, anchor, errors)
        if anchors:
            anchored_text.append(text)
    corpus = "\n".join(anchored_text)
    for module in sorted(calc_dir.glob("*.py")):
        if module.name in CALC_EXCLUDED:
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                call = rf"\b{re.escape(module.stem)}\.{re.escape(node.name)}\("
                if not re.search(call, corpus):
                    errors.append(
                        "calculator function without anchored test: "
                        f"{module.stem}.{node.name}"
                    )


def validate_data_integrity(root: Path, errors: list[str]) -> None:
    """Снимок железа совпадает с записью в SOURCE.json, пути в ней разрешаются,
    в текстах скилла нет CJK-остатков перевода.

    SOURCE.json в корне отсчитывает пути от корня репозитория, SOURCE.json
    плагина — от каталога плагина, как его видит установленный плагин.
    """
    data = root / SKILL_DIRECTORY / "data" / "hardware.json"
    actual = hashlib.sha256(data.read_bytes()).hexdigest() if data.is_file() else None
    for source, base in (
        (Path("SOURCE.json"), root),
        (PLUGIN_DIRECTORY / "SOURCE.json", root / PLUGIN_DIRECTORY),
    ):
        try:
            payload = json.loads((root / source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # отсутствие и неверный JSON уже сообщены общими проверками
        for key, expect_directory in (
            ("bundled_sources.directory", True),
            ("data.hardware_json.path", False),
        ):
            value = _dotted(payload, key)
            if not isinstance(value, str):
                errors.append(f"SOURCE.json lacks {key}: {source}")
                continue
            target = base / value
            if not (target.is_dir() if expect_directory else target.is_file()):
                errors.append(
                    f"SOURCE.json path does not resolve: {source} {key}={value}"
                )
        expected = _dotted(payload, "data.hardware_json.sha256")
        if not isinstance(expected, str):
            errors.append(f"SOURCE.json lacks data.hardware_json.sha256: {source}")
        elif actual is not None and actual != expected:
            errors.append(
                f"hardware.json sha256 mismatch with {source}: "
                f"recorded {expected}, actual {actual}"
            )

    references = root / SKILL_DIRECTORY / "references"
    for document in sorted(references.rglob("*.md")):
        if SOURCE_BOOK_DIRECTORY_NAME in document.relative_to(references).parts:
            continue
        lines = document.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if CJK.search(line):
                errors.append(f"CJK artifact: {document.relative_to(root)}:{number}")


def _dotted(payload: object, key: str) -> object:
    for part in key.split("."):
        if not isinstance(payload, dict):
            return None
        payload = payload.get(part)
    return payload


def validate_repository(root: Path) -> list[str]:
    errors: list[str] = []
    line_counts: dict[Path, int] = {}
    for relative_path in REQUIRED_PATHS:
        if not (root / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    for relative_path, required_values in REQUIRED_ATTRIBUTIONS.items():
        path = root / relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for required_value in required_values:
            if required_value not in text:
                errors.append(
                    f"missing attribution in {relative_path}: {required_value}"
                )

    skill_path = root / SKILL_DIRECTORY / "SKILL.md"
    if skill_path.is_file():
        frontmatter = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
        if frontmatter.get("name") != "designing-ai-infra" or not frontmatter.get(
            "description"
        ):
            errors.append(
                "invalid SKILL.md frontmatter: expected name "
                "designing-ai-infra and a non-empty description"
            )

    for json_path in root.rglob("*.json"):
        relative_path = json_path.relative_to(root)
        if is_excluded_path(relative_path.parts):
            continue
        try:
            json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid JSON in {relative_path}: {error}")

    for jsonl_path in root.rglob("*.jsonl"):
        relative_path = jsonl_path.relative_to(root)
        if is_excluded_path(relative_path.parts):
            continue
        for line_number, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"invalid JSONL in {relative_path}:{line_number}: {error}"
                )

    validate_marketplace(root, errors)
    validate_plugin_manifest(root, errors)
    validate_claude_marketplace(root, errors)
    validate_claude_plugin_manifest(root, errors)
    validate_plugin_versions(root, errors)

    lock = load_lock(root, errors)
    validate_source_lock(root, lock, errors)
    validate_chapter_quotes(root, errors)
    validate_calculator_coverage(root, errors)
    validate_data_integrity(root, errors)
    validate_skill_routing(root, errors)
    validate_benchmark_coverage(root, errors)

    for document_path in iter_source_anchor_documents(root):
        text = document_path.read_text(encoding="utf-8")
        for match in NON_LOCAL_SOURCE_PATH.finditer(text):
            relative_path = document_path.relative_to(root)
            errors.append(
                f"non-local source anchor in {relative_path}: {match.group(0)}"
            )
        for match in LOCAL_SOURCE_ANCHOR.finditer(text):
            source_path = root / SKILL_DIRECTORY / match.group("path")
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            if not source_path.is_file():
                errors.append(f"source anchor file missing: {match.group('path')}")
                continue
            if source_path not in line_counts:
                line_counts[source_path] = len(
                    source_path.read_text(encoding="utf-8").splitlines()
                )
            maximum = line_counts[source_path]
            if start < 1 or end < start or end > maximum:
                errors.append(
                    "source anchor out of range: "
                    f"{match.group(0)} (file has {maximum} lines)"
                )

    for markdown_path in iter_markdown_documents(root):
        text = markdown_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = parse_link_target(match.group("target"))
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved_target = (markdown_path.parent / target).resolve()
            if not resolved_target.exists():
                relative_path = markdown_path.relative_to(root)
                errors.append(
                    f"broken Markdown link in {relative_path}: {match.group('target')}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the designing-ai-infra plugin repository"
    )
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    errors = validate_repository(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Validation passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
