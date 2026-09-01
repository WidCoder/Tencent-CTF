"""Safe, deterministic helpers for importing Google CTF challenges."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import shlex
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - optional for conversion-only environments
    yaml = None

LOGGER = logging.getLogger("google_ctf_import")

CATEGORIES = {
    "pwn", "crypto", "web", "misc", "forensics", "forensic", "rev",
    "reversing", "hardware", "mobile", "network", "osint", "sandbox",
}
CATEGORY_MAP = {"reversing": "rev", "forensic": "forensics"}
ROUND_NAMES = {"qual", "quals", "qualification", "qualifications", "final", "finals"}
SKIP_DIR_NAMES = {".git", ".github", "__pycache__", ".venv", "venv", "node_modules"}
SECRET_FILE_NAMES = {"flag", "flag.txt", "flag.json"}
HASH_FILE_NAMES = {"flag.sha256", ".flag.sha256", "flag.sha256.txt"}
CHECKER_FILE_NAMES = {"checker.py", "verify.py", "checker.sh", "verify.sh"}
VERIFIER_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".bzr", "_darcs", "solution", "writeup", "writeups",
}
DESCRIPTION_FILE_NAMES = {"readme", "readme.md", "readme.txt", "description.md", "description.txt"}
PLAINTEXT_FLAG_RE = re.compile(r"(?i)\b(?:ctf|flag|pwn\.college|gc)\{[^\r\n}]{2,256}\}")


def read_source_metadata(source: Path) -> dict:
    """Read safe routing metadata without copying plaintext solutions."""
    for name in ("challenge.json", "challenge.yaml", "challenge.yml", "metadata.yaml", "metadata.yml"):
        path = source / name
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".json":
                value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            elif yaml is not None:
                value = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            else:
                value = {}
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    match = re.match(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$", line)
                    if match:
                        raw = match.group(2).strip().strip("'\"")
                        if raw.isdigit():
                            value[match.group(1)] = int(raw)
                        elif raw and raw.lower() not in {"null", "~"}:
                            value[match.group(1)] = raw
        except Exception as error:
            LOGGER.warning("Unable to parse metadata file %s: %s", path, error)
            continue
        if isinstance(value, dict):
            nested = value.get("challenge")
            return {**value, **nested} if isinstance(nested, dict) else value
    return {}


def runtime_metadata(source: Path) -> dict:
    raw = read_source_metadata(source)
    result: dict = {}
    for key in ("target_host", "flag_format", "proto"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    for key in ("box", "server_name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result["box"] = value.strip()
            break
    for key in ("internal_port", "port"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            result["internal_port"] = value
            break
        if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
            result["internal_port"] = int(value)
            break
    return result


def is_private_verifier_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in HASH_FILE_NAMES
        or lowered in CHECKER_FILE_NAMES
        or "flagcheck" in lowered
    )


def canonical_category(value: str) -> str:
    normalized = value.strip().lower()
    return CATEGORY_MAP.get(normalized, normalized)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "challenge"


def stable_task_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip("/")
    return f"{safe_name(normalized.replace('/', '__'))}-{hashlib.sha1(normalized.encode()).hexdigest()[:10]}"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


def iter_year_dirs(root: Path, year: str | None = None) -> Iterable[Path]:
    wanted = str(year) if year else None
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir() and re.fullmatch(r"20\d{2}", path.name) and (wanted is None or path.name == wanted):
            yield path


def should_skip_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.startswith(".") or part in SKIP_DIR_NAMES for part in parts)


def infer_event(path: Path, year_dir: Path) -> str:
    parts = [part.lower().replace("_", "-") for part in path.relative_to(year_dir).parts]
    for part in parts:
        if part in ROUND_NAMES:
            return "quals" if part.startswith("qual") else "finals"
    text = path.name.lower().replace("_", "-")
    if "-final" in text:
        return "finals"
    if "-qual" in text:
        return "quals"
    return "unknown"


def infer_category(path: Path, year_dir: Path) -> str | None:
    parts = [part.lower().replace("_", "-") for part in path.relative_to(year_dir).parts]
    for part in parts:
        normalized = canonical_category(part)
        if part in CATEGORIES or normalized in CATEGORIES:
            return normalized
    for part in reversed(parts):
        match = re.match(r"(?:20\d{2}-)?(?:(?:qual|quals|final|finals)-)?([a-z]+)(?:-|$)", part)
        if match:
            normalized = canonical_category(match.group(1))
            if normalized in CATEGORIES:
                return normalized
    return None


def is_challenge_dir(path: Path, year_dir: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        entries = list(path.iterdir())
    except OSError:
        return False
    names = {entry.name.lower() for entry in entries}
    markers = {
        "dockerfile", "docker-compose.yml", "docker-compose.yaml", "challenge.json",
        "description.md", "readme.md", "attachments", "dist", "public", "src",
        "source", "handout", "solution", "writeup", "writeups",
    }
    if names & markers:
        return True
    # Do not classify a category/event directory merely because it has a README.
    return any(entry.is_file() and not entry.name.startswith(".") for entry in entries) and not any(
        entry.is_dir() and infer_category(entry, year_dir) is not None for entry in entries
    )


def iter_files(path: Path) -> Iterable[Path]:
    """Yield regular files without following symlinked directories."""
    pending = [path]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        try:
            if current.is_symlink():
                continue
            key = str(current.resolve())
            if key in visited:
                continue
            visited.add(key)
            entries = sorted(current.iterdir(), key=lambda item: item.name.lower(), reverse=True)
        except OSError as error:
            LOGGER.warning("Unable to scan %s: %s", current, error)
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    pending.append(entry)
                elif entry.is_file():
                    yield entry
            except OSError:
                continue


def find_named(path: Path, names: set[str]) -> list[Path]:
    wanted = {name.lower() for name in names}
    return sorted((candidate for candidate in iter_files(path) if candidate.name.lower() in wanted), key=str)


def _metadata_verifier_values(metadata: dict) -> list[object]:
    values: list[object] = []
    verification = metadata.get("verification")
    if isinstance(verification, dict):
        if "files" in verification:
            values.append(verification["files"])
        for key in ("flagCheck", "flag_check", "verifier", "checker"):
            if key in verification:
                values.append(verification[key])
    for key in ("flagCheck", "flag_check", "verifier", "checker"):
        if key in metadata:
            values.append(metadata[key])
    return values


def _declared_verifier_paths(source: Path) -> list[Path]:
    metadata = read_source_metadata(source)
    paths: list[Path] = []
    for value in _metadata_verifier_values(metadata):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                item = item.get("path") or item.get("file") or item.get("command")
            if not isinstance(item, str) or not item.strip():
                continue
            try:
                tokens = shlex.split(item)
            except ValueError:
                tokens = [item]
            for token in tokens:
                candidate = Path(token)
                if candidate.is_absolute():
                    resolved = candidate.resolve()
                else:
                    resolved = (source / candidate).resolve()
                try:
                    resolved.relative_to(source.resolve())
                except ValueError:
                    continue
                relative_parts = {part.lower() for part in resolved.relative_to(source.resolve()).parts[:-1]}
                if relative_parts & VERIFIER_EXCLUDED_DIRS:
                    continue
                if resolved.is_file() and resolved not in paths:
                    paths.append(resolved)
    return paths


def verification_files(source: Path) -> list[Path]:
    declared = _declared_verifier_paths(source)
    discovered = [
        path for path in iter_files(source)
        if is_private_verifier_name(path.name)
        and not ({part.lower() for part in path.relative_to(source.resolve()).parts[:-1]} & VERIFIER_EXCLUDED_DIRS)
    ]
    return sorted({*declared, *discovered}, key=str)


def validate_verifier(path: Path, source: Path) -> bool:
    try:
        path.relative_to(source.resolve())
        if path.name.lower() in HASH_FILE_NAMES:
            text = path.read_text(encoding="utf-8", errors="replace")
            return bool(re.search(r"\b[0-9a-fA-F]{64}\b", text))
        return path.is_file() and path.stat().st_size > 0
    except (OSError, ValueError):
        return False


def copy_verification_files(source: Path, task: Path) -> list[str]:
    copied: list[str] = []
    for candidate in verification_files(source):
        if not validate_verifier(candidate, source):
            LOGGER.warning("Skipping invalid verifier artifact: %s", candidate)
            continue
        relative = candidate.relative_to(source)
        destination = task / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        copied.append(relative.as_posix())
    return sorted(copied)


def verification_metadata(copied: list[str]) -> dict:
    hash_files = [path for path in copied if Path(path).name.lower() in HASH_FILE_NAMES]
    checker_files = [
        path for path in copied
        if Path(path).name.lower() in CHECKER_FILE_NAMES
        or "flagcheck" in Path(path).name.lower()
    ]
    if hash_files:
        return {"status": "eligible", "method": "sha256", "files": hash_files, "hash_input": "full"}
    if checker_files:
        return {"status": "eligible", "method": "checker", "files": checker_files}
    return {"status": "pending_validation", "method": "unknown", "files": []}


def read_description(path: Path) -> tuple[str, bool]:
    for name in ("DESCRIPTION.md", "description.md", "README.md", "readme.md"):
        candidate = path / name
        if candidate.is_file():
            try:
                original = candidate.read_text(encoding="utf-8", errors="replace").strip()
                # Only remove whole lines that are explicit answer metadata. This avoids
                # destroying normal prose that happens to mention the word "flag".
                cleaned = re.sub(
                    r"(?im)^\s*(?:[#;>*-]\s*)?(?:the\s+)?flag\s*(?:is|=|:)\s*.+$\n?",
                    "",
                    original,
                )
                cleaned = re.sub(r"(?im)^\s*(?:solution|answer)\s*(?:=|:)\s*.+$\n?", "", cleaned)
                cleaned = PLAINTEXT_FLAG_RE.sub("[REDACTED_FLAG]", cleaned)
                return cleaned.strip(), cleaned != original
            except OSError:
                pass
    return f"Google CTF challenge: {path.name}", False


def extract_plaintext_flag(path: Path) -> str | None:
    """Return one unambiguous flag found in a public description, if any.

    Generic mentions such as ``submit the flag`` are intentionally ignored.
    Multiple different candidates are rejected for manual review rather than
    guessing which value is authoritative.
    """
    candidates: set[str] = set()
    for name in ("DESCRIPTION.md", "description.md", "README.md", "readme.md"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        candidates.update(match.group(0).strip() for match in PLAINTEXT_FLAG_RE.finditer(text))
    return next(iter(candidates)) if len(candidates) == 1 else None


def copy_tree_contents(source: Path, target: Path, root: Path, visited: set[str] | None = None) -> None:
    """Copy a tree while rejecting external symlinks and plaintext flag files."""
    visited = visited if visited is not None else set()
    try:
        resolved = source.resolve()
        resolved.relative_to(root.resolve())
        key = str(resolved)
        if key in visited:
            return
        visited.add(key)
        target.mkdir(parents=True, exist_ok=True)
        entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
    except (OSError, ValueError):
        return
    for item in entries:
        if (item.name in SKIP_DIR_NAMES
                or item.name.lower() in SECRET_FILE_NAMES
                or item.name.lower() in DESCRIPTION_FILE_NAMES
                or is_private_verifier_name(item.name)):
            continue
        try:
            resolved_item = item.resolve()
            resolved_item.relative_to(root.resolve())
            if resolved_item.is_dir():
                copy_tree_contents(resolved_item, target / item.name, root, visited)
            elif resolved_item.is_file():
                destination = target / item.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved_item, destination)
        except (OSError, ValueError):
            LOGGER.warning("Skipping unsafe or unreadable path: %s", item)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
