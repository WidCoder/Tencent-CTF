"""Safe, deterministic helpers for importing Google CTF challenges."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterable

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


def verification_files(source: Path) -> list[Path]:
    return [path for path in iter_files(source) if path.name.lower() in HASH_FILE_NAMES or "flagcheck" in path.name.lower()]


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
    checker_files = [path for path in copied if "flagcheck" in Path(path).name.lower()]
    if hash_files:
        return {"status": "eligible", "method": "sha256", "files": hash_files}
    if checker_files:
        return {"status": "eligible", "method": "flagcheck", "files": checker_files}
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
                return cleaned.strip(), cleaned != original
            except OSError:
                pass
    return f"Google CTF challenge: {path.name}", False


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
        if item.name in SKIP_DIR_NAMES or item.name.lower() in SECRET_FILE_NAMES:
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
