#!/usr/bin/env python3
"""Audit a converted CTF-Dojo dataset for answer/solution leakage.

This command is deliberately read-only.  It scans every challenge directory
containing ``challenge.json`` and reports explicit flag metadata in README
files, challenge metadata, and other text artifacts that could be copied into
the agent workspace.  It also reports solution/checker/verifier-like files so
they can be reviewed before trajectory generation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


README_NAMES = {"readme", "readme.md", "readme.txt", "description.md", "description.txt"}
SKIP_DIRS = {".git", ".github", "__pycache__", ".venv", "venv", "node_modules"}
SENSITIVE_PARTS = ("solution", "writeup", "checker", "verify", "flagcheck", "exploit")
FLAG_LINE_RE = re.compile(
    r"(?im)^\s*(?:[#;>*\-]\s*)?(?:the\s+)?flag\s*(?:is|=|:)\s*(\S.*)$"
)
FLAG_VALUE_RE = re.compile(r"(?i)\b(?:ctf|flag|pwn\.college|gc){[^\r\n}]{2,256}\}")
ANSWER_ASSIGN_RE = re.compile(r"(?im)^\s*(?:answer|solution|secret)\s*(?:=|:)\s*\S.+$")
FLAG_WORD_RE = re.compile(r"(?i)\bflag\b")
MAX_TEXT_BYTES = 2 * 1024 * 1024


def is_probably_text(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return False
        sample = path.read_bytes()[:8192]
    except OSError:
        return False
    return b"\x00" not in sample


def redacted(value: str) -> str:
    value = value.strip()
    if len(value) <= 12:
        return "<redacted>"
    return value[:6] + "..." + value[-4:]


def scan_file(path: Path) -> dict[str, Any] | None:
    if not is_probably_text(path):
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits: list[dict[str, Any]] = []
    flag_word_mentions = 0
    for number, line in enumerate(text.splitlines(), 1):
        flag_word_mentions += len(FLAG_WORD_RE.findall(line))
        if FLAG_LINE_RE.search(line):
            confidence = "high" if FLAG_VALUE_RE.search(line) else "medium"
            hits.append({"line": number, "kind": "explicit_flag_line", "confidence": confidence, "preview": redacted(line)})
        elif FLAG_VALUE_RE.search(line):
            hits.append({"line": number, "kind": "flag_format_value", "confidence": "high", "preview": redacted(line)})
        elif ANSWER_ASSIGN_RE.search(line):
            hits.append({"line": number, "kind": "answer_assignment", "confidence": "medium", "preview": redacted(line)})
    if not hits:
        return None
    return {"path": str(path), "hits": hits, "flag_word_mentions": flag_word_mentions}


def challenge_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("challenge.json")})


def audit_challenge(challenge_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    sensitive_files: list[str] = []
    readme_paths: list[str] = []
    readme_flag_word_mentions = 0
    all_flag_word_mentions = 0
    for path in sorted(challenge_dir.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        lowered_parts = tuple(part.lower() for part in path.relative_to(challenge_dir).parts)
        if path.name.lower() in README_NAMES:
            readme_paths.append(str(path.relative_to(challenge_dir)))
            try:
                readme_flag_word_mentions += len(FLAG_WORD_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
        if is_probably_text(path):
            try:
                all_flag_word_mentions += len(FLAG_WORD_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
        if any(any(token in part for token in SENSITIVE_PARTS) for part in lowered_parts):
            sensitive_files.append(str(path.relative_to(challenge_dir)))
        hit = scan_file(path)
        if hit:
            hit["path"] = str(path.relative_to(challenge_dir))
            files.append(hit)
    readme_hits = [item for item in files if Path(item["path"]).name.lower() in README_NAMES]
    metadata_hits = [item for item in files if Path(item["path"]).name.lower() in {"challenge.json", "challenge.yaml", "challenge.yml", "metadata.yaml", "metadata.yml"}]
    return {
        "challenge_dir": str(challenge_dir),
        "challenge_id": challenge_dir.name,
        "readme_flag_leak": bool(readme_hits),
        "readme_files": readme_paths,
        "readme_flag_word_mentions": readme_flag_word_mentions,
        "all_flag_word_mentions": all_flag_word_mentions,
        "metadata_flag_leak": bool(metadata_hits),
        "text_hits": files,
        "sensitive_files": sorted(set(sensitive_files)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, help="also write a JSON report")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")
    dirs = challenge_dirs(root)
    reports = [audit_challenge(path) for path in dirs]
    orphan_readmes = []
    known = set(dirs)
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in README_NAMES and path.parent not in known:
            orphan_readmes.append(str(path.relative_to(root)))
    summary = {
        "dataset_root": str(root),
        "challenge_count": len(reports),
        "readme_flag_leaks": sum(r["readme_flag_leak"] for r in reports),
        "challenges_with_readme": sum(bool(r["readme_files"]) for r in reports),
        "readme_file_count": sum(len(r["readme_files"]) for r in reports),
        "metadata_flag_leaks": sum(r["metadata_flag_leak"] for r in reports),
        "challenges_with_text_hits": sum(bool(r["text_hits"]) for r in reports),
        "challenges_with_sensitive_files": sum(bool(r["sensitive_files"]) for r in reports),
        "orphan_readmes": orphan_readmes,
        "challenges": reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
