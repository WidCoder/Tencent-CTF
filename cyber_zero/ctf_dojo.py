"""Import validated CTF-Dojo tasks into Cyber-Zero task metadata JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def solution_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for row in read_jsonl(path):
        value = str(row.get("solution", "")).strip()
        for key in (row.get("task_id"), row.get("task_name"), row.get("task_dir")):
            if key and value:
                values[str(key)] = value
    return values


def read_writeup(task_dir: Path) -> tuple[str, str]:
    for base in (task_dir / "solution", task_dir / "writeup", task_dir / "writeups"):
        if not base.is_dir():
            continue
        parts = []
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
        if parts:
            return str(base.resolve()), "\n\n".join(parts)
    return "", ""


def literal_solution(metadata: dict[str, Any]) -> str:
    value = metadata.get("flag") or metadata.get("solution") or ""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    # File references such as flag.json are not answers and must be mapped explicitly.
    if not value or "/" in value or "\\" in value or value.lower().endswith((".json", ".txt", ".sha256")):
        return ""
    return value


def task_directories(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [Path(row["task_dir"]) for row in read_jsonl(input_path) if row.get("task_dir")]
    return [path.parent for path in input_path.rglob("challenge.json")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CTF-Dojo dataset root or task manifest JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Cyber-Zero task metadata JSONL")
    parser.add_argument("--solutions", type=Path, help="Audited JSONL mapping task_id/task_name/task_dir to a literal solution")
    parser.add_argument("--allow-missing-writeup", action="store_true", help="Permit tasks without a local solution/writeup")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.output}; use --overwrite to replace it")

    mapped_solutions = solution_map(args.solutions)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for task_dir in task_directories(args.input):
        challenge_path = task_dir / "challenge.json"
        if not challenge_path.is_file():
            skipped.append(f"{task_dir}: missing challenge.json")
            continue
        try:
            metadata = json.loads(challenge_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(f"{task_dir}: invalid challenge.json")
            continue

        task_id = task_dir.name
        solution = (
            mapped_solutions.get(task_id)
            or mapped_solutions.get(str(task_dir.resolve()))
            or mapped_solutions.get(str(metadata.get("name", "")))
            or literal_solution(metadata)
        )
        writeup_path, writeup = read_writeup(task_dir)
        if not solution:
            skipped.append(f"{task_id}: no audited literal solution")
            continue
        if not writeup and not args.allow_missing_writeup:
            skipped.append(f"{task_id}: no local writeup")
            continue

        files = metadata.get("files", [])
        if not isinstance(files, list):
            files = []
        rows.append({
            "task_name": metadata.get("name") or task_id,
            "task_tag": metadata.get("category") or "misc",
            "task_points": str(metadata.get("points", metadata.get("score", "unknown"))),
            "task_description": metadata.get("description", ""),
            "solution": solution,
            "task_files": files,
            "server_description": "Task source: %s. Runtime is simulated from the audited writeup." % metadata.get("source", "ctf-dojo"),
            "writeup_path": writeup_path or str(task_dir.resolve()),
            "task_writeup": writeup or None,
            "source": metadata.get("source", "ctf-dojo"),
            "source_task_dir": str(task_dir.resolve()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Exported {len(rows)} validated tasks to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} tasks; first reasons:")
        print("\n".join(skipped[:20]))


if __name__ == "__main__":
    main()
