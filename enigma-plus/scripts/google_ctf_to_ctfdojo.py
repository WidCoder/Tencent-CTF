#!/usr/bin/env python3
"""Convert Google CTF source trees to verifier-aware CTF-Dojo tasks."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

try:
    from .google_ctf_common import (
        canonical_category, copy_tree_contents, copy_verification_files,
        find_named, infer_category, infer_event, is_challenge_dir,
        iter_year_dirs, read_description, extract_plaintext_flag, runtime_metadata, safe_name, setup_logging,
        should_skip_path, stable_task_id, verification_metadata, write_json, is_private_verifier_name,
    )
except ImportError:
    from google_ctf_common import (
        canonical_category, copy_tree_contents, copy_verification_files,
        find_named, infer_category, infer_event, is_challenge_dir,
        iter_year_dirs, read_description, extract_plaintext_flag, runtime_metadata, safe_name, setup_logging,
        should_skip_path, stable_task_id, verification_metadata, write_json, is_private_verifier_name,
    )


def iter_challenges(root: Path, year: str | None, category: str | None):
    wanted = canonical_category(category) if category else None
    for year_dir in iter_year_dirs(root, year):
        candidate_paths = []
        for path in sorted(p for p in year_dir.rglob("*") if p.is_dir() and not should_skip_path(p, year_dir)):
            inferred = infer_category(path, year_dir)
            if inferred and (not wanted or inferred == wanted) and is_challenge_dir(path, year_dir):
                candidate_paths.append((path, inferred))
        selected = {path for path, _ in candidate_paths}
        for path, inferred in candidate_paths:
            if any(parent in selected for parent in path.parents if parent != path):
                continue
            yield path, year_dir.name, inferred, infer_event(path, year_dir)


def convert_one(source: Path, root: Path, output: Path, year: str, category: str, event: str, link: bool) -> dict:
    relative = source.relative_to(root).as_posix()
    task_id = stable_task_id(f"google-ctf/{relative}")
    task = output / "challenges" / safe_name(f"google_ctf_{year}_{event}_{category}_{source.name}_{task_id[-10:]}")
    if (task / "challenge.json").exists():
        return {"challenge": source.name, "status": "skipped", "path": str(task), "task_id": task_id}
    task.mkdir(parents=True, exist_ok=True)
    dockerfiles = find_named(source, {"Dockerfile"})
    if dockerfiles:
        shutil.copy2(dockerfiles[0], task / "Dockerfile")
    compose_files = find_named(source, {"docker-compose.yml", "docker-compose.yaml"})
    if compose_files:
        shutil.copy2(compose_files[0], task / "docker-compose.yml")
    for child in sorted(source.iterdir(), key=lambda item: item.name.lower()):
        lowered = child.name.lower()
        if lowered in {"dockerfile", "docker-compose.yml", "docker-compose.yaml", "challenge.json", "solution", "writeup", "writeups"}:
            continue
        if child.is_dir() or child.is_symlink():
            copy_tree_contents(child, task / child.name, root.resolve())
        elif child.is_file() and lowered not in {
            "flag", "flag.txt", "flag.json", "flag.sha256", ".flag.sha256", "flag.sha256.txt",
            "readme", "readme.md", "readme.txt", "description.md", "description.txt",
        } and "flagcheck" not in lowered:
            shutil.copy2(child, task / child.name)
    copied = copy_verification_files(source, task)
    verification = verification_metadata(copied)
    description, sanitized = read_description(source)
    plaintext_flag = extract_plaintext_flag(source)
    if plaintext_flag and verification["method"] == "unknown":
        # The batch runner moves this value to a temporary host-side verifier
        # before creating the agent repository; it is never exposed to the
        # model.  Keeping it here preserves later verification for README-only
        # Google challenges.
        verification = dict(verification)
        verification["status"] = "eligible"
        verification["method"] = "plaintext"
        verification["files"] = []
    runtime = runtime_metadata(source)
    has_compose = (task / "docker-compose.yml").is_file()
    if has_compose and "box" not in runtime and "internal_port" not in runtime:
        match = re.match(r"^(?:https?://)?([^/:]+):(\d+)(?:/.*)?$", runtime.get("target_host", ""))
        if match:
            runtime["box"], runtime["internal_port"] = match.group(1), int(match.group(2))
    # Recreate a public, sanitized README before computing ``files`` instead
    # of copying the original answer-bearing README into the task repository.
    if description:
        (task / "README.md").write_text(description + "\n", encoding="utf-8")
    metadata = {
        "task_id": task_id,
        "name": source.name,
        "description": description,
        "description_sanitized": sanitized,
        "category": category,
        "category_friendly": category,
        "source": "google-ctf",
        "year": year,
        "event": event,
        "source_relative_path": relative,
        "dockerfile": "Dockerfile" if (task / "Dockerfile").is_file() else "",
        "dockerfile_path": "Dockerfile" if (task / "Dockerfile").is_file() else "",
        "runtime": "compose" if has_compose else "static",
        "files": sorted(
            path.relative_to(task).as_posix()
            for path in task.rglob("*")
            if path.is_file() and not is_private_verifier_name(path.name)
        ),
        "solution": "",
        "has_plaintext_solution": False,
        "verification": verification,
        "verification_method": verification["method"],
    }
    if plaintext_flag and verification["method"] == "plaintext":
        metadata["flag"] = plaintext_flag
    if has_compose:
        metadata["docker_compose"] = "docker-compose.yml"
    metadata.update(runtime)
    if verification["method"] == "sha256":
        metadata["sha256_file"] = verification["files"][0]
        metadata["sha256_flag_file"] = verification["files"][0]
    elif verification["method"] == "flagcheck":
        metadata["flag_check"] = verification["files"][0]
    write_json(task / "challenge.json", metadata)
    return {"challenge": source.name, "status": "converted", "path": str(task), "task_id": task_id, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ctfdojo_dataset"))
    parser.add_argument("--category")
    parser.add_argument("--year")
    parser.add_argument("--link", action="store_true", help="Retained for CLI compatibility; imports always copy safely.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    rows = []
    for source, year, category, event in iter_challenges(args.input.resolve(), args.year, args.category):
        rows.append(convert_one(source, args.input.resolve(), args.output, year, category, event, args.link))
    write_json(args.output / "conversion_report.json", rows)
    print(f"Converted {sum(row['status'] == 'converted' for row in rows)} challenges -> {args.output}")


if __name__ == "__main__":
    main()
