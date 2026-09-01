#!/usr/bin/env python3
"""Resumable EnIGMA+ trajectory generation for a CTF-Dojo challenge archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import shlex
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ENIGMA_ROOT = SCRIPT_DIR.parent
RUN_SCRIPT = ENIGMA_ROOT / "run.py"
STATE_FILE_NAME = "state.json"
SUMMARY_FILE_NAME = "summary.json"
QUALITY_REPORT_FILE_NAME = "dataset_quality_report.json"
LOGS_DIR_NAME = "logs"
TRAJECTORY_FILE_NAME = "trajectory.jsonl"
LOCK_FILE_NAME = ".ctfdojo_batch.lock"
AGENT_IMAGE = "sweagent/enigma:latest"
PROTECTED_CONTAINER_PREFIXES = ("cybergym", "proxy")


@dataclasses.dataclass(frozen=True)
class Challenge:
    challenge_json: str
    repo_path: str
    relative_path: str
    event: str
    name: str
    category: str
    task_id: str
    dockerfile_path: str
    compose_path: str


@dataclasses.dataclass(frozen=True)
class WorkerRequest:
    challenge: Challenge
    output_dir: str
    model_name: str
    image_name: str
    config_file: str
    step_limit: int
    task_timeout: int
    python_executable: str
    messages_api_url: str
    messages_api_key: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "unnamed"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def atomic_write_trajectory(path: Path, payload: dict[str, Any]) -> None:
    """Write one episode as one JSONL record, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def to_text(value: Any) -> str:
    """Convert subprocess and diagnostic values into safe log text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _text(value: Any) -> str:
    return to_text(value)


def _messages_tool_call(step: dict[str, Any]) -> dict[str, Any]:
    """Represent an EnIGMA shell action using the target dictionary schema."""
    return {"name": "Bash", "arguments": {"command": _text(step.get("action")).strip()}}


def _is_terminal_step(step: dict[str, Any]) -> bool:
    if step.get("terminal") is True:
        return True
    action = _text(step.get("action")).strip()
    return bool(re.fullmatch(r"(?:step_\d+_hit|task_timeout|runner_exception|environment_error)", action))


def trajectory_to_messages(payload: dict[str, Any], task: Challenge) -> dict[str, Any]:
    """Convert an internal EnIGMA trajectory into the public messages JSONL schema."""
    history = payload.get("history")
    history = history if isinstance(history, list) else []
    system = next((entry for entry in history if isinstance(entry, dict) and entry.get("role") == "system"), None)
    initial_user = next((entry for entry in history if isinstance(entry, dict) and entry.get("role") == "user"), None)
    messages: list[dict[str, Any]] = []

    if system is not None:
        messages.append({"role": "system", "content": _text(system.get("content"))})
    user_text = _text(initial_user.get("content")) if initial_user is not None else task.name
    messages.append({"role": "user", "content": user_text})

    trajectory = payload.get("trajectory")
    trajectory = trajectory if isinstance(trajectory, list) else []
    for raw_step in trajectory:
        if not isinstance(raw_step, dict):
            continue
        if _is_terminal_step(raw_step):
            messages.append({
                "role": "assistant",
                "content": _text(raw_step.get("terminal_reason") or raw_step.get("action")),
                "reasoning_content": _text(raw_step.get("thought")),
            })
            messages.append({"role": "tool", "content": _text(raw_step.get("observation"))})
            continue
        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": _text(raw_step.get("thought")),
            "tool_calls": [_messages_tool_call(raw_step)],
        })
        messages.append({"role": "tool", "content": _text(raw_step.get("observation"))})
    return {"id": task.task_id, "sample_type": "main", "messages": messages}


def write_public_trajectory(path: Path, payload: dict[str, Any], task: Challenge) -> None:
    atomic_write_trajectory(path, trajectory_to_messages(payload, task))

def status_payload(payload: dict[str, Any], exit_status: str) -> dict[str, Any]:
    """Restore status fields after reading a public messages artifact."""
    if "messages" not in payload:
        return payload
    return {
        "info": {"exit_status": exit_status},
        "trajectory": [message for message in payload.get("messages", [])
                        if isinstance(message, dict) and message.get("role") == "tool"],
    }

def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def category_from(metadata: dict[str, Any]) -> str:
    for key in ("category", "category_friendly"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    values = metadata.get("categories")
    if isinstance(values, list) and values and isinstance(values[0], str) and values[0].strip():
        return values[0].strip()
    return "unknown"


CHECKER_FILE_NAMES = {"checker.py", "verify.py", "checker.sh", "verify.sh"}
HASH_FILE_NAMES = {"flag.sha256", ".flag.sha256", "flag.sha256.txt"}
VERIFIER_COMMAND_NAMES = {"python", "python2", "python3", "bash", "sh", "env"}
VERIFIER_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", ".bzr", "_darcs", "solution", "writeup", "writeups",
}


def _is_private_verifier_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in HASH_FILE_NAMES
        or lowered in CHECKER_FILE_NAMES
        or "flagcheck" in lowered
    )


def _metadata_verifier_values(metadata: dict[str, Any]) -> list[object]:
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


def _metadata_verifier_paths(root: Path, metadata: dict[str, Any]) -> list[Path]:
    """Resolve declared verifier paths while ignoring command interpreters."""
    paths: list[Path] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            visit(value.get("path") or value.get("file") or value.get("command"))
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.strip():
            try:
                tokens = shlex.split(value)
            except ValueError:
                tokens = [value]
            for token in tokens:
                if Path(token).name.lower() in VERIFIER_COMMAND_NAMES:
                    continue
                candidate = Path(token)
                resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
                if not candidate.is_absolute():
                    try:
                        resolved.relative_to(root.resolve())
                    except ValueError:
                        continue
                if resolved.is_file() and resolved not in paths:
                    paths.append(resolved)

    for value in _metadata_verifier_values(metadata):
        visit(value)
    return paths


def _verification_candidates(repo_path: Path, metadata: dict[str, Any]) -> tuple[list[Path], set[Path]]:
    explicit = set(_metadata_verifier_paths(repo_path, metadata))
    discovered = {
        path for path in repo_path.rglob("*")
        if path.is_file()
        and _is_private_verifier_name(path.name)
        and not ({part.lower() for part in path.relative_to(repo_path.resolve()).parts[:-1]} & VERIFIER_EXCLUDED_DIRS)
    }
    return sorted(explicit | discovered, key=str), explicit


def _is_hash_verifier(path: Path, metadata: dict[str, Any]) -> bool:
    method = metadata.get("verification_method")
    verification = metadata.get("verification")
    if isinstance(verification, dict):
        method = verification.get("method", method)
    if path.name.lower() in HASH_FILE_NAMES or path.suffix.lower() == ".sha256":
        return True
    if method != "sha256":
        return False
    try:
        return bool(re.search(r"\b[0-9a-fA-F]{64}\b", path.read_text(encoding="utf-8", errors="replace")))
    except (OSError, UnicodeError):
        return False


def has_verification_evidence(repo_path: Path, metadata: dict[str, Any]) -> bool:
    # A plaintext flag is accepted only as host-side verification evidence.
    # ``stage_challenge`` removes it from the agent repository before launch.
    plaintext_flag = metadata.get("flag")
    verification = metadata.get("verification")
    method = metadata.get("verification_method")
    if isinstance(verification, dict):
        method = verification.get("method", method)
    if isinstance(plaintext_flag, str) and plaintext_flag.strip():
        return True
    if isinstance(verification, dict) and verification.get("method") == "plaintext":
        return bool(isinstance(plaintext_flag, str) and plaintext_flag.strip())
    candidates, explicit = _verification_candidates(repo_path, metadata)
    for path in candidates:
        if not path.is_file():
            continue
        if _is_hash_verifier(path, metadata):
            try:
                if re.search(r"\b[0-9a-fA-F]{64}\b", path.read_text(encoding="utf-8", errors="replace")):
                    return True
            except OSError:
                continue
        elif method != "sha256" and (_is_private_verifier_name(path.name) or path in explicit):
            if path.stat().st_size > 0:
                return True
    return False

def discover_challenges(dataset_root: Path, category: str | None, require_dockerfile: bool, require_verification: bool = False) -> tuple[list[Challenge], list[str]]:
    selected_category = category.lower() if category else None
    challenges: list[Challenge] = []
    warnings: list[str] = []
    for challenge_json in sorted(dataset_root.rglob("challenge.json")):
        try:
            metadata = read_json(challenge_json)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"invalid challenge JSON skipped: {challenge_json}: {error}")
            continue
        repo_path = challenge_json.parent
        if require_verification and not has_verification_evidence(repo_path, metadata):
            warnings.append(f"missing verifier evidence skipped: {repo_path}")
            continue
        relative_path = repo_path.relative_to(dataset_root)
        challenge_category = category_from(metadata)
        if selected_category and challenge_category.lower() != selected_category:
            continue
        dockerfile = repo_path / "Dockerfile"
        if require_dockerfile and not dockerfile.is_file():
            warnings.append(f"missing Dockerfile skipped: {repo_path}")
            continue
        name = metadata.get("name")
        if not isinstance(name, str) or not name.strip():
            name = repo_path.name
        stable_path = relative_path.as_posix()
        metadata_task_id = metadata.get("task_id")
        task_id = safe_component(metadata_task_id) if isinstance(metadata_task_id, str) and metadata_task_id.strip() else f"{safe_component(stable_path.replace('/', '__'))}-{hashlib.sha1(stable_path.encode()).hexdigest()[:10]}"
        metadata_event = metadata.get("event")
        event = metadata_event.strip() if isinstance(metadata_event, str) and metadata_event.strip() else (relative_path.parts[0] if relative_path.parts else "unknown")
        compose_value = metadata.get("docker_compose")
        compose = repo_path / compose_value if isinstance(compose_value, str) and compose_value else repo_path / "docker-compose.yml"
        if not compose.is_file():
            compose = repo_path / "docker-compose.yaml"
        challenges.append(Challenge(
            challenge_json=str(challenge_json.resolve()), repo_path=str(repo_path.resolve()),
            relative_path=stable_path, event=event,
            name=name.strip(), category=challenge_category, task_id=task_id,
            dockerfile_path=str(dockerfile.resolve()) if dockerfile.is_file() else "",
            compose_path=str(compose.resolve()) if compose.is_file() else "",
        ))
    return challenges, warnings


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 2, "created_at": utc_now(), "tasks": {}}
    payload = read_json(path)
    if not isinstance(payload.get("tasks"), dict):
        raise RuntimeError(f"invalid state file: {path}")
    return payload


def trajectory_path(output_dir: Path, task: Challenge) -> Path:
    return output_dir / task.task_id / TRAJECTORY_FILE_NAME


def valid_trajectory(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return isinstance(read_trajectory(path), dict)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def read_trajectory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trajectory must contain a JSON object")
    return payload


def flag_status_from_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    """Use verifier-owned fields only; never infer success from model prose."""
    info = payload.get("info")
    info = info if isinstance(info, dict) else {}
    verified = info.get("flag_verified")
    submitted = info.get("flag_submitted")
    evidence = info.get("flag_verification_evidence")
    if isinstance(verified, bool):
        return {
            "flag_submitted": bool(submitted) or verified,
            "flag_verified": verified,
            "flag_verification_evidence": evidence if isinstance(evidence, str) else (
                "verifier accepted submission" if verified else "verifier rejected submission"
            ),
        }
    if info.get("submission") is not None or str(info.get("exit_status", "")).startswith("submitted"):
        return {"flag_submitted": True, "flag_verified": None, "flag_verification_evidence": "missing structured verifier result"}
    return {"flag_submitted": False, "flag_verified": None, "flag_verification_evidence": "not_attempted"}


def episode_status_from_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    """Map implementation exit statuses to stable dataset episode statuses."""
    info = payload.get("info")
    info = info if isinstance(info, dict) else {}
    trajectory = payload.get("trajectory")
    exit_status = info.get("exit_status")
    if not isinstance(exit_status, str) or not exit_status:
        return {"episode_status": "running", "episode_complete": False}
    if isinstance(exit_status, str) and exit_status.startswith("submitted"):
        status = "completed"
    elif exit_status in {"task_timeout", "timeout", "model_timeout"}:
        status = "timeout"
    elif re.fullmatch(r"step_\d+_hit", exit_status):
        status = "step_limit"
    elif exit_status == "exit_format":
        status = "format_error"
    elif exit_status in {"ctf_server_unavailable", "ctf_server_crashed"}:
        return {"episode_status": exit_status, "episode_complete": False}
    elif any(token in exit_status for token in ("environment", "container", "server_crashed", "unavailable", "docker")):
        status = "environment_error"
    else:
        status = "early_exit"
    return {"episode_status": status, "episode_complete": status == "completed" or (status != "running" and isinstance(trajectory, list))}

def outcome_status_from(flags: dict[str, Any], episode: dict[str, Any], trajectory_generated: bool) -> str:
    if flags.get("flag_verified") is True and episode.get("episode_status") == "completed":
        return "solved"
    if trajectory_generated and episode.get("episode_status") != "running":
        return "unsolved"
    return "unknown"
def enrich_completed_records(state: dict[str, Any], output_dir: Path, tasks: list[Challenge]) -> bool:
    """Backfill reporting fields for completed legacy trajectories without rerunning."""
    changed = False
    for task in tasks:
        record = state["tasks"].get(task.relative_path)
        path = trajectory_path(output_dir, task)
        if not isinstance(record, dict) or not valid_trajectory(path):
            continue
        try:
            payload = read_trajectory(path)
            if "messages" in payload and record.get("episode_status") not in (None, "running"):
                continue
            flags = flag_status_from_trajectory(payload)
            episode = episode_status_from_trajectory(payload)
            values = {"run_status": "generated", "trajectory_generated": True, **episode, **flags, "outcome_status": outcome_status_from(flags, episode, True)}
        except (OSError, ValueError, json.JSONDecodeError):
            values = {"run_status": "failed", "trajectory_generated": True, "episode_status": "running", "episode_complete": False, "flag_submitted": False, "flag_verified": None, "flag_verification_evidence": "not_attempted", "outcome_status": "unknown"}
        for key, value in values.items():
            if key not in record or record.get(key) != value:
                record[key] = value
                changed = True
    return changed


def completed(state: dict[str, Any], output_dir: Path, task: Challenge) -> bool:
    record = state["tasks"].get(task.relative_path, {})
    return (
        record.get("run_status") == "generated"
        and record.get("episode_complete") is True
        and valid_trajectory(trajectory_path(output_dir, task))
    )

def task_record(task: Challenge, **extra: Any) -> dict[str, Any]:
    record = {
        "task": task.name, "task_id": task.task_id, "relative_path": task.relative_path,
        "event": task.event, "category": task.category, "challenge_json": task.challenge_json,
        "repo_path": task.repo_path, "dockerfile_path": task.dockerfile_path,
        "compose_path": task.compose_path, "updated_at": utc_now(),
    }
    record.update(extra)
    return record


def run_checked(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)


PRIVATE_VERIFIER_NAMES = HASH_FILE_NAMES


def _is_private_verifier(path: Path) -> bool:
    return _is_private_verifier_name(path.name)


def stage_challenge(source_dir: Path, workspace: Path) -> Path:
    staged_repo = workspace / "repo"
    source_metadata: dict[str, Any] = {}
    source_challenge_path = source_dir / "challenge.json"
    if source_challenge_path.is_file():
        source_metadata = read_json(source_challenge_path)
    declared_verifier_paths = _metadata_verifier_paths(source_dir, source_metadata)
    declared_verifier_relatives = {
        path.relative_to(source_dir.resolve()).as_posix().lower()
        for path in declared_verifier_paths
        if path.is_file()
    }

    def ignore_private(_directory: str, names: list[str]) -> set[str]:
        excluded = {"solution", "writeup", "writeups", "flag", "flag.txt", "flag.json"}
        vcs_metadata = {".git", ".hg", ".svn", ".bzr", "_darcs"}
        ignored: set[str] = set()
        for name in names:
            lowered = name.lower()
            if (
                lowered in excluded
                or lowered in vcs_metadata
                or lowered in {"readme", "readme.md", "readme.txt", "description.md", "description.txt"}
                or _is_private_verifier_name(name)
            ):
                ignored.add(name)
                continue
            candidate = (Path(_directory) / name).resolve()
            try:
                relative = candidate.relative_to(source_dir.resolve()).as_posix().lower()
            except ValueError:
                continue
            if relative in declared_verifier_relatives:
                ignored.add(name)
        return ignored
    shutil.copytree(source_dir, staged_repo, ignore=ignore_private)
    challenge_path = staged_repo / "challenge.json"
    if challenge_path.is_file():
        try:
            metadata = read_json(challenge_path)
            description = metadata.get("description")
            detected_description_flags: set[str] = set()
            if isinstance(description, str):
                detected_description_flags.update(re.findall(
                    r"(?i)\b(?:ctf|flag|pwn\.college|gc)\{[^\r\n}]{2,256}\}",
                    description,
                ))
                # Existing converted datasets may predate the README sanitizer.
                # Remove explicit answer lines and flag-shaped literals before
                # the challenge JSON is committed into the agent repository.
                description = re.sub(
                    r"(?im)^\s*(?:[#;>*-]\s*)?(?:the\s+)?flag\s*(?:is|=|:)\s*.+$\n?",
                    "",
                    description,
                )
                description = re.sub(
                    r"(?i)\b(?:ctf|flag|pwn\.college|gc)\{[^\r\n}]{2,256}\}",
                    "[REDACTED_FLAG]",
                    description,
                )
                metadata["description"] = description.strip()
            verification = metadata.get("verification") if isinstance(metadata.get("verification"), dict) else {}
            if (
                "flag" not in metadata
                and len(detected_description_flags) == 1
                and verification.get("method", "unknown") == "unknown"
            ):
                metadata["flag"] = next(iter(detected_description_flags))
                verification = dict(verification)
                verification.update({"status": "eligible", "method": "plaintext", "files": []})
            private_files = [
                path for path in source_dir.rglob("*")
                if path.is_file()
                and (
                    _is_private_verifier(path)
                    or path.resolve() in {item.resolve() for item in declared_verifier_paths}
                )
                and not ({part.lower() for part in path.relative_to(source_dir.resolve()).parts[:-1]} & VERIFIER_EXCLUDED_DIRS)
            ]
            # Keep verifier artifacts on the host only.  The temporary root is
            # removed after this task, and is never copied into the agent repo.
            verifier_root = workspace / "verifier"
            host_files: list[str] = []
            plaintext_flag = metadata.get("flag")
            if isinstance(plaintext_flag, str) and plaintext_flag and plaintext_flag != "pwn.college{...}":
                secret_path = verifier_root / "flag.value"
                secret_path.parent.mkdir(parents=True, exist_ok=True)
                secret_path.write_text(plaintext_flag.rstrip("\r\n") + "\n", encoding="utf-8")
                host_files.append(str(secret_path.resolve()))
                verification = dict(verification)
                verification["plaintext_file"] = str(secret_path.resolve())
                flag_match = re.fullmatch(r"([A-Za-z0-9_.-]+)\{.*\}", plaintext_flag.strip())
                if flag_match:
                    metadata["flag_format"] = f"{flag_match.group(1)}{{...}}"
                metadata.pop("flag", None)
            for path in private_files:
                relative = path.relative_to(source_dir)
                destination = verifier_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                host_files.append(str(destination.resolve()))
            if private_files:
                verification = dict(verification)
                if verification.get("method") == "flagcheck":
                    verification["method"] = "checker"
                verification["files"] = []
                verification["host_files"] = host_files
                verification["host_root"] = str(verifier_root.resolve())
                metadata["verification"] = verification
                hash_present = any(_is_hash_verifier(path, metadata) for path in private_files)
                checker_path = next(
                    (path for path in private_files if not _is_hash_verifier(path, metadata)),
                    None,
                )
                if checker_path is not None:
                    verification.update({"status": "eligible", "method": "checker"})
                    metadata["flag_check"] = checker_path.relative_to(source_dir).as_posix()
                elif hash_present:
                    verification.update({"status": "eligible", "method": "sha256"})
                for key in ("sha256_file", "sha256_flag_file", "flag_sha256_file", "hash_file"):
                    metadata.pop(key, None)
            elif plaintext_flag:
                metadata["verification"] = verification
                metadata.pop("flag", None)

            # challenge.files is consumed by SWEEnv when copying the
            # repository into the agent container. Staging intentionally
            # removes README/solution/flag/verifier files, so keep the
            # manifest synchronized with the staged repository. This also
            # covers --include_unverified tasks.
            files = metadata.get("files")
            if isinstance(files, list):
                visible_files: list[str] = []
                for value in files:
                    if not isinstance(value, str) or not value:
                        continue
                    relative = Path(value)
                    if relative.is_absolute() or _is_private_verifier(relative):
                        continue
                    try:
                        candidate = (staged_repo / relative).resolve()
                        candidate.relative_to(staged_repo.resolve())
                    except (OSError, ValueError):
                        continue
                    if candidate.exists():
                        visible_files.append(value)
                metadata["files"] = visible_files

            # Persist sanitized metadata for every staged challenge.
            challenge_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"failed to stage verifier metadata for {source_dir}: {error}") from error
    forbidden_vcs = {".git", ".hg", ".svn", ".bzr", "_darcs"}
    staged_vcs = [
        path for path in staged_repo.rglob("*")
        if path.name.lower() in forbidden_vcs
    ]
    if staged_vcs:
        raise RuntimeError(
            "staged repository contains forbidden VCS metadata: "
            + ", ".join(str(path) for path in staged_vcs)
        )
    run_checked(["git", "init"], staged_repo)
    run_checked(["git", "add", "."], staged_repo)
    run_checked(["git", "-c", "user.name=EnIGMA Batch", "-c", "user.email=enigma-batch@example.invalid", "commit", "-m", "Stage CTF-Dojo challenge for EnIGMA+"], staged_repo)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=staged_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError(f"staged repository is dirty after initialization:\n{status}")
    return staged_repo


def find_trajectory(run_root: Path) -> Path | None:
    # run_root is unique per attempt; do not rely on filesystem timestamp precision.
    candidates = [path for path in run_root.rglob("*.traj") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
def save_failure_log(
    path: Path,
    task: Challenge,
    command: list[str],
    error: Any,
    stdout: Any = "",
    stderr: Any = "",
    trace: Any = "",
) -> bool:
    """Best-effort failure logging that can never replace the real outcome."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        command_text = " ".join(to_text(item) for item in command)
        payload = (
            "timestamp: " + to_text(utc_now())
            + "\nchallenge_json: " + to_text(task.challenge_json)
            + "\nrepo_path: " + to_text(task.repo_path)
            + "\ndockerfile_path: " + to_text(task.dockerfile_path or "(none)")
            + "\ncommand: " + command_text
            + "\n\nerror:\n" + to_text(error)
            + "\n\nstdout:\n" + to_text(stdout)
            + "\n\nstderr:\n" + to_text(stderr)
            + ("\n\ntraceback:\n" + to_text(trace) if trace else "")
        )
        path.write_text(payload, encoding="utf-8")
        return True
    except Exception:
        return False


def docker_agent_container_ids(image_name: str = AGENT_IMAGE) -> set[str]:
    """Return containers created from precisely the EnIGMA agent image."""
    result = subprocess.run(["docker", "ps", "-aq", "--filter", f"ancestor={image_name}"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker ps failed")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def docker_bridge_container_count() -> int:
    result = subprocess.run(["docker", "network", "inspect", "bridge", "--format", "{{json .Containers}}"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker network inspect failed")
    containers = json.loads(result.stdout.strip() or "{}")
    if not isinstance(containers, dict):
        raise RuntimeError("docker network inspect returned an invalid Containers value")
    return len(containers)


def docker_dynamic_network_count(manifest_root: Path | None = None) -> int:
    """Count dynamic networks, optionally restricted to this batch's manifests."""
    result = subprocess.run(
        ["docker", "network", "ls", "-q", "--filter", "name=ctfnet-"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker network ls failed")
    network_ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if manifest_root is None:
        return len(network_ids)
    owned_names: set[str] = set()
    for manifest_path in manifest_root.rglob("resource.json") if manifest_root.is_dir() else []:
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        network_name = to_text(manifest.get("network"))
        if network_name:
            owned_names.add(network_name)
    if not owned_names:
        return 0
    listed = subprocess.run(
        ["docker", "network", "ls", "-q", "--filter", "name=ctfnet-"],
        capture_output=True, text=True, check=False,
    )
    if listed.returncode:
        raise RuntimeError(listed.stderr.strip() or listed.stdout.strip() or "docker network ls failed")
    # Resolve names exactly; Docker's name filter is substring-based.
    details = subprocess.run(
        ["docker", "network", "ls", "--format", "{{.ID}}\t{{.Name}}"],
        capture_output=True, text=True, check=False,
    )
    if details.returncode:
        raise RuntimeError(details.stderr.strip() or details.stdout.strip() or "docker network ls failed")
    return sum(
        1 for line in details.stdout.splitlines()
        if len(line.split("\t", 1)) == 2 and line.split("\t", 1)[0] in network_ids
        and line.split("\t", 1)[1] in owned_names
    )


def docker_owned_bridge_container_count(manifest_root: Path | None = None) -> int:
    """Count live bridge containers described by this batch's manifests."""
    names: set[str] = set()
    projects: set[str] = set()
    if manifest_root and manifest_root.is_dir():
        for manifest_path in manifest_root.rglob("resource.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict):
                agent_container = to_text(manifest.get("agent_container"))
                project = to_text(manifest.get("compose_project"))
                if agent_container:
                    names.add(agent_container)
                if project and not _protected_resource_name(project):
                    projects.add(project)
    else:
        names.add("batch-")

    ids: set[str] = set()
    for name in names:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={name}", "--format", "{{.ID}}"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker ps failed")
        ids.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    for project in projects:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}}"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker ps failed")
        ids.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not ids:
        return 0
    bridge = subprocess.run(
        ["docker", "network", "inspect", "bridge", "--format", "{{json .Containers}}"],
        capture_output=True, text=True, check=False,
    )
    if bridge.returncode:
        raise RuntimeError(bridge.stderr.strip() or bridge.stdout.strip() or "docker network inspect failed")
    containers = json.loads(bridge.stdout.strip() or "{}")
    return sum(container_id in containers for container_id in ids) if isinstance(containers, dict) else 0


def _protected_resource_name(value: str) -> bool:
    normalized = value.strip().lstrip("/").lower()
    return any(normalized == prefix or normalized.startswith(prefix + "-") for prefix in PROTECTED_CONTAINER_PREFIXES)


def cleanup_agent_containers(
    baseline: set[str] | None,
    image_name: str = AGENT_IMAGE,
    resource_token: str | None = None,
    resource_container_name: str | None = None,
) -> dict[str, Any]:
    """Delete only this worker's agent containers.

    A shared image baseline is not sufficient under parallel execution: a
    sibling worker can create a container after our baseline and be mistaken
    for ours. New batch workers embed ``resource_token`` in the container
    name, so cleanup is scoped to that token. The baseline fallback preserves
    compatibility for callers outside the batch runner.
    """
    cleanup: dict[str, Any] = {"attempted": baseline is not None, "removed_container_ids": [], "failed_container_ids": []}
    if (
        _protected_resource_name(image_name)
        or _protected_resource_name(resource_token or "")
        or _protected_resource_name(resource_container_name or "")
    ):
        cleanup["attempted"] = False
        cleanup["error"] = "refusing to clean protected infrastructure resource"
        return cleanup
    if baseline is None:
        cleanup["error"] = "agent container baseline unavailable"
        return cleanup
    try:
        if resource_token:
            name_filter = resource_container_name or resource_token
            result = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"ancestor={image_name}", "--filter", f"name={name_filter}"],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "docker ps failed")
            created_ids = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        else:
            created_ids = sorted(docker_agent_container_ids(image_name) - baseline)
    except Exception as error:
        cleanup["error"] = str(error)
        return cleanup
    for container_id in created_ids:
        try:
            result = subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, text=True, check=False)
            key = "removed_container_ids" if result.returncode == 0 else "failed_container_ids"
            cleanup[key].append(container_id)
        except OSError:
            cleanup["failed_container_ids"].append(container_id)
    return cleanup


def cleanup_attempt_resources(resource_token: str, resource_manifest: Path | None = None) -> dict[str, Any]:
    """Remove Docker resources owned by one batch attempt.

    The outer timeout can terminate ``run.py`` before its normal ``close``
    path runs.  Dynamic compose services and their ``ctfnet-*`` network then
    remain behind and eventually exhaust Docker's address pools.  Every batch
    attempt gets a random token which is embedded in the generated resource
    names; only matching resources are removed here, so parallel workers are
    not affected.
    """
    result: dict[str, Any] = {
        "resource_token": resource_token,
        "removed_containers": [],
        "failed_containers": [],
        "removed_networks": [],
        "failed_networks": [],
        "remaining_containers": [],
        "remaining_networks": [],
        "removed_volumes": [],
        "failed_volumes": [],
        "remaining_volumes": [],
        "removal_errors": {},
    }
    if not resource_token:
        result["error"] = "empty resource token"
        return result
    if _protected_resource_name(resource_token):
        result["error"] = "refusing to clean protected infrastructure resource"
        return result
    if not re.fullmatch(r"batch-[a-z0-9]+", resource_token):
        result["error"] = "refusing to clean a non-batch resource token"
        return result

    manifest: dict[str, Any] = {}
    if resource_manifest and resource_manifest.is_file():
        try:
            loaded = json.loads(resource_manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, ValueError, json.JSONDecodeError) as error:
            result["manifest_error"] = str(error)

    def list_ids(command: list[str]) -> list[str]:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "docker list failed")
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def disconnect_network_endpoints(network_id: str) -> None:
        """Detach any endpoint that prevents removal of an owned network."""
        inspected = subprocess.run(
            ["docker", "network", "inspect", network_id, "--format", "{{json .Containers}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspected.returncode:
            raise RuntimeError(inspected.stderr.strip() or inspected.stdout.strip() or "docker network inspect failed")
        try:
            containers = json.loads(to_text(inspected.stdout).strip() or "{}")
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid network endpoint data for {network_id}: {error}") from error
        if not isinstance(containers, dict):
            return
        for container_id, details in containers.items():
            container_name = details.get("Name") if isinstance(details, dict) else ""
            target = to_text(container_id) or to_text(container_name)
            if not target:
                continue
            disconnected = subprocess.run(
                ["docker", "network", "disconnect", "-f", network_id, target],
                capture_output=True,
                text=True,
                check=False,
            )
            detail = f"{to_text(disconnected.stdout)}\n{to_text(disconnected.stderr)}".lower()
            if disconnected.returncode and not any(
                marker in detail for marker in ("not connected", "no such", "not found")
            ):
                raise RuntimeError(
                    f"failed to disconnect endpoint {target} from {network_id}: "
                    f"{to_text(disconnected.stderr).strip() or to_text(disconnected.stdout).strip()}"
                )

    def remove_with_retry(
        command: list[str],
        removed_key: str,
        failed_key: str,
        resource_id: str,
        disconnect_active_endpoints: bool = False,
    ) -> None:
        completed = None
        for attempt in range(30):
            completed = subprocess.run(command + [resource_id], capture_output=True, text=True, check=False)
            stdout = to_text(completed.stdout)
            stderr = to_text(completed.stderr)
            detail = f"{stdout}\n{stderr}".lower()
            if completed.returncode == 0 or "no such" in detail or "not found" in detail:
                result[removed_key].append(resource_id)
                return
            if disconnect_active_endpoints and "active endpoints" in detail:
                try:
                    disconnect_network_endpoints(resource_id)
                except RuntimeError as error:
                    result.setdefault("endpoint_cleanup_errors", {})[resource_id] = str(error)
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
                continue
            if "409" not in detail and "removal already in progress" not in detail:
                break
            time.sleep(min(2.0, 0.25 * (attempt + 1)))
        if completed is not None:
            result["removal_errors"][resource_id] = (
                f"returncode={completed.returncode}; stdout={to_text(completed.stdout)}; "
                f"stderr={to_text(completed.stderr)}"
            )
        result[failed_key].append(resource_id)

    def run_with_retry(command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        for attempt in range(30):
            detail = f"{to_text(completed.stdout)}\n{to_text(completed.stderr)}".lower()
            if completed.returncode == 0 or (
                "409" not in detail and "removal already in progress" not in detail
            ):
                return completed
            time.sleep(min(2.0, 0.25 * (attempt + 1)))
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return completed

    labels = manifest.get("labels") if isinstance(manifest.get("labels"), dict) else {}
    project = to_text(manifest.get("compose_project")) or to_text(labels.get("com.docker.compose.project"))
    compose_file = to_text(manifest.get("compose_file"))
    if project and not _protected_resource_name(project):
        compose_command = ["docker", "compose"]
        if compose_file and Path(compose_file).is_file():
            compose_command.extend(["-f", compose_file])
        compose_command.extend(["-p", project, "down", "--remove-orphans"])
        compose_result = run_with_retry(compose_command)
        if compose_result.returncode:
            result["compose_down_error"] = compose_result.stderr.strip() or compose_result.stdout.strip()

    try:
        container_filters = ["docker", "ps", "-aq", "--filter", f"name={resource_token}"]
        agent_container = to_text(manifest.get("agent_container"))
        container_ids = list_ids(container_filters)
        # Older runs truncated the token in the container name. Use the
        # manifest's exact name as a compatibility path for those resources.
        if agent_container and not _protected_resource_name(agent_container):
            container_ids.extend(list_ids([
                "docker", "ps", "-aq", "--filter", f"name={agent_container}"
            ]))
        if project:
            container_ids.extend(list_ids([
                "docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={project}"
            ]))
        container_ids = sorted(set(container_ids))
    except Exception as error:
        result["container_list_error"] = str(error)
        container_ids = []
    for container_id in container_ids:
        try:
            remove_with_retry(["docker", "rm", "-f"], "removed_containers", "failed_containers", container_id)
        except OSError:
            result["failed_containers"].append(container_id)

    try:
        network_ids = list_ids(["docker", "network", "ls", "-q", "--filter", f"name=ctfnet-{resource_token}"])
        manifest_network = to_text(manifest.get("network"))
        if manifest_network:
            network_ids.extend(list_ids(["docker", "network", "ls", "-q", "--filter", f"name={manifest_network}"]))
        if project:
            network_ids.extend(list_ids([
                "docker", "network", "ls", "-q", "--filter",
                f"label=com.docker.compose.project={project}",
            ]))
        network_ids = sorted(set(network_ids))
    except Exception as error:
        result["network_list_error"] = str(error)
        network_ids = []
    for network_id in network_ids:
        try:
            remove_with_retry(
                ["docker", "network", "rm"],
                "removed_networks",
                "failed_networks",
                network_id,
                disconnect_active_endpoints=True,
            )
        except OSError:
            result["failed_networks"].append(network_id)

    # Volumes are removed last, after Compose, containers, and networks.
    try:
        volume_ids = list_ids([
            "docker", "volume", "ls", "-q", "--filter",
            f"label=com.docker.compose.project={project}",
        ]) if project else []
    except Exception as error:
        result["volume_list_error"] = str(error)
        volume_ids = []
    for volume_id in sorted(set(volume_ids)):
        try:
            remove_with_retry(["docker", "volume", "rm"], "removed_volumes", "failed_volumes", volume_id)
        except OSError:
            result["failed_volumes"].append(volume_id)

    def remaining(command: list[str], key: str) -> None:
        try:
            result[key] = list_ids(command)
        except Exception as error:
            result[f"{key[:-1]}_check_error"] = str(error)

    try:
        remaining_container_ids = list_ids([
            "docker", "ps", "-aq", "--filter", f"name={resource_token}"
        ])
        agent_container = to_text(manifest.get("agent_container"))
        if agent_container and not _protected_resource_name(agent_container):
            remaining_container_ids.extend(list_ids([
                "docker", "ps", "-aq", "--filter", f"name={agent_container}"
            ]))
        if project:
            remaining_container_ids.extend(list_ids([
                "docker", "ps", "-aq", "--filter",
                f"label=com.docker.compose.project={project}",
            ]))
        result["remaining_containers"] = sorted(set(remaining_container_ids))
    except Exception as error:
        result["remaining_container_check_error"] = str(error)
    try:
        remaining_network_ids = list_ids([
            "docker", "network", "ls", "-q", "--filter", f"name=ctfnet-{resource_token}"
        ])
        manifest_network = to_text(manifest.get("network"))
        if manifest_network:
            remaining_network_ids.extend(list_ids([
                "docker", "network", "ls", "-q", "--filter", f"name={manifest_network}"
            ]))
        if project:
            remaining_network_ids.extend(list_ids([
                "docker", "network", "ls", "-q", "--filter",
                f"label=com.docker.compose.project={project}",
            ]))
        result["remaining_networks"] = sorted(set(remaining_network_ids))
    except Exception as error:
        result["remaining_network_check_error"] = str(error)
    if project:
        remaining(["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"], "remaining_volumes")
    return result


def write_attempt_resource_manifest(path: Path, resource_token: str) -> bool:
    """Publish a cleanup-safe seed before any worker startup can fail."""
    try:
        atomic_write_json(path, {
            "attempt_token": resource_token,
            "agent_container": "",
            "compose_project": "",
            "compose_file": "",
            "network": "",
            "labels": {},
        })
        return True
    except OSError:
        return False


def persist_attempt_compose_file(resource_manifest: Path) -> None:
    """Keep a cleanup-only Compose file after the staging temp directory is gone."""
    try:
        manifest = read_json(resource_manifest)
        compose_file = Path(to_text(manifest.get("compose_file")))
        if not compose_file.is_file():
            return
        persistent = resource_manifest.parent / "resource-compose.yml"
        shutil.copy2(compose_file, persistent)
        manifest["compose_file"] = str(persistent.resolve())
        atomic_write_json(resource_manifest, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        # Label/name cleanup remains available when the copy cannot be made.
        try:
            manifest = read_json(resource_manifest)
            manifest["compose_file_persist_error"] = to_text(error)
            atomic_write_json(resource_manifest, manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def cleanup_stale_attempt_resources(output_dir: Path) -> list[dict[str, Any]]:
    """Reclaim only batch-token resources left by an earlier interrupted run."""
    reports: list[dict[str, Any]] = []
    if not output_dir.is_dir():
        return reports
    for manifest_path in output_dir.rglob("resource.json"):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        token = to_text(manifest.get("attempt_token"))
        if re.fullmatch(r"batch-[a-z0-9]+", token):
            reports.append(cleanup_attempt_resources(token, manifest_path))
    return reports

def docker_network_exhausted(*output: str) -> bool:
    detail = "\n".join(output).lower()
    return "address pools" in detail and (
        "no available ipv4 addresses" in detail
        or "fully subnetted" in detail
        or "subnet" in detail
    )


def model_generation_timed_out(*output: str) -> bool:
    detail = "\n".join(output).lower()
    return any(
        marker in detail
        for marker in (
            "model generation exceeded",
            "model generation timed out",
            "timeouterror: model generation",
            "messages api request timed out",
            "read timed out",
            "error_type=model_timeout",
            '"error_type": "model_timeout"',
        )
    )


def docker_storage_exhausted(*output: str) -> bool:
    """Detect Docker/Buildx failures caused by a full disk or inode store."""
    detail = "\n".join(output).lower()
    return "no space left on device" in detail and any(
        token in detail for token in ("docker", "buildx", "compose", "/var/lib/docker", ".docker")
    )


def persist_terminal_trajectory(
    output_dir: Path, task: Challenge, *, exit_status: str, terminal_reason: str,
    error_type: str | None = None, source: Path | None = None,
) -> Path:
    """Persist a usable terminal trajectory even when the runner was interrupted."""
    destination = trajectory_path(output_dir, task)
    payload: dict[str, Any]
    try:
        payload = read_json(source) if source is not None else {}
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
        payload["trajectory"] = trajectory
    info = payload.get("info")
    info = dict(info) if isinstance(info, dict) else {}
    # A runner-level wrapper must not erase a more specific error already
    # persisted by run.py (for example AttributeError or an environment
    # startup traceback). Keep wrapper details under separate keys.
    original_exit_status = info.get("exit_status")
    original_reason = info.get("terminal_reason") or info.get("error_message")
    original_error_type = info.get("error_type")
    effective_exit_status = original_exit_status or exit_status
    # Promote a generic run.py exception when the outer evidence proves this
    # was an environment-start failure; retain the original status below.
    if exit_status == "environment_error" and original_exit_status == "runner_exception":
        effective_exit_status = exit_status
    if original_exit_status and original_exit_status != exit_status:
        info["wrapped_exit_status"] = exit_status
    if original_reason and original_reason != terminal_reason:
        info["wrapped_terminal_reason"] = terminal_reason
    if original_error_type and error_type and original_error_type != error_type:
        info["wrapped_error_type"] = error_type
    info.update({
        "exit_status": effective_exit_status,
        "terminal_reason": original_reason or terminal_reason,
        "error_type": original_error_type or error_type or "",
        "failure_category": error_type or info.get("failure_category", ""),
        "flag_verified": info.get("flag_verified") if isinstance(info.get("flag_verified"), bool) else None,
        "flag_submitted": bool(info.get("flag_submitted")),
        "episode_end_time": utc_now(),
        "total_steps": len(trajectory),
    })
    payload["trajectory_schema_version"] = 2
    info["artifact_type"] = "partial_trace" if source is not None else "synthetic_error"
    payload["info"] = info
    payload.setdefault("environment", "batch_runner")
    payload.setdefault("history", [])
    if not trajectory or not isinstance(trajectory[-1], dict) or not trajectory[-1].get("terminal"):
        trajectory.append({
            "step_id": len(trajectory) + 1, "thought": "", "action": exit_status,
            "tool_name": "unknown", "tool_args": "", "observation": terminal_reason,
            "return_code": None, "execution_time": 0.0, "error": error_type or "",
            "terminal": True, "response": "",
        })
        info["total_steps"] = len(trajectory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_trajectory(destination, payload)
    return destination


def run_one_task(request: WorkerRequest) -> dict[str, Any]:
    task = request.challenge
    output_dir = Path(request.output_dir)
    attempt_id = f"attempt-{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
    resource_token = f"batch-{uuid.uuid4().hex[:12]}"
    runner_output = output_dir / task.task_id / "enigma_output" / attempt_id
    resource_manifest = output_dir / task.task_id / "resource.json"
    write_attempt_resource_manifest(resource_manifest, resource_token)
    error_log = output_dir / LOGS_DIR_NAME / f"{task.task_id}.error.log"
    command: list[str] = []
    baseline: set[str] | None = None
    started_at = dt.datetime.now().timestamp()
    result: subprocess.CompletedProcess[str] | None = None
    timeout_error: subprocess.TimeoutExpired | None = None
    timeout_trace = ""
    try:
        with tempfile.TemporaryDirectory(prefix=f"enigma-{safe_component(task.name)}-") as temp_dir:
            staged_repo = stage_challenge(Path(task.repo_path), Path(temp_dir))
            command = [request.python_executable, str(RUN_SCRIPT), "--model_name", request.model_name,
                "--image_name", request.image_name, "--data_path", str(staged_repo / "challenge.json"),
                "--repo_path", str(staged_repo), "--config_file", request.config_file,
                "--per_instance_step_limit", str(request.step_limit), "--trajectory_path", str(runner_output),
                "--messages_api_url", request.messages_api_url, "--messages_api_key", request.messages_api_key]
            baseline = docker_agent_container_ids(request.image_name)
            try:
                run_env = os.environ.copy()
                # Keep the inner EnIGMA deadline below the outer subprocess
                # deadline. Callers can override this explicitly.
                # Keep the inner EnIGMA deadline below the outer subprocess
                # deadline.  The previous fixed 1800s default ignored a
                # larger --task_timeout value and could terminate long tasks
                # prematurely.
                if request.task_timeout:
                    inner_timeout = max(60, request.task_timeout - 60)
                    run_env["SWE_AGENT_TASK_TIMEOUT"] = str(inner_timeout)
                else:
                    run_env.setdefault("SWE_AGENT_TASK_TIMEOUT", "1800")
                # GLM reasoning can legitimately take several minutes on a
                # long CTF context. Keep a larger model-level budget than the
                # historical 300s default, while still allowing callers to
                # override it explicitly through the environment.
                run_env.setdefault("SWE_AGENT_MODEL_TIMEOUT", "600")
                # Keep the default gateway protocol compatible with the
                # official Cyber-Zero GLM path; native tool blocks are an
                # explicit experiment via SWE_AGENT_MESSAGES_NATIVE_TOOLS=1.
                run_env.setdefault("SWE_AGENT_MESSAGES_NATIVE_TOOLS", "0")
                run_env.setdefault("SWE_AGENT_FLAG_VERIFIER_IMAGE", request.image_name)
                run_env["ENIGMA_BATCH_ATTEMPT_ID"] = resource_token
                run_env["ENIGMA_BATCH_RESOURCE_MANIFEST"] = str(resource_manifest)
                result = subprocess.run(command, cwd=ENIGMA_ROOT, capture_output=True, text=True,
                                        check=False, timeout=request.task_timeout or None, env=run_env)
            except subprocess.TimeoutExpired as error:
                timeout_error = error
                timeout_trace = traceback.format_exc()
            persist_attempt_compose_file(resource_manifest)
        source = find_trajectory(runner_output)
        if timeout_error is not None:
            reason = f"batch task timeout after {request.task_timeout} seconds"
            destination = persist_terminal_trajectory(output_dir, task, exit_status="task_timeout",
                terminal_reason=reason, error_type="outer_timeout", source=source)
            save_failure_log(error_log, task, command, reason, timeout_error.stdout or "", timeout_error.stderr or "", timeout_trace)
            payload = status_payload(read_trajectory(destination), "task_timeout")
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            effective_info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source) if source else "", exit_status=effective_info.get("exit_status", "task_timeout"),
                error=effective_info.get("terminal_reason", reason), error_code=effective_info.get("error_type", "task_timeout"),
                 error_log=str(error_log), solved=False, steps=len(payload.get("trajectory", [])), **episode, **flags)
        elif result is None or result.returncode != 0:
            reason = f"run.py exited with code {result.returncode}" if result is not None else "run.py did not complete"
            source_detail = ""
            if source is not None:
                try:
                    source_payload = read_trajectory(source)
                    source_info = source_payload.get("info") if isinstance(source_payload.get("info"), dict) else {}
                    source_detail = "\n".join(
                        str(source_info.get(key, "")) for key in ("terminal_reason", "error_type", "traceback")
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    source_detail = ""
            if result and model_generation_timed_out(result.stdout, result.stderr, source_detail):
                error_type = "model_timeout"
            elif result and docker_storage_exhausted(result.stdout, result.stderr, source_detail):
                error_type = "docker_storage_exhausted"
            elif result and docker_network_exhausted(result.stdout, result.stderr, source_detail):
                error_type = "docker_network_address_exhausted"
            else:
                error_type = "runner_error"
            exit_status = "environment_error" if error_type.startswith("docker_") else ("model_timeout" if error_type == "model_timeout" else "runner_exception")
            destination = persist_terminal_trajectory(output_dir, task, exit_status=exit_status,
                terminal_reason=reason, error_type=error_type, source=source)
            save_failure_log(error_log, task, command, reason, result.stdout if result else "", result.stderr if result else "")
            payload = status_payload(read_trajectory(destination), exit_status)
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            effective_info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source) if source else "", exit_status=effective_info.get("exit_status", exit_status),
                error=effective_info.get("terminal_reason", reason), error_code=effective_info.get("error_type", error_type),
                failure_category=effective_info.get("failure_category", error_type),
                error_log=str(error_log), solved=False, steps=len(payload.get("trajectory", [])), **episode, **flags)
        elif source is None:
            reason = "run.py exited successfully but no trajectory file was produced"
            destination = persist_terminal_trajectory(output_dir, task, exit_status="runner_exception", terminal_reason=reason, error_type="missing_trajectory")
            save_failure_log(error_log, task, command, reason)
            payload = status_payload(read_trajectory(destination), "runner_exception")
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            record = task_record(task, run_status="failed", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                exit_status="runner_exception", error=reason, error_code="missing_trajectory", error_log=str(error_log), solved=False,
                steps=len(payload.get("trajectory", [])), **episode, **flags)
        else:
            payload = read_json(source)
            payload["trajectory_schema_version"] = 2
            payload.setdefault("info", {})["artifact_type"] = "agent_trace"
            destination = trajectory_path(output_dir, task)
            destination.parent.mkdir(parents=True, exist_ok=True)
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            record = task_record(task, run_status="generated", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source), exit_status=payload.get("info", {}).get("exit_status", ""),
                solved=flags["flag_verified"] is True, steps=len(payload.get("trajectory", [])), **episode, **flags)
        write_public_trajectory(destination, payload, task)
        record["outcome_status"] = outcome_status_from(flags, episode, True)
    except BaseException as error:
        reason = str(error)
        save_failure_log(error_log, task, command, reason, trace=traceback.format_exc())
        error_code = "interrupted" if isinstance(error, KeyboardInterrupt) else "task_setup_failed"
        destination = persist_terminal_trajectory(output_dir, task, exit_status="runner_exception", terminal_reason=reason, error_type=error_code)
        payload = status_payload(read_trajectory(destination), "runner_exception")
        flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
        record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
            exit_status="runner_exception", error=reason, error_code=error_code, error_log=str(error_log), solved=False,
            steps=len(payload.get("trajectory", [])), outcome_status=outcome_status_from(flags, episode, True), **episode, **flags)
        write_public_trajectory(destination, payload, task)
    finally:
        attempt_cleanup = cleanup_attempt_resources(resource_token, resource_manifest)
        # The attempt cleanup owns Compose resources and is intentionally first;
        # the image-baseline fallback only handles an agent missed by the manifest.
        resource_container_name = None
        try:
            manifest = read_json(resource_manifest)
            candidate = to_text(manifest.get("agent_container"))
            if candidate and not _protected_resource_name(candidate):
                resource_container_name = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        cleanup = cleanup_agent_containers(
            baseline,
            request.image_name,
            resource_token,
            resource_container_name=resource_container_name,
        )
        if 'record' in locals():
            record["attempt_resource_cleanup"] = attempt_cleanup
            record["resource_cleanup_clean"] = not (
                attempt_cleanup.get("remaining_containers")
                or attempt_cleanup.get("remaining_networks")
                or attempt_cleanup.get("remaining_volumes")
                or attempt_cleanup.get("failed_containers")
                or attempt_cleanup.get("failed_networks")
                or attempt_cleanup.get("failed_volumes")
            )
        if 'record' in locals():
            record["container_cleanup"] = cleanup
    return record


def trajectory_quality(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count episode-level quality signals without mixing them with step counts."""
    counts = {
        "solved_trajectories": 0, "unsolved_trajectories": 0, "trainable_trajectories": 0,
        "synthetic_error_trajectories": 0, "timeout": 0, "parser_error_episodes": 0,
        "environment_error": 0, "duplicate_action_steps": 0, "empty_observation_steps": 0,
        "invalid_terminal_episodes": 0, "legacy_schema_episodes": 0,
    }
    for record in records:
        outcome, status = record.get("outcome_status"), record.get("episode_status")
        if outcome == "solved": counts["solved_trajectories"] += 1
        if outcome == "unsolved": counts["unsolved_trajectories"] += 1
        if status == "timeout": counts["timeout"] += 1
        if status in {"environment_error", "ctf_server_unavailable", "ctf_server_crashed"}:
            counts["environment_error"] += 1
        path = record.get("trajectory")
        if not isinstance(path, str) or not valid_trajectory(Path(path)):
            continue
        try:
            payload = read_trajectory(Path(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        artifact_type = info.get("artifact_type", "legacy")
        if artifact_type == "synthetic_error": counts["synthetic_error_trajectories"] += 1
        if payload.get("trajectory_schema_version") != 2: counts["legacy_schema_episodes"] += 1
        steps = payload.get("trajectory") if isinstance(payload.get("trajectory"), list) else []
        terminal_steps = [step for step in steps if isinstance(step, dict) and step.get("terminal") is True]
        if len(terminal_steps) != 1: counts["invalid_terminal_episodes"] += 1
        if outcome == "solved" and artifact_type == "agent_trace" and len(terminal_steps) == 1:
            counts["trainable_trajectories"] += 1
        parser_error = status == "format_error"
        previous = None
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            normalized_action = " ".join(action.split()) if isinstance(action, str) else ""
            if normalized_action and normalized_action == previous:
                counts["duplicate_action_steps"] += 1
            if normalized_action:
                previous = normalized_action
            if not step.get("terminal") and (step.get("observation") is None or not str(step.get("observation")).strip()):
                counts["empty_observation_steps"] += 1
            parser_error = parser_error or step.get("error") == "parse_error"
        if parser_error: counts["parser_error_episodes"] += 1
    return counts
def reporting_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    generated = [record for record in records if record.get("trajectory_generated")]
    flag_submitted = sum(bool(record.get("flag_submitted")) for record in records)
    flag_verified = sum(record.get("flag_verified") is True for record in records)
    flag_unknown = sum(record.get("flag_verified") is None and bool(record.get("flag_submitted")) for record in records)
    flag_not_attempted = sum(not bool(record.get("flag_submitted")) and record.get("flag_verified") is not True for record in records)
    return {
        "generated_trajectories": len(generated),
        "trajectory_generated": len(generated),
        "verified_solved": sum(record.get("outcome_status") == "solved" for record in generated),
        "unsolved": sum(record.get("outcome_status") == "unsolved" for record in generated),
        "generation_failed": sum(record.get("run_status") in {"failed", "error"} for record in records),
        "environment_error": sum(
            record.get("episode_status") in {"environment_error", "ctf_server_unavailable", "ctf_server_crashed"}
            for record in records
        ),
        "flag_submitted": flag_submitted,
        "flag_verified": flag_verified,
        "flag_status_unknown": flag_unknown,
        "flag_not_attempted": flag_not_attempted,
    }


def build_summary(tasks: list[Challenge], state: dict[str, Any], skipped: int) -> dict[str, Any]:
    records = [state["tasks"].get(task.relative_path, {}) for task in tasks]
    counts = reporting_counts(records)
    categories: dict[str, dict[str, int]] = {}
    for task in tasks:
        record = state["tasks"].get(task.relative_path, {})
        category = task.category or "unknown"
        category_counts = categories.setdefault(category, {"trajectory_generated": 0, "flag_submitted": 0, "flag_verified": 0})
        if record.get("trajectory_generated"):
            category_counts["trajectory_generated"] += 1
        if record.get("flag_submitted"):
            category_counts["flag_submitted"] += 1
        if record.get("flag_verified") is True:
            category_counts["flag_verified"] += 1
    cleanup_removed = sum(
        len(record.get("container_cleanup", {}).get("removed_container_ids", []))
        for record in records
        if isinstance(record.get("container_cleanup"), dict)
    )
    attempt_cleanup_removed = sum(
        len(record.get("attempt_resource_cleanup", {}).get("removed_containers", []))
        + len(record.get("attempt_resource_cleanup", {}).get("removed_networks", []))
        for record in records
        if isinstance(record.get("attempt_resource_cleanup"), dict)
    )
    attempt_cleanup_failed = sum(
        len(record.get("attempt_resource_cleanup", {}).get("failed_containers", []))
        + len(record.get("attempt_resource_cleanup", {}).get("failed_networks", []))
        for record in records
        if isinstance(record.get("attempt_resource_cleanup"), dict)
    )
    cleanup_failed = sum(
        len(record.get("container_cleanup", {}).get("failed_container_ids", []))
        for record in records
        if isinstance(record.get("container_cleanup"), dict)
    )
    return {"generated_at": utc_now(), "total_tasks": len(tasks), "skipped": skipped,
            "pending": len(tasks) - sum(bool(record) for record in records), **counts,
            "container_cleanup_removed": cleanup_removed,
            "container_cleanup_failed": cleanup_failed,
            "attempt_resource_cleanup_removed": attempt_cleanup_removed,
            "attempt_resource_cleanup_failed": attempt_cleanup_failed,
            "categories": categories,
            "run_status": {key: sum(record.get("run_status") == key for record in records) for key in ("generated", "failed", "error")},
            "batch_errors": state.get("batch_errors", []),
            "docker_bridge_containers_at_start": state.get("docker_bridge_containers_at_start"),
            "docker_bridge_containers_at_end": state.get("docker_bridge_containers_at_end"),
            "docker_bridge_containers_delta": state.get("docker_bridge_containers_delta"),
            "docker_owned_bridge_containers_at_start": state.get("docker_owned_bridge_containers_at_start"),
            "docker_owned_bridge_containers_at_end": state.get("docker_owned_bridge_containers_at_end"),
            "docker_owned_bridge_containers_delta": state.get("docker_owned_bridge_containers_delta"),
            "docker_dynamic_networks_at_start": state.get("docker_dynamic_networks_at_start"),
            "docker_dynamic_networks_at_end": state.get("docker_dynamic_networks_at_end"),
            "docker_dynamic_networks_delta": state.get("docker_dynamic_networks_delta")}

def check_command(command: list[str], label: str) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return f"{label} command was not found: {command[0]}"
    if result.returncode:
        return f"{label} check failed: {result.stderr.strip() or result.stdout.strip()}"
    return None


def preflight_errors(args: argparse.Namespace, tasks: list[Challenge]) -> list[str]:
    errors: list[str] = []
    if not RUN_SCRIPT.is_file():
        errors.append(f"run.py not found: {RUN_SCRIPT}")
    if not tasks:
        errors.append("no valid challenge.json files matched the requested filter")
    config = Path(args.config_file)
    if not config.is_absolute() and not (ENIGMA_ROOT / config).is_file():
        errors.append(f"config file not found: {ENIGMA_ROOT / config}")
    for command, label in ((["git", "--version"], "Git"), (["docker", "image", "inspect", args.image_name], "Docker image")):
        error = check_command(command, label)
        if error:
            errors.append(error)
    required = [] if re.fullmatch(r"glm52_(?:[1-9]|10)", args.model_name) else ["OPENAI_API_KEY"]
    if re.fullmatch(r"glm52_(?:[1-9]|10)", args.model_name) and not args.messages_api_url:
        errors.append("--messages_api_url is required for glm52_* models")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        errors.append("missing API environment variables: " + ", ".join(missing))
    return errors


class BatchLock:
    def __init__(self, path: Path, force: bool) -> None:
        self.path, self.force = path, force

    def __enter__(self) -> "BatchLock":
        if self.path.exists() and not self.force and self._is_active_lock():
            raise RuntimeError(f"batch lock exists: {self.path}; another batch appears to be running")
        if self.path.exists():
            self.path.unlink()
        self.path.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "boot_id": self._boot_id(), "started_at": utc_now()}), encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, trace: object) -> None:
        self.path.unlink(missing_ok=True)

    @staticmethod
    def _boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _is_active_lock(self) -> bool:
        try:
            record = read_json(self.path)
            pid = int(record.get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if record.get("host") != socket.gethostname():
            return True
        if record.get("boot_id") and record.get("boot_id") != self._boot_id():
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return pid > 0


def record_batch_error(state: dict[str, Any], output_dir: Path, code: str, message: str) -> None:
    entry = {"code": code, "message": message, "time": utc_now()}
    state.setdefault("batch_errors", []).append(entry)
    log_path = output_dir / LOGS_DIR_NAME / "batch.error.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{entry['time']} {code}: {message}\n")


def persist_progress(state_path: Path, summary_path: Path, state: dict[str, Any], tasks: list[Challenge], skipped: int) -> None:
    atomic_write_json(state_path, state)
    summary = build_summary(tasks, state, skipped)
    atomic_write_json(summary_path, summary)
    records = [state["tasks"].get(task.relative_path, {}) for task in tasks]
    atomic_write_json(summary_path.parent / QUALITY_REPORT_FILE_NAME, trajectory_quality(records))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True, help="CTF-Dojo ctf-archive root to scan recursively")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for state, logs, and trajectories")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--image_name", default=AGENT_IMAGE)
    parser.add_argument("--config_file", default="config/default_ctf.yaml")
    parser.add_argument("--step_limit", type=int, default=40)
    parser.add_argument("--task_timeout", type=int, default=3600, help="Seconds per task; 0 disables the outer timeout")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--warmup_single_task",
        action="store_true",
        help="Run one task with one worker first; increase to --workers only after it reaches model interaction",
    )
    parser.add_argument("--max_bridge_containers", type=int, default=180, help="Stop scheduling at this bridge container count; 0 disables the threshold")
    parser.add_argument("--max_dynamic_networks", type=int, default=24, help="Stop before ctfnet-* address pools are exhausted; 0 disables the threshold")
    parser.add_argument("--category")
    parser.add_argument("--max_tasks", type=int)
    parser.add_argument("--require_dockerfile", action="store_true")
    parser.add_argument("--include_unverified", action="store_true", help="Allow tasks without local verifier evidence")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--messages_api_url", default="", help="Anthropic Messages-compatible endpoint for glm52_* models")
    parser.add_argument("--messages_api_key", default="EMPTY", help="API key for --messages_api_url")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_unlock", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the exact --output_dir state; otherwise an existing run gets a new timestamped child directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.step_limit < 1 or args.task_timeout < 0 or args.max_bridge_containers < 0 or args.max_dynamic_networks < 0:
        print("ERROR: workers and step_limit must be positive; timeouts and resource limits cannot be negative", file=sys.stderr)
        return 2
    if args.max_tasks is not None and args.max_tasks < 1:
        print("ERROR: max_tasks must be positive", file=sys.stderr)
        return 2
    if not args.dataset_root.is_dir():
        print(f"ERROR: dataset root does not exist: {args.dataset_root}", file=sys.stderr)
        return 2
    tasks, warnings = discover_challenges(args.dataset_root.resolve(), args.category, args.require_dockerfile, require_verification=not args.include_unverified)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.dry_run:
        print(f"Discovered {len(tasks)} task(s).")
        for task in tasks:
            print(f"[scan] {task.task_id} category={task.category} dockerfile={'yes' if task.dockerfile_path else 'no'} {task.relative_path}")
        return 0
    errors = preflight_errors(args, tasks)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    output_root = args.output_dir.resolve()
    existing_markers = (STATE_FILE_NAME, SUMMARY_FILE_NAME, LOGS_DIR_NAME)
    if not args.resume and any((output_root / marker).exists() for marker in existing_markers):
        run_id = f"run-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        output_dir = output_root / run_id
        print(f"Existing output detected; using fresh run directory: {output_dir}", flush=True)
    else:
        output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_model_timeout = float(os.environ.get("SWE_AGENT_MODEL_TIMEOUT", "600"))
    effective_api_timeout = float(os.environ.get(
        "SWE_AGENT_MESSAGES_API_TIMEOUT",
        str(max(5.0, effective_model_timeout - min(30.0, max(5.0, effective_model_timeout * 0.1)))),
    ))
    print(
        f"Effective runtime configuration: workers={args.workers} step_limit={args.step_limit} "
        f"task_timeout={args.task_timeout}s inner_task_timeout={max(60, args.task_timeout - 60) if args.task_timeout else 1800}s "
        f"model_timeout={effective_model_timeout:.1f}s messages_api_timeout={effective_api_timeout:.1f}s "
        f"max_bridge_containers={args.max_bridge_containers} max_dynamic_networks={args.max_dynamic_networks}",
        flush=True,
    )
    state_path, summary_path = output_dir / STATE_FILE_NAME, output_dir / SUMMARY_FILE_NAME
    exit_code = 0
    with BatchLock(output_dir / LOCK_FILE_NAME, args.force_unlock):
        state = load_state(state_path)
        stale_cleanup = cleanup_stale_attempt_resources(output_dir)
        for cleanup_report in stale_cleanup:
            if (
                cleanup_report.get("remaining_containers")
                or cleanup_report.get("remaining_networks")
                or cleanup_report.get("remaining_volumes")
                or cleanup_report.get("failed_containers")
                or cleanup_report.get("failed_networks")
                or cleanup_report.get("failed_volumes")
            ):
                record_batch_error(
                    state,
                    output_dir,
                    "docker_resource_leak",
                    f"stale attempt cleanup incomplete: {cleanup_report}",
                )
        enrich_completed_records(state, output_dir, tasks)
        selected = [task for task in tasks if not completed(state, output_dir, task)]
        skipped = len(tasks) - len(selected)
        try:
            state["docker_bridge_containers_at_start"] = docker_bridge_container_count()
            state["docker_owned_bridge_containers_at_start"] = docker_owned_bridge_container_count(output_dir)
        except Exception as error:
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge: {error}")
            persist_progress(state_path, summary_path, state, tasks, skipped)
            return 1
        try:
            state["docker_dynamic_networks_at_start"] = docker_dynamic_network_count()
            state["docker_owned_dynamic_networks_at_start"] = docker_dynamic_network_count(output_dir)
        except Exception as error:
            record_batch_error(state, output_dir, "docker_network_capacity_guard", f"unable to inspect ctfnet networks: {error}")
            persist_progress(state_path, summary_path, state, tasks, skipped)
            return 1
        if args.max_dynamic_networks and state["docker_owned_dynamic_networks_at_start"] >= args.max_dynamic_networks:
            message = f"batch owns {state['docker_owned_dynamic_networks_at_start']} ctfnet-* networks (global count: {state['docker_dynamic_networks_at_start']}), submissions paused until resources are reclaimed"
            record_batch_error(state, output_dir, "docker_network_capacity_guard", message)
            print(f"WARNING: docker_network_capacity_guard: {message}", file=sys.stderr)
        if args.max_bridge_containers and state["docker_owned_bridge_containers_at_start"] >= args.max_bridge_containers:
            message = f"batch owns {state['docker_owned_bridge_containers_at_start']} bridge containers, submissions paused until workers/stale attempts are cleaned (global bridge count: {state['docker_bridge_containers_at_start']})"
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", message)
            print(f"WARNING: docker_bridge_capacity_guard: {message}", file=sys.stderr)
        persist_progress(state_path, summary_path, state, tasks, skipped)
        print(f"Discovered {len(tasks)} task(s); running {len(selected)}; skipping {skipped} completed task(s).", flush=True)
        requests = [WorkerRequest(task, str(output_dir), args.model_name, args.image_name, args.config_file, args.step_limit, args.task_timeout, args.python_executable, args.messages_api_url, args.messages_api_key) for task in selected]
        pending_index = 0
        futures: dict[concurrent.futures.Future[dict[str, Any]], Challenge] = {}
        guard_triggered = False
        capacity_paused = False
        interrupted = False
        warmup_active = bool(args.warmup_single_task and selected)
        target_workers = 1 if warmup_active else args.workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            while True:
                capacity_paused = False
                while not guard_triggered and len(futures) < target_workers and pending_index < len(requests):
                    request = requests[pending_index]
                    if state.get("resource_pause"):
                        try:
                            cleanup_stale_attempt_resources(output_dir)
                            owned_network_count = docker_dynamic_network_count(output_dir)
                            owned_bridge_count = docker_owned_bridge_container_count(output_dir)
                        except Exception as error:
                            record_batch_error(state, output_dir, "docker_resource_leak", f"unable to recheck owned resources: {error}")
                            exit_code = 1
                            capacity_paused = True
                            break
                        if owned_network_count == 0 and owned_bridge_count == 0:
                            state.pop("resource_pause", None)
                            state["owned_resource_leak_streak"] = 0
                            print("Owned resources returned to zero; resuming submissions.", flush=True)
                        else:
                            capacity_paused = True
                            print(
                                f"Owned resources still present (containers={owned_bridge_count}, networks={owned_network_count}); waiting before submitting new tasks.",
                                file=sys.stderr, flush=True,
                            )
                            break
                    try:
                        bridge_count = docker_bridge_container_count()
                        owned_bridge_count = docker_owned_bridge_container_count(output_dir)
                    except Exception as error:
                        record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge before {request.challenge.task_id}: {error}")
                        guard_triggered = True
                        exit_code = 1
                        break
                    # Account for workers that have been submitted but whose
                    # containers are not visible in ``docker network inspect``
                    # yet. Without this reservation, workers=8 could submit
                    # eight more containers on top of an already-populated
                    # bridge and trigger Docker creation failures.
                    effective_bridge_count = owned_bridge_count + len(futures)
                    if args.max_bridge_containers and effective_bridge_count >= args.max_bridge_containers:
                        if futures:
                            # Let active workers finish, then re-check and
                            # resume scheduling without dropping this request.
                            break
                        message = f"batch owns {owned_bridge_count} bridge containers, at or above configured limit {args.max_bridge_containers} (global bridge count: {bridge_count}); stopped before {request.challenge.task_id}"
                        record_batch_error(state, output_dir, "docker_bridge_capacity_guard", message)
                        print(f"WARNING: docker_bridge_capacity_guard: {message}; waiting for cleanup", file=sys.stderr)
                        capacity_paused = True
                        break
                    try:
                        dynamic_network_count = docker_dynamic_network_count(output_dir)
                    except Exception as error:
                        record_batch_error(state, output_dir, "docker_network_capacity_guard", f"unable to inspect ctfnet networks before {request.challenge.task_id}: {error}")
                        capacity_paused = True
                        break
                    if args.max_dynamic_networks and dynamic_network_count >= args.max_dynamic_networks:
                        message = f"found {dynamic_network_count} ctfnet-* networks; stopped before {request.challenge.task_id}"
                        record_batch_error(state, output_dir, "docker_network_capacity_guard", message)
                        print(f"ERROR: docker_network_capacity_guard: {message}", file=sys.stderr)
                        capacity_paused = True
                        break
                    pending_index += 1
                    futures[executor.submit(run_one_task, request)] = request.challenge
                if not futures:
                    if guard_triggered or pending_index >= len(requests):
                        break
                    if capacity_paused:
                        cleanup_stale_attempt_resources(output_dir)
                        persist_progress(state_path, summary_path, state, tasks, skipped)
                        time.sleep(2.0)
                        continue
                    break
                try:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                except KeyboardInterrupt:
                    record_batch_error(state, output_dir, "interrupted", "batch interrupted by user; waiting for active task cleanup")
                    interrupted, guard_triggered, exit_code = True, True, 130
                    continue
                for future in done:
                    task = futures.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        log_path = output_dir / LOGS_DIR_NAME / f"{task.task_id}.error.log"
                        save_failure_log(log_path, task, [], str(error), trace=traceback.format_exc())
                        destination = persist_terminal_trajectory(output_dir, task, exit_status="runner_exception", terminal_reason=str(error), error_type="worker_failed")
                        payload = read_trajectory(destination)
                        flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
                        record = task_record(task, run_status="error", trajectory_generated=True, trajectory=str(destination), exit_status="runner_exception", time=utc_now(), error=str(error), error_code="worker_failed", error_log=str(log_path), outcome_status=outcome_status_from(flags, episode, True), **episode, **flags, container_cleanup={"attempted": False, "removed_container_ids": [], "failed_container_ids": []})
                    state["tasks"][task.relative_path] = record
                    persist_progress(state_path, summary_path, state, tasks, skipped)
                    print(f"[{record.get('run_status', 'error')}] {task.task_id}", flush=True)
                    if record.get("resource_cleanup_clean") is False:
                        state["owned_resource_leak_streak"] = state.get("owned_resource_leak_streak", 0) + 1
                        message = f"resource cleanup incomplete for {task.task_id}: {record.get('attempt_resource_cleanup', {})}"
                        record_batch_error(state, output_dir, "docker_resource_leak", message)
                        if state["owned_resource_leak_streak"] >= 2:
                            state["resource_pause"] = True
                    else:
                        state["owned_resource_leak_streak"] = 0
                    if state.get("owned_resource_leak_streak", 0) >= 2:
                        record_batch_error(
                            state,
                            output_dir,
                            "docker_resource_leak",
                            "owned resources grew across consecutive task completions; pausing submissions",
                        )
                        state["resource_pause"] = True
                    if warmup_active:
                        reached_model = (
                            int(record.get("steps", 0) or 0) > 1
                            and record.get("episode_status") not in {"ctf_server_unavailable", "environment_error"}
                        )
                        if reached_model:
                            warmup_active = False
                            target_workers = args.workers
                            print(f"Warm-up task reached model interaction; increasing workers to {target_workers}.", flush=True)
                        else:
                            guard_triggered = True
                            exit_code = 1
                            print("Warm-up task did not reach model interaction; stopping before concurrent scheduling.", file=sys.stderr, flush=True)
        try:
            state["docker_bridge_containers_at_end"] = docker_bridge_container_count()
            state["docker_owned_bridge_containers_at_end"] = docker_owned_bridge_container_count(output_dir)
            state["docker_bridge_containers_delta"] = state["docker_bridge_containers_at_end"] - state["docker_bridge_containers_at_start"]
            state["docker_owned_bridge_containers_delta"] = state["docker_owned_bridge_containers_at_end"] - state["docker_owned_bridge_containers_at_start"]
        except Exception as error:
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge at batch end: {error}")
            exit_code = 1
        try:
            state["docker_dynamic_networks_at_end"] = docker_dynamic_network_count()
            state["docker_owned_dynamic_networks_at_end"] = docker_dynamic_network_count(output_dir)
            state["docker_dynamic_networks_delta"] = state["docker_dynamic_networks_at_end"] - state["docker_dynamic_networks_at_start"]
            state["docker_owned_dynamic_networks_delta"] = state["docker_owned_dynamic_networks_at_end"] - state.get("docker_owned_dynamic_networks_at_start", 0)
        except Exception as error:
            record_batch_error(state, output_dir, "docker_network_capacity_guard", f"unable to inspect ctfnet networks at batch end: {error}")
            exit_code = 1
        persist_progress(state_path, summary_path, state, tasks, skipped)
    summary = build_summary(tasks, state, skipped)
    print(f"Completed batch: total_tasks={summary['total_tasks']} generated_trajectories={summary['generated_trajectories']} verified_solved={summary['verified_solved']} unsolved={summary['unsolved']} generation_failed={summary['generation_failed']} environment_error={summary['environment_error']}")
    return exit_code or (0 if summary["generation_failed"] == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
