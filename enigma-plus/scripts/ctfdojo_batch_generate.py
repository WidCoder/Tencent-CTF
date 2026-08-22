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
import traceback
import uuid
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


def has_verification_evidence(repo_path: Path, metadata: dict[str, Any]) -> bool:
    verification = metadata.get("verification")
    if isinstance(verification, dict) and verification.get("status") == "eligible":
        return True
    if metadata.get("verification_method") in {"sha256", "flagcheck"}:
        return True
    return any(
        path.is_file() and (path.name.lower() in {"flag.sha256", ".flag.sha256", "flag.sha256.txt"} or "flagcheck" in path.name.lower())
        for path in repo_path.rglob("*")
    )

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
        task_id = f"{safe_component(stable_path.replace('/', '__'))}-{hashlib.sha1(stable_path.encode()).hexdigest()[:10]}"
        compose = repo_path / "docker-compose.yml"
        challenges.append(Challenge(
            challenge_json=str(challenge_json.resolve()), repo_path=str(repo_path.resolve()),
            relative_path=stable_path, event=relative_path.parts[0] if relative_path.parts else "unknown",
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
        return {"flag_submitted": True, "flag_verified": "unknown", "flag_verification_evidence": "missing structured verifier result"}
    return {"flag_submitted": False, "flag_verified": "unknown", "flag_verification_evidence": "not_attempted"}


def episode_status_from_trajectory(payload: dict[str, Any]) -> dict[str, Any]:
    """Map implementation exit statuses to stable dataset episode statuses."""
    info = payload.get("info")
    info = info if isinstance(info, dict) else {}
    trajectory = payload.get("trajectory")
    exit_status = info.get("exit_status")
    if not isinstance(exit_status, str) or not exit_status:
        return {"episode_status": "running", "episode_complete": False}
    if exit_status == "submitted":
        status = "completed"
    elif exit_status in {"task_timeout", "timeout"}:
        status = "timeout"
    elif re.fullmatch(r"step_\d+_hit", exit_status):
        status = "step_limit"
    elif exit_status == "exit_format":
        status = "format_error"
    elif any(token in exit_status for token in ("environment", "container", "server_crashed", "docker")):
        status = "environment_error"
    else:
        status = "early_exit"
    return {"episode_status": status, "episode_complete": status != "running" and isinstance(trajectory, list)}


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
            flags = flag_status_from_trajectory(payload)
            episode = episode_status_from_trajectory(payload)
            values = {"run_status": "generated", "trajectory_generated": True, **episode, **flags, "outcome_status": outcome_status_from(flags, episode, True)}
        except (OSError, ValueError, json.JSONDecodeError):
            values = {"run_status": "failed", "trajectory_generated": True, "episode_status": "running", "episode_complete": False, "flag_submitted": False, "flag_verified": "unknown", "flag_verification_evidence": "not_attempted", "outcome_status": "unknown"}
        for key, value in values.items():
            if record.get(key) != value:
                record[key] = value
                changed = True
    return changed


def completed(state: dict[str, Any], output_dir: Path, task: Challenge) -> bool:
    record = state["tasks"].get(task.relative_path, {})
    return record.get("run_status") == "generated" and record.get("episode_status") != "running" and valid_trajectory(trajectory_path(output_dir, task))


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


def stage_challenge(source_dir: Path, workspace: Path) -> Path:
    staged_repo = workspace / "repo"
    shutil.copytree(source_dir, staged_repo)
    run_checked(["git", "init"], staged_repo)
    run_checked(["git", "add", "."], staged_repo)
    run_checked(["git", "-c", "user.name=EnIGMA Batch", "-c", "user.email=enigma-batch@example.invalid", "commit", "-m", "Stage CTF-Dojo challenge for EnIGMA+"], staged_repo)
    return staged_repo


def find_trajectory(run_root: Path) -> Path | None:
    # run_root is unique per attempt; do not rely on filesystem timestamp precision.
    candidates = [path for path in run_root.rglob("*.traj") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
def save_failure_log(path: Path, task: Challenge, command: list[str], error: str, stdout: str = "", stderr: str = "", trace: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp: " + utc_now() + "\nchallenge_json: " + task.challenge_json + "\nrepo_path: " + task.repo_path
        + "\ndockerfile_path: " + (task.dockerfile_path or "(none)") + "\ncommand: " + " ".join(command)
        + "\n\nerror:\n" + error + "\n\nstdout:\n" + stdout + "\n\nstderr:\n" + stderr
        + ("\n\ntraceback:\n" + trace if trace else ""), encoding="utf-8")


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


def cleanup_agent_containers(baseline: set[str] | None, image_name: str = AGENT_IMAGE) -> dict[str, Any]:
    """Delete only containers in (current exact-image set - baseline)."""
    cleanup: dict[str, Any] = {"attempted": baseline is not None, "removed_container_ids": [], "failed_container_ids": []}
    if baseline is None:
        cleanup["error"] = "agent container baseline unavailable"
        return cleanup
    try:
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


def docker_network_exhausted(*output: str) -> bool:
    detail = "\n".join(output).lower()
    return "no available ipv4 addresses" in detail and "address pools" in detail


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
    info.update({
        "exit_status": exit_status,
        "terminal_reason": terminal_reason,
        "error_type": error_type or "",
        "flag_verified": info.get("flag_verified") if isinstance(info.get("flag_verified"), bool) else "unknown",
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
    runner_output = output_dir / task.task_id / "enigma_output" / attempt_id
    error_log = output_dir / LOGS_DIR_NAME / f"{task.task_id}.error.log"
    command: list[str] = []
    baseline: set[str] | None = None
    started_at = dt.datetime.now().timestamp()
    result: subprocess.CompletedProcess[str] | None = None
    timeout_error: subprocess.TimeoutExpired | None = None
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
                result = subprocess.run(command, cwd=ENIGMA_ROOT, capture_output=True, text=True,
                                        check=False, timeout=request.task_timeout or None)
            except subprocess.TimeoutExpired as error:
                timeout_error = error
        source = find_trajectory(runner_output)
        if timeout_error is not None:
            reason = f"batch task timeout after {request.task_timeout} seconds"
            destination = persist_terminal_trajectory(output_dir, task, exit_status="task_timeout",
                terminal_reason=reason, error_type="outer_timeout", source=source)
            save_failure_log(error_log, task, command, reason, timeout_error.stdout or "", timeout_error.stderr or "", traceback.format_exc())
            payload = read_trajectory(destination)
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source) if source else "", exit_status="task_timeout", error=reason, error_code="task_timeout",
                error_log=str(error_log), solved=False, steps=len(payload.get("trajectory", [])), **episode, **flags)
        elif result is None or result.returncode != 0:
            reason = f"run.py exited with code {result.returncode}" if result is not None else "run.py did not complete"
            error_type = "docker_network_address_exhausted" if result and docker_network_exhausted(result.stdout, result.stderr) else "runner_error"
            exit_status = "environment_error" if error_type.startswith("docker_") else "runner_exception"
            destination = persist_terminal_trajectory(output_dir, task, exit_status=exit_status,
                terminal_reason=reason, error_type=error_type, source=source)
            save_failure_log(error_log, task, command, reason, result.stdout if result else "", result.stderr if result else "")
            payload = read_trajectory(destination)
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source) if source else "", exit_status=exit_status, error=reason, error_code=error_type,
                error_log=str(error_log), solved=False, steps=len(payload.get("trajectory", [])), **episode, **flags)
        elif source is None:
            reason = "run.py exited successfully but no trajectory file was produced"
            destination = persist_terminal_trajectory(output_dir, task, exit_status="runner_exception", terminal_reason=reason, error_type="missing_trajectory")
            save_failure_log(error_log, task, command, reason)
            payload = read_trajectory(destination)
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
            atomic_write_trajectory(destination, payload)
            flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
            record = task_record(task, run_status="generated", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
                source_trajectory=str(source), exit_status=payload.get("info", {}).get("exit_status", ""),
                solved=flags["flag_verified"] is True, steps=len(payload.get("trajectory", [])), **episode, **flags)
        record["outcome_status"] = outcome_status_from(flags, episode, True)
    except BaseException as error:
        reason = str(error)
        save_failure_log(error_log, task, command, reason, trace=traceback.format_exc())
        error_code = "interrupted" if isinstance(error, KeyboardInterrupt) else "task_setup_failed"
        destination = persist_terminal_trajectory(output_dir, task, exit_status="runner_exception", terminal_reason=reason, error_type=error_code)
        payload = read_trajectory(destination)
        flags, episode = flag_status_from_trajectory(payload), episode_status_from_trajectory(payload)
        record = task_record(task, run_status="error", trajectory_generated=True, time=utc_now(), trajectory=str(destination),
            exit_status="runner_exception", error=reason, error_code=error_code, error_log=str(error_log), solved=False,
            steps=len(payload.get("trajectory", [])), outcome_status=outcome_status_from(flags, episode, True), **episode, **flags)
    finally:
        cleanup = cleanup_agent_containers(baseline, request.image_name)
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
        if status == "environment_error": counts["environment_error"] += 1
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
    return {"generated_trajectories": len(generated), "verified_solved": sum(record.get("outcome_status") == "solved" for record in generated),
            "unsolved": sum(record.get("outcome_status") == "unsolved" for record in generated),
            "generation_failed": sum(record.get("run_status") in {"failed", "error"} for record in records),
            "environment_error": sum(record.get("episode_status") == "environment_error" for record in records)}


def build_summary(tasks: list[Challenge], state: dict[str, Any], skipped: int) -> dict[str, Any]:
    records = [state["tasks"].get(task.relative_path, {}) for task in tasks]
    counts = reporting_counts(records)
    return {"generated_at": utc_now(), "total_tasks": len(tasks), "skipped": skipped,
            "pending": len(tasks) - sum(bool(record) for record in records), **counts,
            "run_status": {key: sum(record.get("run_status") == key for record in records) for key in ("generated", "failed", "error")},
            "batch_errors": state.get("batch_errors", []),
            "docker_bridge_containers_at_start": state.get("docker_bridge_containers_at_start"),
            "docker_bridge_containers_at_end": state.get("docker_bridge_containers_at_end")}
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
    parser.add_argument("--max_bridge_containers", type=int, default=180, help="Stop scheduling at this bridge container count; 0 disables the threshold")
    parser.add_argument("--category")
    parser.add_argument("--max_tasks", type=int)
    parser.add_argument("--require_dockerfile", action="store_true")
    parser.add_argument("--include_unverified", action="store_true", help="Allow tasks without local verifier evidence")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--messages_api_url", default="", help="Anthropic Messages-compatible endpoint for glm52_* models")
    parser.add_argument("--messages_api_key", default="EMPTY", help="API key for --messages_api_url")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_unlock", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.step_limit < 1 or args.task_timeout < 0 or args.max_bridge_containers < 0:
        print("ERROR: workers and step_limit must be positive; task_timeout and max_bridge_containers cannot be negative", file=sys.stderr)
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path, summary_path = output_dir / STATE_FILE_NAME, output_dir / SUMMARY_FILE_NAME
    exit_code = 0
    with BatchLock(output_dir / LOCK_FILE_NAME, args.force_unlock):
        state = load_state(state_path)
        enrich_completed_records(state, output_dir, tasks)
        selected = [task for task in tasks if not completed(state, output_dir, task)]
        skipped = len(tasks) - len(selected)
        try:
            state["docker_bridge_containers_at_start"] = docker_bridge_container_count()
        except Exception as error:
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge: {error}")
            persist_progress(state_path, summary_path, state, tasks, skipped)
            return 1
        if args.max_bridge_containers and state["docker_bridge_containers_at_start"] >= args.max_bridge_containers:
            message = f"bridge has {state['docker_bridge_containers_at_start']} containers, at or above configured limit {args.max_bridge_containers}"
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", message)
            state["docker_bridge_containers_at_end"] = state["docker_bridge_containers_at_start"]
            persist_progress(state_path, summary_path, state, tasks, skipped)
            print(f"ERROR: docker_bridge_capacity_guard: {message}", file=sys.stderr)
            return 1
        persist_progress(state_path, summary_path, state, tasks, skipped)
        print(f"Discovered {len(tasks)} task(s); running {len(selected)}; skipping {skipped} completed task(s).", flush=True)
        requests = [WorkerRequest(task, str(output_dir), args.model_name, args.image_name, args.config_file, args.step_limit, args.task_timeout, args.python_executable, args.messages_api_url, args.messages_api_key) for task in selected]
        pending = iter(requests)
        futures: dict[concurrent.futures.Future[dict[str, Any]], Challenge] = {}
        guard_triggered = False
        interrupted = False
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            while True:
                while not guard_triggered and len(futures) < args.workers:
                    try:
                        request = next(pending)
                    except StopIteration:
                        break
                    try:
                        bridge_count = docker_bridge_container_count()
                    except Exception as error:
                        record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge before {request.challenge.task_id}: {error}")
                        guard_triggered = True
                        exit_code = 1
                        break
                    if args.max_bridge_containers and bridge_count >= args.max_bridge_containers:
                        message = f"bridge has {bridge_count} containers, at or above configured limit {args.max_bridge_containers}; stopped before {request.challenge.task_id}"
                        record_batch_error(state, output_dir, "docker_bridge_capacity_guard", message)
                        print(f"ERROR: docker_bridge_capacity_guard: {message}", file=sys.stderr)
                        guard_triggered = True
                        exit_code = 1
                        break
                    futures[executor.submit(run_one_task, request)] = request.challenge
                if not futures:
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
        try:
            state["docker_bridge_containers_at_end"] = docker_bridge_container_count()
        except Exception as error:
            record_batch_error(state, output_dir, "docker_bridge_capacity_guard", f"unable to inspect bridge at batch end: {error}")
            exit_code = 1
        persist_progress(state_path, summary_path, state, tasks, skipped)
    summary = build_summary(tasks, state, skipped)
    print(f"Completed batch: total_tasks={summary['total_tasks']} generated_trajectories={summary['generated_trajectories']} verified_solved={summary['verified_solved']} unsolved={summary['unsolved']} generation_failed={summary['generation_failed']} environment_error={summary['environment_error']}")
    return exit_code or (0 if summary["generation_failed"] == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())