#!/usr/bin/env python3
"""Batch-generate EnIGMA+ CTF trajectories from a benchmark directory.

The script discovers challenge.json files, stages each challenge in a temporary
Git repository (required by EnIGMA+ single-instance mode), and invokes run.py
without duplicating any agent or environment logic.
"""

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
STATUS_FILE_NAME = "batch_status.json"
SUMMARY_FILE_NAME = "summary.json"
LOGS_DIR_NAME = "logs"
TRAJECTORY_FILE_NAME = "trajectory.jsonl"


@dataclasses.dataclass(frozen=True)
class Challenge:
    """A discovered CTF challenge and its stable batch identifier."""

    challenge_json: str
    source_dir: str
    relative_path: str
    platform: str
    category: str
    name: str
    task_id: str


@dataclasses.dataclass(frozen=True)
class WorkerRequest:
    """Serializable input for one worker process."""

    challenge: Challenge
    output_dir: str
    model_name: str
    image_name: str
    config_file: str
    step_limit: int
    python_executable: str
    task_timeout: int = 3600


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sanitize_component(value: str) -> str:
    """Return a filesystem-safe, readable path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, dict):
        raise ValueError("challenge.json must contain a JSON object")
    return content


def get_category(metadata: dict[str, Any], relative_parent: Path) -> str:
    category = metadata.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()

    categories = metadata.get("categories")
    if isinstance(categories, list) and categories and isinstance(categories[0], str):
        return categories[0].strip()

    if len(relative_parent.parts) >= 2:
        return relative_parent.parts[-2]
    return "unknown"


def discover_challenges(benchmark_path: Path, category_filter: str | None) -> list[Challenge]:
    """Find valid challenge.json files beneath ``benchmark_path``."""
    discovered: list[Challenge] = []
    used_ids: set[str] = set()
    normalized_filter = category_filter.lower() if category_filter else None

    for challenge_json in sorted(benchmark_path.rglob("challenge.json")):
        try:
            metadata = read_json(challenge_json)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"WARNING: skipping invalid {challenge_json}: {error}", file=sys.stderr)
            continue

        source_dir = challenge_json.parent
        relative_dir = source_dir.relative_to(benchmark_path)
        platform = relative_dir.parts[0] if relative_dir.parts else "benchmark"
        category = get_category(metadata, relative_dir)
        if normalized_filter and category.lower() != normalized_filter:
            continue

        name = metadata.get("name")
        if not isinstance(name, str) or not name.strip():
            name = source_dir.name
        relative_path = source_dir.relative_to(benchmark_path).as_posix()
        base_id = "_".join(
            [sanitize_component(platform), sanitize_component(category), sanitize_component(name)]
        )
        task_id = base_id
        if task_id in used_ids:
            suffix = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
            task_id = f"{base_id}_{suffix}"
        used_ids.add(task_id)

        discovered.append(
            Challenge(
                challenge_json=str(challenge_json),
                source_dir=str(source_dir),
                relative_path=relative_path,
                platform=platform,
                category=category,
                name=name,
                task_id=task_id,
            )
        )
    return discovered


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "tasks": {}}
    try:
        data = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read existing status file {path}: {error}") from error
    if not isinstance(data.get("tasks"), dict):
        raise RuntimeError(f"Invalid status file {path}: expected a tasks object")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def trajectory_path_for(output_dir: Path, task_id: str) -> Path:
    return output_dir / task_id / TRAJECTORY_FILE_NAME


def is_completed(status: dict[str, Any], output_dir: Path, task: Challenge) -> bool:
    record = status["tasks"].get(task.relative_path, {})
    return record.get("status") == "success" and trajectory_path_for(output_dir, task.task_id).is_file()


def run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)


def stage_challenge(source_dir: Path, workspace: Path) -> Path:
    """Copy a challenge and turn the copy into the Git repository run.py needs."""
    repo_path = workspace / "repo"
    vcs_metadata = {".git", ".hg", ".svn", ".bzr", "_darcs"}

    def ignore_vcs(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.lower() in vcs_metadata}

    shutil.copytree(source_dir, repo_path, ignore=ignore_vcs)
    staged_vcs = [
        path for path in repo_path.rglob("*")
        if path.name.lower() in vcs_metadata
    ]
    if staged_vcs:
        raise RuntimeError(
            "staged repository contains forbidden VCS metadata: "
            + ", ".join(str(path) for path in staged_vcs)
        )
    run_checked(["git", "init"], cwd=repo_path)
    run_checked(["git", "add", "."], cwd=repo_path)
    run_checked(
        [
            "git",
            "-c",
            "user.name=Enigma Batch",
            "-c",
            "user.email=enigma-batch@example.invalid",
            "commit",
            "-m",
            "Stage challenge for EnIGMA+",
        ],
        cwd=repo_path,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError(f"staged repository is dirty after initialization:\n{status}")
    return repo_path


def find_latest_trajectory(run_root: Path, started_at: float | None = None) -> Path | None:
    candidates = [
        path for path in run_root.rglob("*.traj") if path.is_file() and (started_at is None or path.stat().st_mtime >= started_at)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_trajectory(source: Path, destination: Path) -> dict[str, Any]:
    """Write EnIGMA+'s JSON trajectory as a one-record JSONL artifact."""
    payload = read_json(source)
    atomic_write_trajectory(destination, payload)
    return payload


def atomic_write_trajectory(path: Path, payload: dict[str, Any]) -> None:
    """Persist a trajectory as one JSONL record without exposing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def persist_failure_trajectory(output_dir: Path, challenge: Challenge, *, reason: str, failure_category: str, source: Path | None = None) -> Path:
    """Always leave a terminal artifact, including when the runner never starts."""
    payload: dict[str, Any] = {}
    if source is not None:
        try:
            payload = read_json(source)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    info = dict(info)
    exit_status = "environment_error" if failure_category.startswith("environment") or failure_category.startswith("docker") else "runner_exception"
    original_exit_status = info.get("exit_status")
    original_reason = info.get("terminal_reason") or info.get("error_message")
    original_error_type = info.get("error_type")
    effective_exit_status = original_exit_status or exit_status
    if exit_status == "environment_error" and original_exit_status == "runner_exception":
        effective_exit_status = exit_status
    if original_exit_status and original_exit_status != exit_status:
        info["wrapped_exit_status"] = exit_status
    if original_reason and original_reason != reason:
        info["wrapped_terminal_reason"] = reason
    if original_error_type and original_error_type != failure_category:
        info["wrapped_error_type"] = failure_category
    info.update({"exit_status": effective_exit_status, "terminal_reason": original_reason or reason, "error_type": original_error_type or failure_category, "failure_category": failure_category, "episode_end_time": utc_now(), "total_steps": len(trajectory)})
    if not trajectory or not isinstance(trajectory[-1], dict) or not trajectory[-1].get("terminal"):
        trajectory.append({"step_id": len(trajectory) + 1, "thought": "", "action": exit_status, "tool_name": "unknown", "tool_args": "", "observation": reason, "return_code": None, "execution_time": 0.0, "error": failure_category, "terminal": True, "response": ""})
        info["total_steps"] = len(trajectory)
    payload.update({"trajectory_schema_version": 2, "artifact_type": "synthetic_error" if source is None else "partial_trace", "environment": payload.get("environment", "batch_runner"), "trajectory": trajectory, "history": payload.get("history", []), "info": info})
    destination = trajectory_path_for(output_dir, challenge.task_id)
    atomic_write_trajectory(destination, payload)
    return destination


def classify_runner_failure(completed: subprocess.CompletedProcess[str] | None, *, missing_trajectory: bool = False) -> str:
    """Classify observable runner failures for state/reporting."""
    if missing_trajectory:
        return "trajectory_save_failure"
    if completed is None:
        return "task_not_executed"
    detail = f"{completed.stdout}\n{completed.stderr}".lower()
    if "no space left on device" in detail and any(token in detail for token in ("docker", "buildx", "compose", ".docker")):
        return "environment_storage_failure"
    if any(token in detail for token in ("timeout", "timed out", "rate limit", "429", "api error", "openai")):
        return "api_call_failure"
    if any(token in detail for token in ("docker", "container", "network", "compose", "port", "address pools", "fully subnetted")):
        return "environment_start_failure"
    return "agent_run_failure"


def model_stats(payload: dict[str, Any]) -> tuple[int, int, int]:
    trajectory = payload.get("trajectory")
    steps = len(trajectory) if isinstance(trajectory, list) else 0
    info = payload.get("info")
    stats = info.get("model_stats", {}) if isinstance(info, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    input_tokens = int(stats.get("tokens_sent", 0) or 0)
    output_tokens = int(stats.get("tokens_received", 0) or 0)
    return steps, input_tokens, output_tokens


def task_record(challenge: Challenge, **values: Any) -> dict[str, Any]:
    record = {
        "task": challenge.name,
        "task_id": challenge.task_id,
        "relative_path": challenge.relative_path,
        "platform": challenge.platform,
        "category": challenge.category,
        "challenge_json": challenge.challenge_json,
        "updated_at": utc_now(),
    }
    record.update(values)
    return record


def run_one_task(request: WorkerRequest) -> dict[str, Any]:
    """Run a task in a temporary Git repo and return a serializable status record."""
    challenge = request.challenge
    output_dir = Path(request.output_dir)
    task_dir = output_dir / challenge.task_id
    attempt_id = f"attempt-{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
    runner_output_dir = task_dir / "enigma_output" / attempt_id
    log_path = output_dir / LOGS_DIR_NAME / f"{challenge.task_id}.error.log"

    try:
        with tempfile.TemporaryDirectory(prefix=f"enigma-{sanitize_component(challenge.name)}-") as temporary_dir:
            staged_repo = stage_challenge(Path(challenge.source_dir), Path(temporary_dir))
            staged_data = staged_repo / "challenge.json"
            command = [
                request.python_executable,
                str(RUN_SCRIPT),
                "--model_name",
                request.model_name,
                "--image_name",
                request.image_name,
                "--data_path",
                str(staged_data),
                "--repo_path",
                str(staged_repo),
                "--config_file",
                request.config_file,
                "--per_instance_step_limit",
                str(request.step_limit),
                "--trajectory_path",
                str(runner_output_dir),
            ]
            run_env = os.environ.copy()
            inner_timeout = max(60, request.task_timeout - 60) if request.task_timeout else 1800
            run_env["SWE_AGENT_TASK_TIMEOUT"] = str(inner_timeout)
            run_env.setdefault("SWE_AGENT_FLAG_VERIFIER_IMAGE", request.image_name)
            completed = subprocess.run(
                command,
                cwd=ENIGMA_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=request.task_timeout or None,
                env=run_env,
            )

        trajectory_source = find_latest_trajectory(runner_output_dir)
        if completed.returncode != 0 or trajectory_source is None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "Command:\n"
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + completed.stdout
                + "\n\nSTDERR:\n"
                + completed.stderr
                + "\n",
                encoding="utf-8",
            )
            reason = f"run.py exited with code {completed.returncode}"
            missing_trajectory = trajectory_source is None
            if missing_trajectory:
                reason += "; no trajectory file was produced"
            failure_category = classify_runner_failure(
                completed,
                missing_trajectory=missing_trajectory and completed.returncode == 0,
            )
            destination = persist_failure_trajectory(
                output_dir,
                challenge,
                reason=reason,
                failure_category=failure_category,
                source=trajectory_source,
            )
            return task_record(
                challenge,
                status="failed",
                trajectory_path=str(destination),
                source_trajectory_path=str(trajectory_source) if trajectory_source else "",
                time=utc_now(),
                error=reason,
                error_code=failure_category,
                failure_category=failure_category,
                error_log=str(log_path),
            )

        normalized_path = trajectory_path_for(output_dir, challenge.task_id)
        payload = normalize_trajectory(trajectory_source, normalized_path)
        steps, input_tokens, output_tokens = model_stats(payload)
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        return task_record(
            challenge,
            status="success",
            trajectory_path=str(normalized_path),
            source_trajectory_path=str(trajectory_source),
            time=utc_now(),
            error="",
            exit_status=info.get("exit_status"),
            # A submitted flag is not proof of correctness; only the
            # verifier-owned boolean may mark a trajectory solved.
            solved=info.get("flag_verified") is True,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as error:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        failure_category = "task_timeout" if isinstance(error, subprocess.TimeoutExpired) else classify_runner_failure(None)
        destination = persist_failure_trajectory(
            output_dir,
            challenge,
            reason=str(error),
            failure_category=failure_category,
            source=None,
        )
        return task_record(
            challenge,
            status="failed",
            trajectory_path=str(destination),
            time=utc_now(),
            error=str(error),
            error_code=failure_category,
            failure_category=failure_category,
            error_log=str(log_path),
        )


def build_summary(tasks: list[Challenge], status: dict[str, Any]) -> dict[str, Any]:
    records = [status["tasks"].get(task.relative_path, {}) for task in tasks]
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") == "failed"]
    categories: dict[str, dict[str, Any]] = {}
    for task, record in zip(tasks, records):
        category_data = categories.setdefault(task.category, {"total": 0, "success": 0, "failed": 0})
        category_data["total"] += 1
        if record.get("status") == "success":
            category_data["success"] += 1
        elif record.get("status") == "failed":
            category_data["failed"] += 1
    for category_data in categories.values():
        total = category_data["total"]
        category_data["success_rate"] = category_data["success"] / total if total else 0.0

    total_steps = sum(int(record.get("steps", 0) or 0) for record in successful)
    total_input = sum(int(record.get("input_tokens", 0) or 0) for record in successful)
    total_output = sum(int(record.get("output_tokens", 0) or 0) for record in successful)
    completed_count = len(successful)
    return {
        "generated_at": utc_now(),
        "total_tasks": len(tasks),
        "success_count": completed_count,
        "failure_count": len(failed),
        "pending_count": len(tasks) - completed_count - len(failed),
        "average_steps": total_steps / completed_count if completed_count else 0.0,
        "average_token_consumption": (total_input + total_output) / completed_count if completed_count else 0.0,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "categories": categories,
    }


def check_command(command: list[str], label: str) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return f"{label} command was not found: {command[0]}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return f"{label} check failed: {detail}"
    return None


def environment_errors(args: argparse.Namespace, challenges: list[Challenge]) -> list[str]:
    errors: list[str] = []
    if not RUN_SCRIPT.is_file():
        errors.append(f"run.py not found at {RUN_SCRIPT}")
    if not challenges:
        errors.append("No valid challenge.json files matched the requested scan/filter")
    errors.extend(
        error
        for error in [
            check_command(["git", "--version"], "Git"),
            check_command(["docker", "image", "inspect", args.image_name], "Docker image"),
        ]
        if error
    )

    model_name = args.model_name.lower()
    if model_name.startswith("deepseek") and not model_name.startswith("deepseek-r"):
        required_variables = ["DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE_URL"]
    else:
        required_variables = ["OPENAI_API_KEY"]
        if model_name.startswith("glm"):
            required_variables.append("OPENAI_API_BASE_URL")
    missing_variables = [name for name in required_variables if not os.environ.get(name)]
    if missing_variables:
        errors.append("Missing required API environment variables: " + ", ".join(missing_variables))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_path", type=Path, required=True, help="Root directory to scan for challenge.json files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root directory for task trajectories and batch state")
    parser.add_argument("--model_name", required=True, help="Model name accepted by run.py, e.g. glm52")
    parser.add_argument("--image_name", default="sweagent/enigma:latest", help="Docker image passed to run.py")
    parser.add_argument("--config_file", default="config/default_ctf.yaml", help="Agent config passed to run.py")
    parser.add_argument("--step_limit", type=int, default=20, help="Per-task EnIGMA+ interaction limit")
    parser.add_argument("--max_tasks", type=int, help="Maximum number of matching tasks to run")
    parser.add_argument("--category", help="Only run challenges whose category matches this value")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent run.py processes (default: 1)")
    parser.add_argument("--python_executable", default=sys.executable, help="Python executable used to invoke run.py")
    parser.add_argument("--task_timeout", type=int, default=3600, help="Maximum seconds per task; 0 disables the outer timeout")
    parser.add_argument("--dry_run", action="store_true", help="Validate and list selected tasks without invoking run.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 2
    if args.step_limit < 1 or args.task_timeout < 0:
        print("ERROR: --step_limit must be at least 1", file=sys.stderr)
        return 2
    if args.max_tasks is not None and args.max_tasks < 1:
        print("ERROR: --max_tasks must be at least 1", file=sys.stderr)
        return 2
    if not args.benchmark_path.is_dir():
        print(f"ERROR: benchmark path does not exist or is not a directory: {args.benchmark_path}", file=sys.stderr)
        return 2

    challenges = discover_challenges(args.benchmark_path.resolve(), args.category)
    if args.max_tasks is not None:
        challenges = challenges[: args.max_tasks]
    errors = environment_errors(args, challenges)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / STATUS_FILE_NAME
    summary_path = args.output_dir / SUMMARY_FILE_NAME
    status = load_status(status_path)
    selected = [task for task in challenges if not is_completed(status, args.output_dir, task)]

    print(f"Discovered {len(challenges)} task(s); {len(selected)} task(s) need execution.")
    print(
        f"Effective runtime configuration: workers={args.workers} step_limit={args.step_limit} "
        f"task_timeout={args.task_timeout}s inner_task_timeout={max(60, args.task_timeout - 60) if args.task_timeout else 1800}s",
        flush=True,
    )
    for task in challenges:
        state = "skip" if task not in selected else "run"
        print(f"[{state}] {task.task_id} ({task.relative_path})")
    if args.dry_run:
        atomic_write_json(summary_path, build_summary(challenges, status))
        return 0

    requests = [
        WorkerRequest(
            challenge=task,
            output_dir=str(args.output_dir.resolve()),
            model_name=args.model_name,
            image_name=args.image_name,
            config_file=args.config_file,
            step_limit=args.step_limit,
            python_executable=args.python_executable,
            task_timeout=args.task_timeout,
        )
        for task in selected
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_one_task, request): request.challenge for request in requests}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as error:
                log_path = args.output_dir / LOGS_DIR_NAME / f"{task.task_id}.error.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
                destination = persist_failure_trajectory(
                    args.output_dir,
                    task,
                    reason=str(error),
                    failure_category="worker_exception",
                )
                record = task_record(
                    task,
                    status="failed",
                    trajectory_path=str(destination),
                    time=utc_now(),
                    error=str(error),
                    error_code="worker_exception",
                    failure_category="task_not_executed",
                    error_log=str(log_path),
                )
            status["tasks"][task.relative_path] = record
            atomic_write_json(status_path, status)
            atomic_write_json(summary_path, build_summary(challenges, status))
            print(f"[{record['status']}] {task.task_id}")

    summary = build_summary(challenges, status)
    atomic_write_json(summary_path, summary)
    print(
        "Completed batch: "
        f"success={summary['success_count']}, failed={summary['failure_count']}, total={summary['total_tasks']}"
    )
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
