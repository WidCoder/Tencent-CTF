"""Concurrent, isolated trajectory generation utilities."""
from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def generate_trajectories(
    tasks: list[dict[str, Any]],
    run_one: Callable[[dict[str, Any], Path, int], Any],
    *,
    output_dir: str | Path,
    workers: int = 1,
    trajectories_per_task: int = 1,
) -> list[Any]:
    """Run independent task/trajectory jobs with isolated output directories."""
    root = Path(output_dir)
    jobs = [(task, n) for task in tasks for n in range(1, max(1, trajectories_per_task) + 1)]

    def invoke(job: tuple[dict[str, Any], int]) -> Any:
        task, n = job
        task_id = str(task.get("instance_id", task.get("task_id", "task")))
        run_dir = root / task_id / f"traj_{n:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_one(task, run_dir, n)

    if workers <= 1:
        return [invoke(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ctf-agent") as pool:
        future_jobs = {pool.submit(invoke, job): job for job in jobs}
        results: list[Any] = []
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                results.append(future.result())
            except BaseException as error:
                task, n = job
                task_id = str(task.get("instance_id", task.get("task_id", "task")))
                run_dir = root / task_id / f"traj_{n:03d}"
                payload = {
                    "trajectory_schema_version": 2,
                    "artifact_type": "synthetic_error",
                    "trajectory": [{
                        "step_id": 1,
                        "action": "runner_exception",
                        "observation": str(error),
                        "terminal": True,
                        "error": type(error).__name__,
                    }],
                    "info": {
                        "exit_status": "runner_exception",
                        "failure_category": "worker_exception",
                        "error_type": type(error).__name__,
                        "traceback": traceback.format_exc(),
                        "total_steps": 1,
                    },
                }
                destination = run_dir / "trajectory.json"
                temporary = run_dir / f".trajectory.{os.getpid()}.{n}.tmp"
                try:
                    with temporary.open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle, ensure_ascii=False)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                results.append({
                    "status": "failed",
                    "task_id": task_id,
                    "trajectory_path": str(destination),
                    "error": str(error),
                })
        return results

