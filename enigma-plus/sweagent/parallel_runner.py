"""Concurrent, isolated trajectory generation utilities."""
from __future__ import annotations

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
        futures = [pool.submit(invoke, job) for job in jobs]
        return [future.result() for future in as_completed(futures)]

