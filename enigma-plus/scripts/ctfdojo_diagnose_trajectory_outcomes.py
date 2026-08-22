#!/usr/bin/env python3
"""Diagnose verifier evidence and terminal states in CTF-Dojo trajectories.

This tool never infers flag correctness from model prose. It backfills a state
record only when a trajectory or state contains a structured verifier result.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        first = next((line for line in text.splitlines() if line.strip()), "")
        value = json.loads(first)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def structured_verdict(info: dict[str, Any]) -> dict[str, Any]:
    """Return only a verifier-owned verdict; model text is deliberately ignored."""
    verified = info.get("flag_verified")
    evidence = info.get("flag_verification_evidence")
    if verified is True:
        return {"flag_submitted": True, "flag_verified": True, "flag_verification_evidence": evidence if isinstance(evidence, str) else "verifier accepted submission"}
    if verified is False:
        return {"flag_submitted": True, "flag_verified": False, "flag_verification_evidence": evidence if isinstance(evidence, str) else "verifier rejected submission"}
    # Legacy CTF trajectories set this exact state after validate_submission.
    if info.get("exit_status") == "submitted" and info.get("submission") not in (None, ""):
        return {"flag_submitted": True, "flag_verified": True, "flag_verification_evidence": "verifier accepted submission (legacy exit_status)"}
    if info.get("submission") not in (None, "") or str(info.get("exit_status", "")).startswith("submitted"):
        return {"flag_submitted": True, "flag_verified": None, "flag_verification_evidence": "unknown"}
    return {"flag_submitted": False, "flag_verified": None, "flag_verification_evidence": "unknown"}


def info_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("flag_submitted", "flag_verified", "flag_verification_evidence", "exit_status", "submission") if key in record}


def trajectory_paths(output_dir: Path, record: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    if isinstance(record.get("trajectory"), str):
        paths.append(Path(record["trajectory"]))
    task_id = record.get("task_id")
    if isinstance(task_id, str) and task_id:
        task_dir = output_dir / task_id
        paths.append(task_dir / "trajectory.jsonl")
        runner_dir = task_dir / "enigma_output"
        if runner_dir.is_dir():
            paths.extend(sorted(runner_dir.rglob("*.traj"), key=lambda item: item.stat().st_mtime, reverse=True))
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            result.append(path)
    return result


def compact_steps(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    result: list[dict[str, Any]] = []
    for step in trajectory[-limit:]:
        if not isinstance(step, dict):
            continue
        item = {key: step[key] for key in ("action", "state") if key in step}
        for key in ("observation", "response"):
            if isinstance(step.get(key), str):
                item[f"{key}_tail"] = step[key][-1200:]
        result.append(item)
    return result


LOG_PATTERNS = {
    "docker_network_address_exhausted": r"no available ipv4 addresses.*address pools",
    "model_generation_timeout": r"model generation exceeded .* timeout",
    "agent_subprocess_unexpected_exit": r"subprocess exited unexpectedly",
    "challenge_service_connection_failure": r"could not connect to 127\.0\.0\.1|connection refused",
    "runner_timeout": r"task timeout after|task execution timed out",
    "ctf_server_unavailable": r"ctf server.*(unavailable|not accessible|failed)",
}


def log_signals(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return []
    return [name for name, pattern in LOG_PATTERNS.items() if re.search(pattern, content, re.DOTALL)]


def merged_verdict(record: dict[str, Any], payloads: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, Any], list[str], bool]:
    sources: list[tuple[str, dict[str, Any]]] = [("state", info_from_record(record))]
    for path, payload in payloads:
        if isinstance(payload.get("info"), dict):
            sources.append((str(path), payload["info"]))
    verdicts = [(source, structured_verdict(info)) for source, info in sources]
    explicit = {value["flag_verified"] for _source, value in verdicts if value["flag_verified"] is not None}
    conflict = explicit == {True, False}
    if True in explicit:
        selected = next(value for _source, value in verdicts if value["flag_verified"] is True)
    elif False in explicit:
        selected = next(value for _source, value in verdicts if value["flag_verified"] is False)
    else:
        selected = next((value for _source, value in verdicts if value["flag_submitted"]), verdicts[0][1])
    return selected, [source for source, value in verdicts if value["flag_verified"] is not None], conflict


def exit_status_for(record: dict[str, Any], payloads: list[tuple[Path, dict[str, Any]]]) -> str | None:
    for _path, payload in payloads:
        info = payload.get("info")
        if isinstance(info, dict) and isinstance(info.get("exit_status"), str) and info["exit_status"]:
            return info["exit_status"]
    value = record.get("exit_status")
    return value if isinstance(value, str) and value else None


def diagnose(exit_status: str | None, signals: list[str], verdict: dict[str, Any], conflict: bool) -> str:
    if conflict:
        return "conflicting_structured_verifier_results"
    if verdict["flag_verified"] is True:
        return "backfilled_verifier_success"
    if verdict["flag_verified"] is False:
        return "verifier_rejected_submission"
    for signal, label in (
        ("docker_network_address_exhausted", "docker_network_address_exhausted"),
        ("agent_subprocess_unexpected_exit", "agent_subprocess_unexpected_exit"),
        ("model_generation_timeout", "model_generation_timeout"),
        ("challenge_service_connection_failure", "challenge_runtime_failure"),
        ("ctf_server_unavailable", "challenge_runtime_failure"),
    ):
        if signal in signals:
            return label
    if exit_status is None:
        return "missing_exit_status"
    if re.fullmatch(r"step_\d+_hit", exit_status):
        return "step_limit_hit_without_verifier"
    if exit_status == "early_exit":
        return "early_exit_environment_or_command_failure"
    if exit_status == "exit_format":
        return "model_output_format_failure"
    if exit_status.startswith("exit_"):
        return "agent_or_model_exit"
    return "terminal_state_without_verifier_evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--last_steps", type=int, default=5, choices=range(3, 6), metavar="3..5")
    parser.add_argument("--apply_state_backfill", action="store_true", help="Write only unambiguous structured verifier results to state.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "state.json"
    state = read_json(state_path)
    if state is None or not isinstance(state.get("tasks"), dict):
        raise SystemExit(f"ERROR: invalid state file: {state_path}")

    reports: list[dict[str, Any]] = []
    changed = 0
    for relative_path, record in state["tasks"].items():
        if not isinstance(record, dict) or not record.get("trajectory_generated", record.get("status") == "success"):
            continue
        if record.get("flag_verified") is not None:
            continue
        payloads = [(path, payload) for path in trajectory_paths(output_dir, record) if (payload := read_json(path)) is not None]
        verdict, verdict_sources, conflict = merged_verdict(record, payloads)
        exit_status = exit_status_for(record, payloads)
        primary_payload = payloads[0][1] if payloads else {}
        configured_log = record.get("runner_log")
        run_log = Path(configured_log) if isinstance(configured_log, str) else output_dir / "logs" / f"{record.get('task_id', '')}.run.log"
        signals = log_signals(run_log)
        report = {
            "relative_path": relative_path,
            "task_id": record.get("task_id"),
            "task": record.get("task"),
            "category": record.get("category", "unknown"),
            "state_exit_status": record.get("exit_status"),
            "resolved_exit_status": exit_status,
            "trajectory_sources": [str(path) for path, _payload in payloads],
            "structured_verdict_sources": verdict_sources,
            "log_signals": signals,
            "diagnosis": diagnose(exit_status, signals, verdict, conflict),
            "flag_submitted": verdict["flag_submitted"],
            "flag_verified": verdict["flag_verified"],
            "flag_verification_evidence": verdict["flag_verification_evidence"],
            "verdict_conflict": conflict,
            "last_steps": compact_steps(primary_payload, args.last_steps),
        }
        reports.append(report)
        if args.apply_state_backfill and not conflict and verdict["flag_verified"] is not None:
            record.update(verdict)
            if exit_status and not record.get("exit_status"):
                record["exit_status"] = exit_status
            record["updated_at"] = utc_now()
            changed += 1

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    detail_path = logs_dir / "trajectory-terminal-diagnosis.jsonl"
    detail_path.write_text("".join(json.dumps(report, ensure_ascii=False) + "\n" for report in reports), encoding="utf-8")
    by_category_and_diagnosis: dict[str, dict[str, int]] = {}
    for category in sorted({report["category"] for report in reports}):
        by_category_and_diagnosis[category] = dict(sorted(collections.Counter(report["diagnosis"] for report in reports if report["category"] == category).items()))
    summary = {
        "generated_at": utc_now(),
        "unknown_records_analyzed": len(reports),
        "state_records_backfilled": changed,
        "by_diagnosis": dict(sorted(collections.Counter(report["diagnosis"] for report in reports).items())),
        "by_category": dict(sorted(collections.Counter(report["category"] for report in reports).items())),
        "by_category_and_diagnosis": by_category_and_diagnosis,
        "detail_file": str(detail_path),
    }
    summary_path = logs_dir / "trajectory-terminal-diagnosis-summary.json"
    write_json(summary_path, summary)
    if args.apply_state_backfill and changed:
        write_json(state_path, state)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
