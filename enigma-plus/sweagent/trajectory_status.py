"""Canonical trajectory outcome handling.

The runner's ``exit_status`` describes the mechanism that terminated an
episode.  This module provides the separate, stable outcome consumed by
trajectory evaluators and batch reporting.
"""

from typing import Any, Dict, Iterable, Optional


SUCCESS = "SUCCESS"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
ERROR = "ERROR"
UNKNOWN = "UNKNOWN"
OUTCOMES = frozenset((SUCCESS, FAILED, TIMEOUT, ERROR, UNKNOWN))


def is_verified_success(info: Dict[str, Any]) -> bool:
    """Only an explicit verifier result can make an episode successful."""
    return info.get("flag_verified") is True


def derive_status(
    info: Dict[str, Any],
    *,
    run_completed: bool = True,
    trajectory: Optional[Iterable[Dict[str, Any]]] = None,
) -> str:
    """Derive a canonical outcome without inspecting model prose.

    ``UNKNOWN`` is reserved for incomplete/corrupt artifacts.  A completed
    run without a verifier success is a failure, even when it produced a
    valid trajectory file.
    """
    if is_verified_success(info):
        return SUCCESS

    exit_status = str(info.get("exit_status") or "")
    error_statuses = {
        "error",
        "exit_error",
        "exit_api",
        "exit_context",
        "exit_cost",
        "exit_format",
        "ctf_server_crashed",
        "ctf_server_unavailable",
        "verifier_error",
        "model_generation_error",
    }
    timeout_statuses = {
        "task_timeout",
        "runner_timeout",
        "model_generation_timeout",
    }
    failed_statuses = {"skipped", "exit_forfeit", "step_limit_hit"}

    if exit_status in timeout_statuses or info.get("timeout_reason"):
        return TIMEOUT
    if exit_status in error_statuses or exit_status.startswith("error"):
        return ERROR
    if exit_status.startswith("step_") and exit_status.endswith("_hit"):
        return FAILED
    if exit_status in failed_statuses or exit_status.startswith("exit_"):
        return FAILED
    if info.get("flag_verified") is False:
        return FAILED if run_completed else UNKNOWN
    if run_completed:
        # A normal end without a verifier success is not a solve.
        if exit_status or trajectory is not None:
            return FAILED
        return UNKNOWN
    return UNKNOWN


def finalize_info(
    info: Dict[str, Any],
    *,
    run_completed: bool = True,
    terminal_reason: Optional[str] = None,
    trajectory: Optional[Iterable[Dict[str, Any]]] = None,
) -> str:
    """Write the canonical status and preserve the low-level exit status."""
    status = derive_status(
        info,
        run_completed=run_completed,
        trajectory=trajectory,
    )
    info["status"] = status
    if terminal_reason:
        info["terminal_reason"] = terminal_reason
    elif "terminal_reason" not in info:
        info["terminal_reason"] = status.lower()
    info.setdefault("flag_submitted", False)
    info.setdefault("flag_verified", None)
    info.setdefault("flag_verification_evidence", "no verifier evidence")
    return status
