#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only batch diagnosis. The report is deliberately created in the
# caller's current directory so it can be collected from an SSH session.
DEFAULT_OUTPUT_DIR="/data2/nfs/wangyingqi/Cyber-Zero/trajectories/ctfdojo_3traj_20260830_144722/traj_001"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"
REPORT_PATH="${2:-$PWD/ctfdojo_diagnosis_$(date -u +%Y%m%dT%H%M%SZ).txt}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: output directory does not exist: $OUTPUT_DIR" >&2
  exit 2
fi

REPORT_PATH="$(realpath -m "$REPORT_PATH")"
mkdir -p "$(dirname "$REPORT_PATH")"

# Print everything and save the exact same output in the current directory.
exec > >(tee "$REPORT_PATH") 2>&1

echo "CTF-Dojo full diagnosis"
echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "output_dir=$(realpath "$OUTPUT_DIR")"
echo "report_path=$REPORT_PATH"
echo

if [[ -f "$OUTPUT_DIR/state.json" || -f "$OUTPUT_DIR/summary.json" ]]; then
  python3 - "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1]).resolve()

def load_json(path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Prefer a regular JSON document, then support JSONL batch files.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{path}: invalid JSON: {exc}")
    return None

def show_json_file(name):
    path = root / name
    if not path.is_file():
        print(f"{name}: MISSING")
        return None
    value = load_json(path)
    print(f"{name}: {'OK' if value is not None else 'INVALID'} ({path})")
    if isinstance(value, dict):
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)[:12000])
    return value

print("=== SUMMARY FILES ===")
summary = show_json_file("summary.json")
state = show_json_file("state.json")
print()

tasks = state.get("tasks") if isinstance(state, dict) else None
if not isinstance(tasks, dict):
    print("No state.tasks dictionary found; task-level analysis skipped.")
    raise SystemExit(0)

def value(record, *keys, default=None):
    for key in keys:
        item = record.get(key)
        if item not in (None, ""):
            return item
    return default

def text_of(record):
    parts = []
    for key in ("error", "error_message", "exception", "failure_reason", "message"):
        item = record.get(key)
        if item not in (None, ""):
            parts.append(str(item))
    return " | ".join(parts)

def generated(record):
    return bool(record.get("trajectory_generated")) or record.get("status") in {"success", "generated"}

def classify(record):
    raw = " ".join(str(record.get(k, "")) for k in (
        "error_code", "failure_category", "run_status", "exit_status", "status",
        "error", "error_message", "exception", "failure_reason", "message"))
    low = raw.lower()
    if any(x in low for x in ("timeout", "timed out", "deadline exceeded")):
        return "timeout"
    if any(x in low for x in ("format", "parse", "json", "schema", "structured output")):
        return "model_output_or_parse"
    if any(x in low for x in ("api", "rate limit", "429", "gateway", "connection", "messages")):
        return "model_api_or_transport"
    if any(x in low for x in ("docker", "container", "network", "environment", "setup")):
        return "environment_or_docker"
    if any(x in low for x in ("worker", "future", "process", "executor", "subprocess", "killed", "oom")):
        return "worker_or_host"
    if record.get("exit_status"):
        return "terminal_without_success"
    return "unknown"

def dist(label, values):
    print(label)
    counts = Counter("<missing>" if x in (None, "") else str(x) for x in values)
    for key, count in counts.most_common():
        print(f"  {count:6d}  {key}")
    print()

records = [record for record in tasks.values() if isinstance(record, dict)]
dist("=== RUN STATUS DISTRIBUTION ===", [r.get("run_status") for r in records])
dist("=== EPISODE STATUS DISTRIBUTION ===", [r.get("episode_status") for r in records])
dist("=== EXIT STATUS DISTRIBUTION ===", [r.get("exit_status") for r in records])
dist("=== ERROR CODE DISTRIBUTION ===", [r.get("error_code") for r in records])
dist("=== FAILURE CATEGORY DISTRIBUTION ===", [r.get("failure_category") for r in records])
dist("=== HEURISTIC FAILURE CLASSIFICATION ===", [classify(r) for r in records if not generated(r)])

failed = [r for r in records if not generated(r) and any(r.get(k) not in (None, "") for k in (
    "error", "error_code", "error_message", "exception", "failure_reason", "run_status", "exit_status"))]
print("=== FAILED TASKS ===")
print(f"records={len(records)} generated={sum(generated(r) for r in records)} failed_or_terminal={len(failed)}")
for index, record in enumerate(failed, 1):
    task_id = value(record, "task_id", "id", default="<no-task-id>")
    name = value(record, "task", "name", "instance_id", default="<no-name>")
    category = value(record, "category", default="<no-category>")
    status = value(record, "run_status", "exit_status", "error_code", default="<no-status>")
    detail = text_of(record)[:700]
    print(f"{index:4d}. task_id={task_id} category={category} status={status} task={name}")
    if detail:
        print(f"      detail={detail}")
print()

print("=== GENERATED / SUBMITTED / VERIFIED ===")
for label, predicate in (
    ("generated", generated),
    ("submitted", lambda r: r.get("flag_submitted") is True),
    ("verified", lambda r: r.get("flag_verified") is True),
):
    print(f"{label}={sum(predicate(r) for r in records)}")
print()

print("=== ERROR LOG REFERENCES ===")
seen = set()
for record in records:
    for key in ("error_log", "runner_log", "log_file", "stderr_log"):
        item = record.get(key)
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            path = Path(item)
            if not path.is_absolute():
                path = root / path
            print(f"{'EXISTS' if path.is_file() else 'MISSING'}  {path}")
print()

print("=== STATE RESOURCE FIELDS ===")
for key in ("attempt_resource_cleanup_failed", "attempt_resource_cleanup_removed",
            "container_cleanup_failed", "container_cleanup_removed",
            "remaining_containers", "remaining_networks", "remaining_volumes",
            "docker_bridge_containers_at_end", "docker_dynamic_networks_at_end"):
    if isinstance(state, dict) and key in state:
        print(f"{key}={state[key]}")
print()

print("=== RECENT TASK LOG SIGNALS ===")
patterns = {
    "timeout": ("timeout", "timed out", "deadline exceeded"),
    "api_or_rate_limit": ("429", "rate limit", "gateway", "messages api", "api error"),
    "format_or_parse": ("parse", "json", "schema", "format", "structured"),
    "docker_or_network": ("docker", "network", "active endpoints", "container"),
    "oom_or_killed": ("oom", "out of memory", "killed"),
}
log_files = sorted({p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".log", ".txt", ".err", ".out"}}, key=lambda p: p.stat().st_mtime, reverse=True)
totals = Counter()
for path in log_files:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        continue
    hits = [name for name, needles in patterns.items() if any(needle in content for needle in needles)]
    if hits:
        for name in hits:
            totals[name] += 1
        print(f"{path}: {', '.join(hits)}")
        for line in content.splitlines():
            if any(needle in line for needles in patterns.values() for needle in needles):
                print(f"  {line[:500]}")
                break
print(f"log_signal_files={sum(totals.values())} by_signal={dict(totals)}")
PY
else
  echo "No state.json or summary.json found under $OUTPUT_DIR"
fi

echo
echo "=== CURRENT PROCESS ==="
if command -v pgrep >/dev/null 2>&1; then
  pgrep -a -f 'ctfdojo|batch_generate|parallel_runner|sweagent|python' || true
else
  ps -ef | rg -i 'ctfdojo|batch_generate|parallel_runner|sweagent|python' || true
fi

echo
echo "=== DOCKER RESOURCE SNAPSHOT ==="
if command -v docker >/dev/null 2>&1; then
  echo "-- containers --"
  docker ps -a --format '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Networks}}' || true
  echo "-- ctf networks --"
  docker network ls --format '{{.ID}}\t{{.Name}}\t{{.Driver}}' | awk '$2 ~ /^ctfnet-|^batch-/ {print}' || true
  echo "-- active endpoints for ctf networks --"
  while read -r network_id _; do
    [[ -z "$network_id" ]] && continue
    docker network inspect "$network_id" --format '{{.Name}} endpoints={{json .Containers}}' 2>&1 || true
  done < <(docker network ls --format '{{.ID}} {{.Name}}' | awk '$2 ~ /^ctfnet-|^batch-/ {print $1, $2}')
else
  echo "docker command not found"
fi

echo
echo "=== HOST OOM CHECK ==="
if command -v dmesg >/dev/null 2>&1; then
  dmesg -T 2>/dev/null | rg -i 'out of memory|oom-killer|killed process|memory cgroup' | tail -n 80 || true
fi
if command -v journalctl >/dev/null 2>&1; then
  journalctl -k --no-pager -n 300 2>/dev/null | rg -i 'out of memory|oom-killer|killed process|memory cgroup' | tail -n 80 || true
fi

echo
echo "=== REPORT COMPLETE ==="
echo "$REPORT_PATH"
