#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR COMMAND [ARG ...]" >&2
  exit 2
fi

output_dir=$1
shift
mkdir -p "$output_dir/logs"

pid_file="$output_dir/batch.pid"
log_file="$output_dir/logs/batch.log"
exit_file="$output_dir/exit_code"

if [[ -s "$pid_file" ]]; then
  old_pid=$(<"$pid_file")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "batch already running with pid $old_pid" >&2
    exit 1
  fi
fi

rm -f "$exit_file"
exit_file=$(cd "$(dirname "$exit_file")" && pwd)/$(basename "$exit_file")

# nohup keeps the worker alive after SSH disconnect. The wrapper records the
# command's exit code even when the command itself fails.
nohup bash -c '
  exit_path=$1
  shift
  "$@"
  code=$?
  printf "%s\n" "$code" > "$exit_path"
  exit "$code"
' _ "$exit_file" "$@" >>"$log_file" 2>&1 &

pid=$!
printf "%s\n" "$pid" > "$pid_file"
printf "pid=%s\nlog=%s\nexit_code=%s\n" "$pid" "$log_file" "$exit_file"
