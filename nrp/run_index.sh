#!/usr/bin/env bash
# Runs one line of the runs list inside an NRP Indexed Job pod, then uploads
# results/ to the object store. Line N (0-based, ignoring blank and # lines)
# goes to pod JOB_COMPLETION_INDEX=N.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_FILE="${RUNS_FILE:-/config/runs.txt}"
INDEX="${JOB_COMPLETION_INDEX:?not running as an Indexed Job}"
ARGS="$(grep -v -E '^\s*(#|$)' "$RUNS_FILE" | sed -n "$((INDEX + 1))p")"
if [[ -z "$ARGS" ]]; then
  echo "index $INDEX: no run listed (runs.txt shorter than completions), nothing to do"
  exit 0
fi

echo "index $INDEX: run_backtest.py $ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

# shellcheck disable=SC2086  # ARGS is a list of CLI flags
python scripts/run_backtest.py --device "${DEVICE:-cuda}" $ARGS
python scripts/nrp_results.py upload
