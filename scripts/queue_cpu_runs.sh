#!/usr/bin/env bash
# Sequential CPU backtests used for the first leaderboard.
set -u
run() { .venv/bin/python scripts/run_backtest.py "$@" 2>&1 | grep -E "wrote|Error|Traceback" ; }
run --model timesfm --mode univariate --batch-size 32
run --model timesfm --mode covariates --batch-size 32
for lag in 6 24; do
  run --model timestral-5M --seed-lag $lag
  run --model climatology --seed-lag $lag
done
echo QUEUE_DONE
