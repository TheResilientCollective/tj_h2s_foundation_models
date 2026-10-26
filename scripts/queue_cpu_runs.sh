#!/usr/bin/env bash
# Resumable CPU backtests for the first leaderboard: finished runs are skipped,
# and the weather-forecast archive caches per issuance, so re-running after an
# interruption picks up where it stopped.
set -u
cd "$(dirname "$0")/.."
run() { .venv/bin/python scripts/run_backtest.py --skip-existing "$@" 2>&1 | grep -E --line-buffered "wrote|exists|forecast archive|Error|Traceback" ; }

.venv/bin/python -c "
import pandas as pd
from tsfm_h2s import data
o = pd.DatetimeIndex(sorted(data.load_validation_store().origin.unique()))
by, iss = data.forecast_weather_for_origins(o)
print('weather archive ready:', iss.notna().sum(), 'of', len(o), 'origins covered', flush=True)
" 2>&1 | grep -E --line-buffered "forecast archive|ready|Error|Traceback"

run --model timestral-5M --mode covariates --weather forecast
run --model timestral-5M --mode multivariate_covariates --weather forecast
run --model timesfm --mode covariates --weather forecast --batch-size 32
run --model timesfm --mode covariates --weather oracle --batch-size 32
for lag in 6 24; do
  run --model timestral-5M --seed-lag $lag
  run --model climatology --seed-lag $lag
done
echo QUEUE_DONE
