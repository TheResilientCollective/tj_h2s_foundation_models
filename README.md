# tj_h2s_foundation_models

Can time-series **foundation models** forecast hydrogen sulfide (H2S) in the
Tijuana River Valley? This repo backtests
[Mistral Timestral](https://github.com/mistralai/timestral) and
[Google TimesFM 3.0](https://github.com/google-research/timesfm), zero-shot,
against simple baselines and the production per-station XGBoost models from
[`tj_h2s_prediction`](https://github.com/TheResilientCollective/tj_h2s_prediction),
on the **same forecast origins** and with the **same hazard rubric**.

See **[PLAN.md](PLAN.md)** for the question, protocol, model matrix and go/no-go criteria.

## Layout

```
PLAN.md                          experiment design + decision criteria
src/tsfm_h2s/
  data.py                        public S3 loaders (obs + covariates, production validation store)
  backtest.py                    rolling-origin harness (leads 1-24, production run_ts semantics)
  metrics.py                     P(>k) from quantiles, asymmetric rubric, recall/FAR/AUC/Brier, bootstrap
  compare.py                     load results/*.parquet, paired alignment
  plots.py                       fan charts, skill-by-lead, reliability
  models/                        baselines + Timestral + TimesFM adapters (one interface)
scripts/run_backtest.py          CLI: one model/config -> results/<tag>.parquet
notebooks/                       marimo notebooks (plain .py, git-friendly)
  00_data_tour.py                coverage, heavy tail, diurnal cycle, seasonality
  01_baselines.py                baselines vs production XGBoost on production origins
  02_timestral.py                interactive Timestral slices (size/mode/context)
  03_timesfm.py                  interactive TimesFM slices
  04_leaderboard.py              paired leaderboard, reliability, bootstrap vs XGBoost, event case studies
colab/run_backtest_colab.ipynb   GPU runner for the heavy configs -> results to Google Drive
tests/                           leakage guard, baselines, P(>k), rubric parity
```

## Quickstart

```bash
uv sync --extra timestral --extra timesfm --extra dev   # CPU torch is fine for 5M/28M
uv run pytest

# notebooks
uv run marimo edit notebooks/00_data_tour.py

# full runs (write results/*.parquet, then open 04_leaderboard)
uv run python scripts/run_backtest.py --model climatology
uv run python scripts/run_backtest.py --model timestral-5M --mode covariates
uv run python scripts/run_backtest.py --model timesfm --mode covariates --device cuda
uv run python scripts/run_backtest.py --model timestral-28M --origins extended --start 2025-10-01
```

On Linux without a GPU, add
`--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu`
to `uv pip install` to avoid pulling CUDA wheels.

Heavy configurations (Timestral-123M/209M, TimesFM full backtests, 8k contexts)
run in Colab: open `colab/run_backtest_colab.ipynb`, choose a GPU runtime, and
add a `GH_TOKEN` secret. Copy the parquet files from Drive into `results/`
locally to include them in the leaderboard.

## Data

Everything comes from the public `resilentpublic` bucket. No credentials are
needed. Files are cached in `data/`.

* `latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet`: hourly H2S + weather/tide/flow/SBIWTP, 3 stations, 2023-11 → present
* `tijuana/forecast/validation_store/validation.parquet`: production XGBoost products (Lean/Evidence, leads 1–24) joined to measured H2S, 2026-06-28 → present

## Licensing notes

* Timestral: code Apache-2.0; weights are Modified MIT (see the HF model card).
* TimesFM 3.0: code Apache-2.0; **self-hosted weights are non-commercial /
  non-production**. Research use only here. For production, use BigQuery ML or
  the Apache-2.0 TimesFM 2.5 weights.
