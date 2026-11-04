# Marimo Notebooks for TimeSeries Foundation Model experiments

Each notebook can be opened directly in [marimo online](https://marimo.app)
from GitHub. Online, the notebooks run in the browser (WASM), so cells that
need the local `data/` cache, `results/*.parquet`, or a GPU may not run there;
use the links to read and explore, and run locally for full results.

| Notebook | What it does | Open online |
| --- | --- | --- |
| [`00_data_tour.py`](00_data_tour.py) | Hourly H2S at three APCD stations plus the covariates the production XGBoost models use | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/00_data_tour.py) |
| [`01_baselines.py`](01_baselines.py) | Baselines and the production XGBoost reference every foundation model has to beat | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/01_baselines.py) |
| [`02_timestral.py`](02_timestral.py) | Mistral Timestral, zero-shot | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/02_timestral.py) |
| [`03_timesfm.py`](03_timesfm.py) | Google TimesFM 3.0, zero-shot | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/03_timesfm.py) |
| [`04_leaderboard.py`](04_leaderboard.py) | Paired leaderboard over `results/*.parquet`: do the foundation models clear the bar? | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/04_leaderboard.py) |

## Running locally

```bash
uv run marimo edit notebooks/00_data_tour.py
```
