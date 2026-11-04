# Marimo Notebooks for TimeSeries Foundation Model experiments

Each notebook can be opened directly in [marimo online](https://marimo.app)
from GitHub. Online, the notebooks run in the browser (Pyodide/WASM). Each
notebook carries a [PEP 723](https://peps.python.org/pep-0723/) header that
lists its dependencies, including the `tsfm_h2s` wheel attached to the GitHub
release named there; marimo.app installs them on load. Torch, Timestral and
TimesFM are excluded from the wheel under Pyodide (environment markers in
`pyproject.toml`), so the data tour, baselines and leaderboard notebooks can
run online, while the Timestral and TimesFM notebooks are read-only there.
Cells that need the local `data/` cache or `results/*.parquet` still need a
local run.

After changing `src/tsfm_h2s`, rebuild and re-publish the wheel, bumping the
version in `pyproject.toml` and the notebook headers:

```bash
uv build --wheel
gh release create v0.1.0 dist/tsfm_h2s-0.1.0-py3-none-any.whl
```

| Notebook | What it does | Open online |
| --- | --- | --- |
| [`00_data_tour.py`](00_data_tour.py) | Hourly H2S at three APCD stations plus the covariates the production XGBoost models use | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/00_data_tour.py) |
| [`01_baselines.py`](01_baselines.py) | Baselines and the production XGBoost reference every foundation model has to beat | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/01_baselines.py) |
| [`02_timestral.py`](02_timestral.py) | Mistral Timestral, zero-shot | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/02_timestral.py) |
| [`03_timesfm.py`](03_timesfm.py) | Google TimesFM 3.0, zero-shot | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/03_timesfm.py) |
| [`04_leaderboard.py`](04_leaderboard.py) | Paired leaderboard over `results/*.parquet`: do the foundation models clear the bar? | [![Open in marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/TheResilientCollective/tj_h2s_foundation_models/blob/main/notebooks/04_leaderboard.py) |

## Running locally

```bash
uv run marimo edit notebooks

```
Select one of the notebooks above, and run it in the marimo editor. The notebooks are plain Python files, so you can also run them directly with `uv run python notebooks/00_data_tour.py` (or any other notebook).

