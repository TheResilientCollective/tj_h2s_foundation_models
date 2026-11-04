# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas>=2.2",
#     "pyarrow",
#     "matplotlib",
#     "scipy",
#     "requests",
#     "tzdata",
#     "tsfm_h2s @ https://raw.githubusercontent.com/TheResilientCollective/tj_h2s_foundation_models/main/wheels/tsfm_h2s-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from tsfm_h2s import data

    return data, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # 00 · Data tour — what are we asking a foundation model to forecast?

    Hourly H2S at three APCD stations plus the covariates the production
    XGBoost models use. Everything loads from the public `resilentpublic`
    bucket (cached under `data/`). See `PLAN.md` for the experiment design.
    """)
    return


@app.cell
def _(data):
    obs = data.load_obs()
    panel = data.to_panel(obs)
    return (panel,)


@app.cell
def _(data, mo, panel, pd):
    rows = []
    for _s, _g in panel.items():
        _y = _g[data.TARGET]
        _first = _y.first_valid_index()
        _y = _y.loc[_first:]
        rows.append({
            "station": _s,
            "first obs": _first,
            "last obs": _y.last_valid_index(),
            "hours": len(_y),
            "coverage": round(_y.notna().mean(), 3),
            "median ppb": _y.median(),
            "p95 ppb": round(_y.quantile(0.95), 1),
            "max ppb": _y.max(),
            "hours ≥10": int((_y >= 10).sum()),
            "hours ≥30": int((_y >= 30).sum()),
        })
    coverage = pd.DataFrame(rows)
    mo.vstack([mo.md("## Coverage & hazard counts"), mo.ui.table(coverage, selection=None)])
    return


@app.cell
def _(data, np, panel, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for _s, _g in panel.items():
        _y = _g[data.TARGET].dropna()
        axes[0].hist(np.log1p(_y.clip(lower=0)), bins=80, histtype="step", label=_s, density=True)
        _prof = _y.groupby(_y.index.tz_convert("America/Los_Angeles").hour).mean()
        axes[1].plot(_prof.index, _prof.values, marker=".", label=_s)
    axes[0].set_xlabel("log1p(H2S ppb)")
    axes[0].set_title("Heavy tail: the model sees mostly ~1 ppb")
    axes[1].set_xlabel("local hour")
    axes[1].set_ylabel("mean H2S (ppb)")
    axes[1].set_title("Diurnal cycle (nights are worst)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig
    return


@app.cell
def _(data, panel, pd, plt):
    _monthly = pd.DataFrame({_s: (_g[data.TARGET] >= 10).resample("MS").sum() for _s, _g in panel.items()})
    _monthly.index = [d.strftime("%Y-%m") for d in _monthly.index]  # plain labels: a tz-aware period axis is not plottable everywhere
    _ax = _monthly.plot(kind="bar", figsize=(12, 3.5), width=0.85)
    _ax.tick_params(axis="x", rotation=90, labelsize=7)
    _ax.set_ylabel("hours ≥10 ppb")
    _ax.set_title("Seasonality of smell-level hours — note which months the production validation store covers")
    plt.tight_layout()
    _ax.figure
    return


@app.cell
def _(mo, panel):
    station = mo.ui.dropdown(list(panel), value="NESTOR - BES", label="station")
    days = mo.ui.slider(3, 60, value=14, label="days shown")
    end = mo.ui.date(label="window end")
    mo.hstack([station, days, end])
    return days, end, station


@app.cell
def _(data, days, end, panel, pd, plt, station):
    _g = panel[station.value]
    _end = pd.Timestamp(end.value, tz="UTC") if end.value else _g[data.TARGET].last_valid_index()
    _w = _g.loc[_end - pd.Timedelta(days=days.value) : _end]
    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    _a1.plot(_w.index, _w[data.TARGET], color="black", lw=1)
    for _k, _c in ((5, "#d4b106"), (10, "#e8890c"), (30, "#d4380d")):
        _a1.axhline(_k, color=_c, ls="--", lw=0.8)
    #_a1.set_yscale("symlog", linthresh=5)
    _a1.set_yscale("linear")
    _a1.set_ylabel("H2S ppb")
    _a2.plot(_w.index, _w["wind_speed_10m"], label="wind speed")
    _a2b = _a2.twinx()
    _a2b.plot(_w.index, _w["tide_height"], color="C1", alpha=0.6, label="tide")
    _a2.set_ylabel("wind speed")
    _a2b.set_ylabel("tide (m)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Covariate roles for the TSFMs

    * **known-future** (`data.KNOWN_FUTURE_COVARIATES`): weather, tide,
      calendar — available over the horizon in operations from forecasts.
      In the backtest they are *observed* ⇒ an **oracle-weather** upper bound.
    * **past-only** (`data.PAST_ONLY_COVARIATES`): border flow + SBIWTP
      flow/anomaly/deficit.

    Gaps in the target stay NaN (both models accept missing values);
    covariates are interpolated / filled (`data.fill_covariates`).
    """)
    return


if __name__ == "__main__":
    app.run()
