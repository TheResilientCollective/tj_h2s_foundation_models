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
#     "tsfm_h2s @ https://github.com/TheResilientCollective/tj_h2s_foundation_models/releases/download/v0.1.0/tsfm_h2s-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from tsfm_h2s import backtest, compare, data, metrics, plots
    from tsfm_h2s.models import HourOfDayClimatology, Persistence, SeasonalNaive

    return (
        HourOfDayClimatology,
        Persistence,
        SeasonalNaive,
        backtest,
        compare,
        data,
        metrics,
        mo,
        pd,
        plots,
    )


@app.cell
def _(mo):
    mo.md("""
    # 01 · Baselines and the production XGBoost reference

    The bar a foundation model has to clear. Every model is scored on the
    **same forecast origins as the production products** (`run_ts` in the
    validation store), so rows pair up hour-for-hour with the deployed
    XGBoost Lean / Evidence forecasts.

    * persistence · seasonal-naive (same hour yesterday) · 28-day hour-of-day
      climatology (a probabilistic baseline)
    * `xgboost_lean` / `xgboost_evidence` straight from the validation store
    """)
    return


@app.cell
def _(data):
    obs = data.load_obs()
    panel = data.fill_covariates(data.to_panel(obs))
    validation = data.load_validation_store()
    return panel, validation


@app.cell
def _(
    HourOfDayClimatology,
    Persistence,
    SeasonalNaive,
    backtest,
    panel,
    pd,
    validation,
):
    origins = backtest.production_origins(validation)
    cfg = backtest.BacktestConfig()
    baseline_runs = pd.concat(
        [backtest.run_backtest(m, panel, origins, cfg, progress=False)
         for m in (Persistence(), SeasonalNaive(), HourOfDayClimatology())],
        ignore_index=True,
    )
    reference = pd.concat([backtest.production_reference(validation, v) for v in ("lean", "evidence")], ignore_index=True)
    return baseline_runs, origins, reference


@app.cell
def _(baseline_runs, compare, mo, origins, pd, reference):
    allruns = pd.concat([baseline_runs, reference], ignore_index=True)
    allruns["label"] = compare.label(allruns)
    paired_runs = compare.paired(allruns)
    mo.md(
        f"**{len(origins)} production origins** "
        f"({origins.min():%Y-%m-%d} → {origins.max():%Y-%m-%d}); "
        f"{paired_runs.groupby('label').size().iloc[0]:,} paired (origin, station, lead) cells per model."
    )
    return (paired_runs,)


@app.cell
def _(metrics, mo, paired_runs):
    cols = ["label", "product", "n", "mae", "spearman", "tolerant_acc", "dangerous_miss", "smell_miss",
            "mean_cost", "n_ge10", "recall10", "far10", "auc10", "brier10", "n_ge30", "recall30", "auc30"]
    summary = metrics.summarize(paired_runs, by=["label", "product"])[cols].round(3)
    mo.vstack([
        mo.md("## Paired scores by product (lower `mean_cost` is better; `auc` is threshold-free)"),
        mo.ui.table(summary.sort_values(["product", "mean_cost"]), selection=None),
    ])
    return


@app.cell
def _(metrics, paired_runs, plots):
    import matplotlib.pyplot as _plt

    by_lead = metrics.summarize(paired_runs, by=["label", "lead_hour"])
    _fig, _axes = _plt.subplots(1, 3, figsize=(15, 3.6))
    for _ax, _m in zip(_axes, ["spearman", "auc10", "mean_cost"]):
        plots.skill_by_lead(by_lead, _m, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Reading this

    * The validation store only spans summer–fall 2026, the quiet season, so
      ≥30 ppb counts are small. Treat `recall30` as anecdotal here and use the
      extended (winter) backtest in `04_leaderboard` for hazard conclusions.
    * `recall@0.5` punishes models whose probabilities are calibrated but
      low for rare events; `auc` and the reliability diagram in
      `04_leaderboard` separate *ranking* skill from the choice of cutoff.
    """)
    return


if __name__ == "__main__":
    app.run()
