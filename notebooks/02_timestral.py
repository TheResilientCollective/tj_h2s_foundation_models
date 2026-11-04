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
    import pandas as pd

    from tsfm_h2s import backtest, compare, data, metrics, plots
    from tsfm_h2s.models import HourOfDayClimatology

    return (
        HourOfDayClimatology,
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
    # 02 · Timestral, zero-shot

    [Mistral Timestral](https://github.com/mistralai/timestral): patch
    transformer, 21 quantiles (0.01–0.99), context up to 32 768 h,
    multivariate targets, past-only and known-future covariates. Runs on CPU
    (5M/28M take seconds; 123M/209M are better on a GPU via
    `colab/run_backtest_colab.ipynb`).

    Experiment ladder (PLAN §5): **univariate → + covariates →
    3 stations jointly (+ covariates)**, then a size and context sweep.
    The controls below run a quick slice interactively. Full runs go through
    `scripts/run_backtest.py` and are compared in `04_leaderboard`.
    """)
    return


@app.cell
def _(data):
    panel = data.fill_covariates(data.to_panel(data.load_obs()))
    validation = data.load_validation_store()
    return panel, validation


@app.cell
def _(mo):
    size = mo.ui.dropdown(["5M", "28M", "123M", "209M"], value="5M", label="model size")
    mode = mo.ui.dropdown(["univariate", "covariates", "multivariate", "multivariate_covariates"],
                          value="covariates", label="mode")
    context = mo.ui.dropdown(["512", "2048", "8192"], value="2048", label="context (h)")
    transform = mo.ui.dropdown(["log1p", "none"], value="log1p", label="target transform")
    n_origins = mo.ui.slider(20, 711, value=120, step=20, label="production origins (latest N)")
    device = mo.ui.dropdown(["cpu", "cuda"], value="cpu", label="device")
    go = mo.ui.run_button(label="Run slice")
    mo.vstack([mo.hstack([size, mode, context, transform]), mo.hstack([n_origins, device, go])])
    return context, device, go, mode, n_origins, size, transform


@app.cell
def _(device, size):
    from tsfm_h2s.models.timestral import Timestral

    model = Timestral(size=size.value, device=device.value)
    return (model,)


@app.cell
def _(
    HourOfDayClimatology,
    backtest,
    compare,
    context,
    go,
    mo,
    mode,
    model,
    n_origins,
    panel,
    pd,
    transform,
    validation,
):
    mo.stop(not go.value, mo.md("Choose a configuration and press **Run slice**."))
    origins = backtest.production_origins(validation)[-n_origins.value :]
    cfg = backtest.BacktestConfig(mode=mode.value, context_length=int(context.value), transform=transform.value)
    run = backtest.run_backtest(model, panel, origins, cfg, progress=False)
    clim = backtest.run_backtest(HourOfDayClimatology(), panel, origins, backtest.BacktestConfig(), progress=False)
    ref = backtest.production_reference(validation, "lean")
    slice_runs = pd.concat([run, clim, ref[ref["origin"].isin(origins)]], ignore_index=True)
    slice_runs["label"] = compare.label(slice_runs)
    slice_runs = compare.paired(slice_runs)
    return (slice_runs,)


@app.cell
def _(metrics, mo, slice_runs):
    _cols = ["label", "product", "n", "mae", "spearman", "mean_cost", "n_ge10", "recall10", "auc10", "brier10", "n_ge30", "auc30"]
    mo.ui.table(metrics.summarize(slice_runs, by=["label", "product"])[_cols].round(3), selection=None)
    return


@app.cell
def _(mo, slice_runs):
    _peaks = (slice_runs.groupby(["origin", "station"])["actual_h2s"].max()
              .sort_values(ascending=False).head(15).reset_index())
    event = mo.ui.dropdown(
        {f"{r.origin:%Y-%m-%d %H}Z · {r.station} · peak {r.actual_h2s:.0f} ppb": (r.origin, r.station)
         for r in _peaks.itertuples()},
        label="largest events in slice",
    )
    event
    return (event,)


@app.cell
def _(event, mo, panel, plots, slice_runs):
    mo.stop(event.value is None)
    plots.fan_chart(slice_runs, panel, *event.value).figure
    return


if __name__ == "__main__":
    app.run()
