import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from tsfm_h2s import compare, data, metrics, plots

    return compare, data, metrics, mo, pd, plots, plt


@app.cell
def _(mo):
    mo.md("""
    # 04 · Leaderboard: do the foundation models clear the bar?

    Reads every `results/*.parquet` written by `scripts/run_backtest.py`
    (locally or downloaded from Colab) and scores them **paired**: only
    (origin, station, lead) cells that every selected model forecast.

    Go / no-go criteria (PLAN §7), on production origins vs `xgboost_lean`:
    recall@10 at leads 1–6 with ≤ +2 pp FAR · ≥10% lower mean rubric cost
    (95% CI excludes 0) · or better Brier when ensembled.
    """)
    return


@app.cell
def _(mo):
    origin_set = mo.ui.radio(["production", "extended"], value="production", label="origin set")
    origin_set
    return (origin_set,)


@app.cell
def _(compare, mo, origin_set):
    results = compare.load_results(origin_set.value)
    mo.stop(results.empty, mo.md(f"No `results/*__{origin_set.value}.parquet` yet. Run `scripts/run_backtest.py`."))
    picker = mo.ui.multiselect(sorted(results["label"].unique()), value=sorted(results["label"].unique()), label="models")
    picker
    return picker, results


@app.cell
def _(compare, mo, picker, results):
    mo.stop(len(picker.value) < 1)
    paired_runs = compare.paired(results, list(picker.value))
    mo.md(f"{paired_runs.groupby('label').size().min():,} paired cells per model; "
          f"{paired_runs['origin'].nunique()} origins")
    return (paired_runs,)


@app.cell
def _(metrics, mo, paired_runs):
    _cols = ["label", "product", "n", "mae", "mae_log", "spearman", "tolerant_acc", "dangerous_miss", "smell_miss",
             "mean_cost", "pinball", "n_ge10", "recall10", "far10", "auc10", "brier10",
             "n_ge30", "recall30", "far30", "auc30", "brier30"]
    leaderboard = metrics.summarize(paired_runs, by=["label", "product"])
    mo.vstack([mo.md("## By product"),
               mo.ui.table(leaderboard[[c for c in _cols if c in leaderboard]].round(3)
                           .sort_values(["product", "mean_cost"]), selection=None)])
    return


@app.cell
def _(metrics, paired_runs, plots, plt):
    by_lead = metrics.summarize(paired_runs, by=["label", "lead_hour"])
    _fig, _axes = plt.subplots(2, 3, figsize=(16, 7))
    for _ax, _m in zip(_axes.flat, ["mae_log", "spearman", "mean_cost", "auc10", "brier10", "auc30"]):
        plots.skill_by_lead(by_lead, _m, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(paired_runs, plots, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for _ax, _k in zip(_axes, (5, 10, 30)):
        plots.reliability(paired_runs, k=_k, ax=_ax)
        _ax.set_title(f"P(>{_k} ppb) reliability")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, paired_runs):
    _labels = sorted(paired_runs["label"].unique())
    baseline = mo.ui.dropdown(_labels, value="xgboost_lean" if "xgboost_lean" in _labels else _labels[0],
                              label="paired bootstrap vs")
    baseline
    return (baseline,)


@app.cell
def _(baseline, metrics, mo, paired_runs, pd):
    _ann = metrics.annotate_verdicts(paired_runs)
    _rows = []
    for _label, _g in _ann.groupby("label"):
        if _label == baseline.value:
            continue
        for _product, _gp in _g.groupby("product"):
            _ref = _ann[(_ann["label"] == baseline.value) & (_ann["product"] == _product)]
            _d, _lo, _hi = metrics.bootstrap_diff(_gp, _ref, "cost")
            _rows.append({"model": _label, "product": _product, "Δ mean cost": _d, "95% lo": _lo, "95% hi": _hi,
                          "verdict": "better" if _hi < 0 else ("worse" if _lo > 0 else "n.s.")})
    mo.vstack([mo.md(f"## Paired run-level bootstrap of rubric cost vs `{baseline.value}` (negative = better)"),
               mo.ui.table(pd.DataFrame(_rows).round(4), selection=None)])
    return


@app.cell
def _(mo, paired_runs):
    _peaks = (paired_runs.groupby(["origin", "station"])["actual_h2s"].max()
              .sort_values(ascending=False).head(25).reset_index())
    event = mo.ui.dropdown(
        {f"{r.origin:%Y-%m-%d %H}Z · {r.station} · peak {r.actual_h2s:.0f} ppb": (r.origin, r.station)
         for r in _peaks.itertuples()},
        label="event case study",
    )
    event
    return (event,)


@app.cell
def _(data, event, mo, paired_runs, plots):
    mo.stop(event.value is None)
    _panel = data.to_panel(data.load_obs())
    plots.fan_chart(paired_runs, _panel, *event.value).figure
    return


if __name__ == "__main__":
    app.run()
