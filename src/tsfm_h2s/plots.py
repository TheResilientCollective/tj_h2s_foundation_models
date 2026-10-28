"""Matplotlib figures shared by the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TARGET
from .metrics import quantile_columns

TIER_LINES = {5: "#d4b106", 10: "#e8890c", 30: "#d4380d"}


def _tiers(ax):
    for k, c in TIER_LINES.items():
        ax.axhline(k, color=c, lw=0.8, ls="--", alpha=0.7)


def fan_chart(results: pd.DataFrame, panel: dict, origin, station: str, history_hours: int = 48,
              series: str = "label", ax=None):
    """Observed H2S around one origin with each model's median and 10-90% band."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    origin = pd.Timestamp(origin)
    obs = panel[station][TARGET]
    window = obs.loc[origin - pd.Timedelta(hours=history_hours) : origin + pd.Timedelta(hours=24)]
    ax.plot(window.index, window.values, color="black", lw=1.5, label="measured")
    sel = results[(results["origin"] == origin) & (results["station"] == station)]
    for i, (model, g) in enumerate(sel.groupby(series)):
        g = g.sort_values("time")
        color = f"C{i}"
        ax.plot(g["time"], g["h2s_pred"], color=color, lw=1.5, label=model)
        q = quantile_columns(g)
        lo = next((c for c in q if np.isclose(float(c[1:]), 0.1)), None)
        hi = next((c for c in q if np.isclose(float(c[1:]), 0.9)), None)
        if lo and hi:
            ax.fill_between(g["time"], g[lo], g[hi], color=color, alpha=0.15)
    ax.axvline(origin, color="grey", lw=1)
    _tiers(ax)
    #ax.set_yscale("symlog", linthresh=5)
    ax.set_yscale("linear")
    ax.set_ylabel("H2S (ppb)")
    ax.set_title(f"{station} — origin {origin:%Y-%m-%d %H:%M} UTC")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def skill_by_lead(summary: pd.DataFrame, metric: str, series: str = "label", ax=None):
    """``summary`` from ``metrics.summarize(df, by=[series, 'lead_hour'])``."""
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    for model, g in summary.groupby(series):
        ax.plot(g["lead_hour"], g[metric], marker=".", label=model)
    for edge in (3.5, 6.5):
        ax.axvline(edge, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("lead hour")
    ax.set_title(metric)
    ax.legend(fontsize=7)
    return ax


def reliability(df: pd.DataFrame, k: int = 10, bins: int = 10, series: str = "label", ax=None):
    """Calibration of P(>k): forecast probability vs observed frequency, per model."""
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    for model, g in df.dropna(subset=[f"p{k}", "actual_h2s"]).groupby(series):
        cut = pd.cut(g[f"p{k}"], np.linspace(0, 1, bins + 1), include_lowest=True)
        r = g.groupby(cut, observed=True).agg(p=(f"p{k}", "mean"), o=("actual_h2s", lambda y: (y >= k).mean()), n=("actual_h2s", "size"))
        ax.plot(r["p"], r["o"], marker="o", label=model)
    ax.set_xlabel(f"forecast P(>{k} ppb)")
    ax.set_ylabel("observed frequency")
    ax.legend(fontsize=8)
    return ax
