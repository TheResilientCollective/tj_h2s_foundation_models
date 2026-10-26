"""Collect saved backtest results and align them for paired comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import backtest, data

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
KEYS = ["origin", "station", "lead_hour"]


def label(df: pd.DataFrame) -> pd.Series:
    """Display name: model plus mode/context/lag when they are not the defaults."""
    name = df["model"].astype(str)
    if "mode" in df:
        name = name.where(df["mode"].isna(), name + "|" + df["mode"].fillna("").astype(str))
    if "context_length" in df:
        ctx = df["context_length"]
        name = name.where(ctx.isna() | (ctx == 2048), name + "|ctx" + ctx.fillna(0).astype(int).astype(str))
    if "weather" in df:
        wx = df["weather"]
        name = name.where(wx.isna() | wx.isin(["none", "oracle"]), name + "|wx-" + wx.fillna("").astype(str))
    if "seed_lag_hours" in df:
        lag = df["seed_lag_hours"]
        name = name.where(lag.isna() | (lag == 0), name + "|lag" + lag.fillna(0).astype(int).astype(str))
    return name


def load_results(origins: str = "production", include_xgboost: bool = True) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(RESULTS_DIR.glob(f"*__{origins}.parquet"))]
    if include_xgboost and origins == "production":
        v = data.load_validation_store()
        frames += [backtest.production_reference(v, variant) for variant in ("lean", "evidence")]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["label"] = label(df)
    return df


def paired(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Keep only (origin, station, lead) cells that every selected model forecast and that were measured."""
    labels = labels or sorted(df["label"].unique())
    df = df[df["label"].isin(labels)].dropna(subset=["h2s_pred", "actual_h2s"])
    counts = df.groupby(KEYS)["label"].nunique()
    full = counts[counts == len(labels)].index
    return df.set_index(KEYS).loc[lambda x: x.index.isin(full)].reset_index()
