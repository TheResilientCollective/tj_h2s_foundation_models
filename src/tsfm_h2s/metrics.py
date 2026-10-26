"""Scoring: exceedance probabilities from quantiles, hazard metrics, and the asymmetric rubric.

The rubric is a vectorised port of ``h2s.forecasting.performance_report`` in
tj_h2s_prediction so TSFM scores are directly comparable with the production report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

THRESHOLDS = (5.0, 10.0, 30.0)
CATEGORIES = ["green", "yellow_low", "yellow_high", "orange"]  # severity 0..3
PRODUCTS = {"nowcast": (1, 3), "nearcast": (4, 6), "forecast": (7, 24)}

VERDICT_COST = {
    "match": 0.0,
    "yellow_band": 0.25,
    "early_warning_ok": 0.25,
    "expected_low": 0.5,
    "soft_miss": 1.0,
    "false_alarm": 1.5,
    "smell_miss": 2.5,
    "dangerous_miss": 5.0,
}
ACCEPTABLE = {"match", "yellow_band", "early_warning_ok"}
TOLERANCE_HOURS = 2


def product_of(lead_hour) -> np.ndarray:
    lead = np.asarray(lead_hour)
    return np.select([lead <= 3, lead <= 6], ["nowcast", "nearcast"], "forecast")


def severity(values) -> np.ndarray:
    """0 green (<5), 1 yellow_low (5-10), 2 yellow_high (10-30), 3 orange (>=30); -1 NaN."""
    v = np.asarray(values, dtype=float)
    sev = np.digitize(v, [5.0, 10.0, 30.0])
    return np.where(np.isnan(v), -1, sev)


# --------------------------------------------------------------------------- probabilities


def prob_exceed(quantiles: np.ndarray, levels, k: float) -> np.ndarray:
    """P(Y > k) from predicted quantiles (rows x levels) on the ppb scale.

    Interpolates the CDF linearly in log1p space between quantiles and
    extrapolates an exponential tail beyond the outermost levels, so a model
    whose top quantile is 0.9 (TimesFM) can still express P(>30) < 0.1.
    """
    q = np.log1p(np.clip(np.asarray(quantiles, dtype=float), 0, None))
    q = np.maximum.accumulate(q, axis=1)
    tau = np.asarray(levels, dtype=float)
    x = np.log1p(k)
    out = np.empty(len(q))
    for i, row in enumerate(q):
        if np.isnan(row).any():
            out[i] = np.nan
        elif x >= row[-1]:
            # upper tail: 1-F(x) = (1-tau_Q) * exp(-(x - x_Q) / s)
            dx = max(row[-1] - row[-2], 1e-6)
            s = dx / np.log((1 - tau[-2]) / (1 - tau[-1]))
            out[i] = (1 - tau[-1]) * np.exp(-(x - row[-1]) / s)
        elif x <= row[0]:
            # lower tail: F(x) = tau_1 * exp((x - x_1) / s)
            dx = max(row[1] - row[0], 1e-6)
            s = dx / np.log(tau[1] / tau[0])
            out[i] = 1 - tau[0] * np.exp((x - row[0]) / s)
        else:
            # ties in the quantile function → take the right-most level
            j = np.searchsorted(row, x, side="right")
            x0, x1 = row[j - 1], row[j]
            w = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
            out[i] = 1 - (tau[j - 1] + w * (tau[j] - tau[j - 1]))
    return out


def add_exceedance_probs(df: pd.DataFrame, levels) -> pd.DataFrame:
    qcols = quantile_columns(df)
    df = df.copy()
    for k in THRESHOLDS:
        df[f"p{int(k)}"] = prob_exceed(df[qcols].to_numpy(), levels, k)
    return df


def quantile_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]


def pinball_loss(df: pd.DataFrame) -> float:
    qcols = quantile_columns(df)
    if not qcols:
        return np.nan
    y = df["actual_h2s"].to_numpy()[:, None]
    tau = np.array([float(c[1:]) for c in qcols])[None, :]
    q = df[qcols].to_numpy()
    loss = np.maximum(tau * (y - q), (tau - 1) * (y - q))
    return float(np.nanmean(loss)) if np.isfinite(loss).any() else np.nan


# --------------------------------------------------------------------------- rubric


def annotate_verdicts(df: pd.DataFrame, point_col: str = "h2s_pred") -> pd.DataFrame:
    """Add pred_sev / actual_sev / verdict / cost (rubric of performance_report.py)."""
    df = df.copy()
    ps = severity(df[point_col])
    as_ = severity(df["actual_h2s"])

    # max observed severity within ±TOLERANCE_HOURS of each target hour, per station
    actual = (
        df[["station", "time", "actual_h2s"]].dropna().drop_duplicates(["station", "time"])
    )
    actual["sev"] = severity(actual["actual_h2s"])
    windowed = []
    for station, g in actual.groupby("station"):
        s = g.set_index("time")["sev"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        roll = s.rolling(2 * TOLERANCE_HOURS + 1, center=True, min_periods=1).max()
        windowed.append(pd.DataFrame({"station": station, "time": roll.index, "near_sev": roll.values}))
    near = pd.concat(windowed) if windowed else pd.DataFrame(columns=["station", "time", "near_sev"])
    near_sev = df[["station", "time"]].merge(near, on=["station", "time"], how="left")["near_sev"].to_numpy()

    verdict = np.full(len(df), "match", dtype=object)
    under = ps < as_
    over = ps > as_
    yellow_band = (ps != as_) & np.isin(ps, [1, 2]) & np.isin(as_, [1, 2])
    verdict[under] = "soft_miss"
    verdict[under & (as_ == 3) & (ps == 0)] = "dangerous_miss"
    verdict[under & (as_ == 2) & (ps == 0)] = "smell_miss"
    verdict[under & (as_ == 1) & (ps == 0)] = "expected_low"
    verdict[over] = np.where(near_sev[over] >= ps[over], "early_warning_ok", "false_alarm")
    verdict[yellow_band] = "yellow_band"
    verdict[(ps < 0) | (as_ < 0)] = None

    df["pred_sev"], df["actual_sev"], df["verdict"] = ps, as_, verdict
    df["cost"] = df["verdict"].map(VERDICT_COST)
    return df


# --------------------------------------------------------------------------- summary


def roc_auc(pos: np.ndarray, score: np.ndarray) -> float:
    """Threshold-free ranking skill of a probability for a rare event (Mann-Whitney)."""
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return np.nan
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _hazard_stats(y: np.ndarray, p: np.ndarray, k: float, cut: float = 0.5) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok], p[ok]
    pos = y >= k
    call = p > cut
    tp, fp = int((pos & call).sum()), int((~pos & call).sum())
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    k = int(k)
    return {
        f"n_ge{k}": n_pos,
        f"recall{k}": tp / n_pos if n_pos else np.nan,
        f"precision{k}": tp / (tp + fp) if (tp + fp) else np.nan,
        f"far{k}": fp / n_neg if n_neg else np.nan,
        f"brier{k}": float(np.mean((p - pos) ** 2)) if len(p) else np.nan,
        f"auc{k}": roc_auc(pos, p),
    }


def summarize(df: pd.DataFrame, by: list[str] | None = None, point_col: str = "h2s_pred") -> pd.DataFrame:
    """One row per group: n, MAE, Spearman, rubric headline, hazard recall/FAR/Brier at 5/10/30."""
    df = df.dropna(subset=["actual_h2s", point_col])
    if "verdict" not in df:
        df = annotate_verdicts(df, point_col)
    by = by or ["model"]
    rows = []
    for key, g in df.groupby(by, dropna=False):
        y, yhat = g["actual_h2s"].to_numpy(), g[point_col].to_numpy()
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row |= {
            "n": len(g),
            "mae": float(np.mean(np.abs(y - yhat))),
            "mae_log": float(np.mean(np.abs(np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(yhat, 0, None))))),
            "spearman": float(spearmanr(y, yhat).statistic) if len(g) > 2 else np.nan,
            "tolerant_acc": float(g["verdict"].isin(ACCEPTABLE).mean()),
            "dangerous_miss": int((g["verdict"] == "dangerous_miss").sum()),
            "smell_miss": int((g["verdict"] == "smell_miss").sum()),
            "mean_cost": float(g["cost"].mean()),
            "pinball": pinball_loss(g),
        }
        for k in THRESHOLDS:
            col = f"p{int(k)}"
            if col in g:
                row |= _hazard_stats(y, g[col].to_numpy(dtype=float), k)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str = "cost", n_boot: int = 2000, seed: int = 0):
    """Paired run-level bootstrap of mean(metric_a) - mean(metric_b) over shared (origin, station, lead).

    Resamples whole origins (runs) so serial correlation within a run is respected.
    Returns (diff, lo95, hi95).
    """
    keys = ["origin", "station", "lead_hour"]
    m = a[keys + [metric]].merge(b[keys + [metric]], on=keys, suffixes=("_a", "_b")).dropna()
    per_run = m.groupby("origin")[[f"{metric}_a", f"{metric}_b"]].agg(["sum", "count"])
    sa, sb = per_run[(f"{metric}_a", "sum")].to_numpy(), per_run[(f"{metric}_b", "sum")].to_numpy()
    cnt = per_run[(f"{metric}_a", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cnt), size=(n_boot, len(cnt)))
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / cnt[idx].sum(1)
    point = (sa.sum() - sb.sum()) / cnt.sum()
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
