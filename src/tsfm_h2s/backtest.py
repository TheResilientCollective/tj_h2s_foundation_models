"""Rolling-origin backtest that mirrors the production products (leads 1-24 from each run).

Origin semantics match the validation store: lead L targets ``origin + (L-1)h``
where ``origin = floor(run_ts)``; the last usable observation is
``origin - 1h - seed_lag_hours``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from .data import (
    DETERMINISTIC_COVARIATES,
    FORECAST_WEATHER_COVARIATES,
    KNOWN_FUTURE_COVARIATES,
    PAST_ONLY_COVARIATES,
    STATIONS,
    TARGET,
)
from .metrics import add_exceedance_probs, product_of

Mode = Literal["univariate", "covariates", "multivariate", "multivariate_covariates"]
# Station whose (valley-wide) weather/flow covariates drive the multivariate runs.
SHARED_COVARIATE_STATION = "NESTOR - BES"


class Forecaster(Protocol):
    name: str
    quantile_levels: np.ndarray

    def predict(
        self,
        target: np.ndarray,  # (B, V, T), NaN = missing
        horizon: int,
        past_covariates: np.ndarray | None = None,  # (B, Cp, T)
        known_covariates: np.ndarray | None = None,  # (B, Ck, T + horizon)
    ) -> np.ndarray:  # (B, V, horizon, Q) quantiles on the (transformed) input scale
        ...


@dataclass
class BacktestConfig:
    mode: Mode = "univariate"
    context_length: int = 2048
    horizon: int = 24
    transform: Literal["log1p", "none"] = "log1p"
    seed_lag_hours: int = 0
    # "oracle": observed weather over the horizon (upper bound).
    # "forecast": archived Open-Meteo forecast available at the origin (operational realism);
    #             weather the feed does not forecast is held at its last observed value.
    weather: Literal["oracle", "forecast"] = "oracle"
    batch_size: int = 64
    stations: list[str] = field(default_factory=lambda: list(STATIONS))
    known_covariates: list[str] = field(default_factory=lambda: list(KNOWN_FUTURE_COVARIATES))
    past_covariates: list[str] = field(default_factory=lambda: list(PAST_ONLY_COVARIATES))


def production_origins(validation: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(validation["origin"].unique()))


def synthetic_origins(start, end, every_hours: int = 6) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq=f"{every_hours}h")


def _forward(x: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None)) if transform == "log1p" else x


def _inverse(x: np.ndarray, transform: str) -> np.ndarray:
    return np.expm1(x) if transform == "log1p" else x


def _slice(frame: pd.DataFrame, cols: list[str], start: int, stop: int) -> np.ndarray:
    """Rows [start, stop) of ``cols`` as (C, stop-start), NaN-padded when start < 0."""
    arr = frame[cols].to_numpy(dtype=np.float32).T
    pad = max(0, -start)
    out = arr[:, max(start, 0):stop]
    if pad:
        out = np.concatenate([np.full((len(cols), pad), np.nan, np.float32), out], axis=1)
    return out


def _forecast_horizon(known: np.ndarray, cfg: BacktestConfig, fc: pd.DataFrame, horizon_times: pd.DatetimeIndex) -> np.ndarray:
    """Replace the horizon part of the known-covariate block with what was knowable at the origin."""
    known = known.copy()
    n_ctx = known.shape[1] - len(horizon_times)
    fc = fc.reindex(horizon_times).ffill().bfill()
    for row, col in enumerate(cfg.known_covariates):
        if col in DETERMINISTIC_COVARIATES:
            continue
        if col in FORECAST_WEATHER_COVARIATES:
            known[row, n_ctx:] = fc[col].to_numpy()
        else:  # not forecast by the feed: persistence of the last observation
            known[row, n_ctx:] = known[row, n_ctx - 1]
    return known


def run_backtest(
    model: Forecaster,
    panel: dict[str, pd.DataFrame],
    origins: pd.DatetimeIndex,
    cfg: BacktestConfig,
    progress: bool = True,
    weather_forecast: tuple | None = None,
) -> pd.DataFrame:
    """Forecast every origin x station and return long rows joined to the measured H2S.

    ``weather_forecast`` is ``data.forecast_weather_for_origins(origins)``; required
    when ``cfg.weather == "forecast"`` with a covariate mode. Origins without a
    usable archived forecast are skipped.
    """
    index = next(iter(panel.values())).index
    multivariate = cfg.mode.startswith("multivariate")
    with_cov = cfg.mode in ("covariates", "multivariate_covariates")
    use_fc = with_cov and cfg.weather == "forecast"
    if use_fc and weather_forecast is None:
        raise ValueError("cfg.weather='forecast' needs weather_forecast=data.forecast_weather_for_origins(origins)")
    by_issue, issued = weather_forecast if use_fc else ({}, None)
    pred_len = cfg.horizon + cfg.seed_lag_hours

    # one work item per (origin, station) — or per origin for multivariate
    items = []
    for origin in origins:
        pos = index.get_indexer([origin])[0]
        if pos < 0 or pos + cfg.horizon > len(index):
            continue  # origin outside the observed grid (or horizon runs past the data)
        ctx_end = pos - cfg.seed_lag_hours  # exclusive
        ctx_start = ctx_end - cfg.context_length
        issue = issued.get(origin) if use_fc else None
        if use_fc and pd.isna(issue):
            continue  # no archived weather forecast for this origin
        groups = [cfg.stations] if multivariate else [[s] for s in cfg.stations]
        for stations in groups:
            cov_station = SHARED_COVARIATE_STATION if multivariate else stations[0]
            cov_frame = panel[cov_station]
            target = np.stack([_slice(panel[s], [TARGET], ctx_start, ctx_end)[0] for s in stations])
            item = {"origin": origin, "stations": stations, "pos": pos,
                    "target": _forward(target, cfg.transform)}
            if with_cov:
                item["past"] = _slice(cov_frame, cfg.past_covariates, ctx_start, ctx_end)
                item["known"] = _slice(cov_frame, cfg.known_covariates, ctx_start, ctx_end + pred_len)
                if use_fc:
                    fc = by_issue.get((issue, cov_station))
                    if fc is None:
                        continue
                    item["known"] = _forecast_horizon(item["known"], cfg, fc, index[ctx_end : ctx_end + pred_len])
                    item["issued_at"] = issue
            items.append(item)

    rows = []
    batches = range(0, len(items), cfg.batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc=model.name)
        except ImportError:
            pass
    for b in batches:
        chunk = items[b : b + cfg.batch_size]
        target = np.stack([it["target"] for it in chunk])
        past = np.stack([it["past"] for it in chunk]) if with_cov else None
        known = np.stack([it["known"] for it in chunk]) if with_cov else None
        q = model.predict(target, pred_len, past_covariates=past, known_covariates=known)
        q = _inverse(np.asarray(q, dtype=np.float64), cfg.transform)[:, :, cfg.seed_lag_hours:, :]
        for it, qi in zip(chunk, q):
            for v, station in enumerate(it["stations"]):
                for lead in range(1, cfg.horizon + 1):
                    rows.append((it["origin"], station, lead, it["pos"] + lead - 1, it.get("issued_at"), *qi[v, lead - 1]))

    levels = [round(float(t), 4) for t in model.quantile_levels]
    qcols = [f"q{t}" for t in levels]
    out = pd.DataFrame(rows, columns=["origin", "station", "lead_hour", "_pos", "weather_issued_at", *qcols])
    out["time"] = index[out["_pos"].to_numpy()]
    out["actual_h2s"] = np.nan
    for station in out["station"].unique():
        m = out["station"] == station
        out.loc[m, "actual_h2s"] = panel[station][TARGET].to_numpy()[out.loc[m, "_pos"].to_numpy()]
    out = out.drop(columns="_pos")
    out["h2s_pred"] = _median(out[qcols].to_numpy(), levels)
    out["product"] = product_of(out["lead_hour"])
    out["model"] = model.name
    out["mode"] = cfg.mode
    out["context_length"] = cfg.context_length
    out["seed_lag_hours"] = cfg.seed_lag_hours
    out["weather"] = cfg.weather if with_cov else "none"
    return add_exceedance_probs(out, levels)


def _median(q: np.ndarray, levels: list[float]) -> np.ndarray:
    levels = np.asarray(levels)
    if np.any(np.isclose(levels, 0.5)):
        return q[:, int(np.argmin(np.abs(levels - 0.5)))]
    return np.array([np.interp(0.5, levels, row) for row in q])


def production_reference(validation: pd.DataFrame, variant: str = "lean") -> pd.DataFrame:
    """Validation-store rows reshaped to the backtest schema (for paired comparison)."""
    v = validation.loc[validation["variant"] == variant].copy()
    # several runs can share an origin hour; keep the latest, like an operator would see
    v = v.sort_values("run_ts").drop_duplicates(["origin", "station", "lead_hour"], keep="last")
    v["model"] = f"xgboost_{variant}"
    cols = ["origin", "station", "lead_hour", "time", "actual_h2s", "h2s_pred", "p5", "p10", "p30", "product", "model"]
    return v[cols].reset_index(drop=True)
