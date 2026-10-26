"""Load the public H2S observation + covariate table and the production validation store."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PUBLIC_BASE = "https://oss.resilientservice.mooo.com/resilentpublic"
OBS_URL = f"{PUBLIC_BASE}/latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
VALIDATION_URL = f"{PUBLIC_BASE}/tijuana/forecast/validation_store/validation.parquet"

CACHE_DIR = Path(os.environ.get("TSFM_H2S_CACHE", Path(__file__).resolve().parents[2] / "data"))

STATIONS = ["IB CIVIC CTR", "NESTOR - BES", "SAN YSIDRO"]
TARGET = "H2S"

# Known over the forecast horizon in operations (from weather / tide forecasts).
# In the backtest these are observed values => "oracle weather" upper bound.
KNOWN_FUTURE_COVARIATES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "dewpoint_2m",
    "precipitation",
    "tide_height",
    "hour_sin",
    "hour_cos",
    "is_night",
]
# Only known up to the origin (no forecast exists).
PAST_ONLY_COVARIATES = [
    "Flow (m^3/s)--Border",
    "sbiwtp_flow_mgd",
    "sbiwtp_anomaly",
    "sbiwtp_deficit",
]


def _fetch(url: str, refresh: bool = False) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if path.exists() and not refresh:
        return path.read_bytes()
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.content


def load_obs(refresh: bool = False) -> pd.DataFrame:
    """Long table (time UTC, site_name, H2S, covariates...) as published by the Dagster pipeline."""
    df = pd.read_parquet(io.BytesIO(_fetch(OBS_URL, refresh)))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["site_name", "time"]).drop_duplicates(["site_name", "time"], keep="last")
    return df.reset_index(drop=True)


def load_validation_store(refresh: bool = False) -> pd.DataFrame:
    """Production XGBoost product rows joined to measured H2S (the comparison reference)."""
    df = pd.read_parquet(io.BytesIO(_fetch(VALIDATION_URL, refresh)))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["origin"] = pd.to_datetime(df["run_ts"], format="%Y-%m-%dT%H%MZ", utc=True).dt.floor("h")
    return df


def to_panel(obs: pd.DataFrame, columns: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Per-station frames on a regular hourly UTC grid shared by all stations (gaps = NaN)."""
    columns = columns or [TARGET, *KNOWN_FUTURE_COVARIATES, *PAST_ONLY_COVARIATES]
    start, end = obs["time"].min(), obs["time"].max()
    grid = pd.date_range(start, end, freq="h", tz="UTC")
    panel = {}
    for station in STATIONS:
        g = obs.loc[obs["site_name"] == station].set_index("time")[columns]
        g = g.apply(pd.to_numeric, errors="coerce").astype("float32")
        panel[station] = g.reindex(grid)
    return panel


def fill_covariates(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Covariates must not have long NaN runs for the models; target gaps stay NaN.

    Calendar columns are recomputed exactly; the rest are time-interpolated
    (short gaps) then forward/back filled.
    """
    out = {}
    for station, g in panel.items():
        g = g.copy()
        hours = g.index.hour.to_numpy()
        g["hour_sin"] = np.sin(2 * np.pi * hours / 24).astype("float32")
        g["hour_cos"] = np.cos(2 * np.pi * hours / 24).astype("float32")
        cov = [c for c in g.columns if c != TARGET]
        g[cov] = g[cov].interpolate(method="time", limit=6).ffill().bfill()
        out[station] = g
    return out
