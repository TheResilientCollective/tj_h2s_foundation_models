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


# --------------------------------------------------------------------------- archived weather forecasts
#
# Observed (historic) weather lives in modeldata and fills the context window.
# For the forecast horizon, operations only has a weather *forecast*: the
# Open-Meteo 15-min feed (``resilient_workflows_public`` asset
# ``openmeteo_15min_forecast``) archives every raw response, roughly hourly
# since 2026-04-10, under FORECAST_ARCHIVE_PREFIX. Each file covers the past
# 24 h and next 24 h for all three stations.

FORECAST_ARCHIVE_PREFIX = "tijuana/weather/raw/forecast_15min/"

STATION_COORDS = {  # tj_h2s_prediction constants.STATIONS
    "SAN YSIDRO": (32.552794, -117.047286),
    "NESTOR - BES": (32.567097, -117.090656),
    "IB CIVIC CTR": (32.576139, -117.115361),
}

# Known-future covariates the forecast feed provides; the rest of the weather
# covariates (surface_pressure, cloud_cover) are held at their last observed
# value in forecast-weather mode. Tide and calendar are deterministic.
FORECAST_WEATHER_COVARIATES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
]
DETERMINISTIC_COVARIATES = ["tide_height", "hour_sin", "hour_cos", "is_night"]


def list_forecast_issues() -> pd.Series:
    """issued_at (UTC) -> object key for every archived forecast, oldest first."""
    import re

    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": FORECAST_ARCHIVE_PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        xml = requests.get(f"{PUBLIC_BASE}/", params=params, timeout=60).text
        keys += re.findall(r"<Key>([^<]*\.json)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    stamps = pd.to_datetime([k.rsplit("_", 1)[1][:-5] for k in keys], format="%Y%m%dT%H%M%SZ", utc=True)
    return pd.Series(keys, index=stamps).sort_index()


_FORECAST_FIELDS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                    "wind_speed_10m", "wind_direction_10m"]


def _load_forecast_json(text: str) -> list[dict]:
    """Parse an archived response, repairing the common write-time truncation.

    Many archive objects are cut off (~32 KB) inside the *last* array of the
    last site (``weather_code``, unused here). Closing the JSON recovers every
    field we use; the result is checked so a deeper truncation still fails.
    """
    import json

    try:
        payload = json.loads(text)
    except ValueError:
        body = text.rstrip()
        body = body[: body.rfind(",")]  # drop the possibly partial last value
        payload = json.loads(body + "]}}]")
    if not isinstance(payload, list):  # an archived API error response
        raise ValueError("archive file is not a forecast response")
    for site in payload:
        m = site["minutely_15"]
        n = len(m["time"])
        if any(len(m.get(f, [])) != n for f in _FORECAST_FIELDS):
            raise ValueError("archive file truncated inside a used field")
        m.pop("weather_code", None)
    return payload


def _parse_forecast(issued_at: pd.Timestamp, payload: list[dict]) -> pd.DataFrame:
    frames = []
    for site in payload:
        lat, lon = site["latitude"], site["longitude"]
        station = min(STATION_COORDS, key=lambda s: (STATION_COORDS[s][0] - lat) ** 2 + (STATION_COORDS[s][1] - lon) ** 2)
        m = pd.DataFrame({k: v for k, v in site["minutely_15"].items() if k == "time" or k in _FORECAST_FIELDS})
        m["time"] = pd.to_datetime(m["time"], utc=True)
        m = m.set_index("time")
        rad = np.deg2rad(m["wind_direction_10m"])
        m["wind_direction_sin"], m["wind_direction_cos"] = np.sin(rad), np.cos(rad)
        hourly = m.drop(columns=["wind_direction_10m"]).astype(float).resample("h").mean()
        hourly["precipitation"] = m["precipitation"].resample("h").sum(min_count=1)
        hourly = hourly.rename(columns={"dew_point_2m": "dewpoint_2m"})
        hourly["station"] = station
        frames.append(hourly.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["issued_at"] = issued_at
    return out


def load_forecast_archive(issued: list[pd.Timestamp] | None = None, workers: int = 4) -> pd.DataFrame:
    """Long table (issued_at, station, time, forecast weather) for the requested issuances.

    Each parsed issuance is cached as ``data/forecast_archive/<issued>.parquet``
    as soon as it is fetched, so an interrupted load resumes where it stopped.
    Unreadable (truncated) archive files are remembered in ``_unreadable.txt``.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    index = list_forecast_issues()
    wanted = index if issued is None else index[index.index.isin(pd.DatetimeIndex(issued))]
    cache = CACHE_DIR / "forecast_archive"
    cache.mkdir(parents=True, exist_ok=True)
    bad_file = cache / "_unreadable.txt"
    bad = set(bad_file.read_text().split()) if bad_file.exists() else set()

    def path(ts):
        return cache / f"{ts:%Y%m%dT%H%M%SZ}.parquet"

    missing = [(ts, key) for ts, key in wanted.items() if not path(ts).exists() and key not in bad]

    def fetch(item):
        ts, key = item
        for attempt in range(4):
            try:
                r = requests.get(f"{PUBLIC_BASE}/{key}", timeout=60)
                r.raise_for_status()
                _parse_forecast(ts, _load_forecast_json(r.text)).to_parquet(path(ts), index=False)
                return True
            except requests.RequestException:
                time.sleep(2**attempt)
            except (ValueError, KeyError, TypeError):
                return False  # stored object is unusable; retrying will not help
        return False

    if missing:
        failed = []
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(fetch, item): item for item in missing}
            for i, fut in enumerate(as_completed(futures), 1):
                if not fut.result():
                    failed.append(futures[fut][1])
                if i % 200 == 0 or i == len(missing):
                    print(f"forecast archive: {i}/{len(missing)} fetched, {len(failed)} unreadable", flush=True)
        if failed:
            # a few archive files are truncated at write time (~32 KB); those
            # origins fall back to the previous readable issuance
            bad_file.write_text("\n".join(sorted(bad | set(failed))))

    frames = [pd.read_parquet(path(ts)) for ts in wanted.index if path(ts).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["issued_at", "station", "time"])


def issuance_for_origins(origins: pd.DatetimeIndex, max_age_hours: float = 6.0) -> pd.Series:
    """origin -> latest archived issuance at or before it (NaT if none within ``max_age_hours``)."""
    issues = pd.DataFrame({"issued_at": list_forecast_issues().index})
    left = pd.DataFrame({"origin": pd.DatetimeIndex(origins)}).sort_values("origin")
    m = pd.merge_asof(left, issues, left_on="origin", right_on="issued_at", direction="backward",
                      tolerance=pd.Timedelta(hours=max_age_hours))
    return m.set_index("origin")["issued_at"]


def forecast_weather_for_origins(origins: pd.DatetimeIndex, max_age_hours: float = 6.0):
    """Archived forecasts usable at each origin.

    Returns ``(by_issue, issued)``: ``by_issue[(issued_at, station)]`` is an hourly
    frame of FORECAST_WEATHER_COVARIATES, and ``issued`` maps origin -> the latest
    *readable* issuance at or before it (NaT when none within ``max_age_hours``).
    """
    origins = pd.DatetimeIndex(origins)
    stamps = pd.Series(list_forecast_issues().index)
    # only the last few issuances before each origin can ever be chosen
    candidates = set()
    for origin in origins:
        hi = stamps.searchsorted(origin, side="right")
        recent = stamps.iloc[max(0, hi - 3) : hi]
        candidates.update(recent[recent >= origin - pd.Timedelta(hours=max_age_hours)])
    archive = load_forecast_archive(sorted(candidates))
    loaded = pd.DataFrame({"issued_at": sorted(archive["issued_at"].unique())})
    left = pd.DataFrame({"origin": origins}).sort_values("origin")
    issued = pd.merge_asof(left, loaded, left_on="origin", right_on="issued_at", direction="backward",
                           tolerance=pd.Timedelta(hours=max_age_hours)).set_index("origin")["issued_at"]
    by_issue = {key: g.set_index("time")[FORECAST_WEATHER_COVARIATES].astype("float32")
                for key, g in archive.groupby(["issued_at", "station"])}
    return by_issue, issued
