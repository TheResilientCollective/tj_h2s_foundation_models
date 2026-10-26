"""Cheap reference forecasters with the same interface as the TSFM adapters."""

from __future__ import annotations

import numpy as np

LEVELS = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def _last_valid(x: np.ndarray) -> np.ndarray:
    """Last non-NaN value along the final axis (NaN if none)."""
    valid = ~np.isnan(x)
    idx = np.where(valid.any(-1), x.shape[-1] - 1 - np.argmax(valid[..., ::-1], axis=-1), 0)
    out = np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]
    return np.where(valid.any(-1), out, np.nan)


class Persistence:
    """Last observed value, held flat (degenerate quantiles)."""

    name = "persistence"
    quantile_levels = LEVELS

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        last = _last_valid(target)  # (B, V)
        return np.broadcast_to(last[:, :, None, None], (*last.shape, horizon, len(LEVELS))).copy()


class SeasonalNaive:
    """Value at the same hour on the previous day."""

    name = "seasonal_naive_24h"
    quantile_levels = LEVELS

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        T = target.shape[-1]
        lead = np.arange(horizon)
        src = T - 24 + (lead % 24)
        vals = target[:, :, src]  # (B, V, H)
        return np.repeat(vals[..., None], len(LEVELS), axis=-1)


class HourOfDayClimatology:
    """Empirical quantiles of the last ``days`` days at each hour of day — a probabilistic baseline."""

    quantile_levels = LEVELS

    def __init__(self, days: int = 28):
        self.days = days
        self.name = f"hour_climatology_{days}d"

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        T = target.shape[-1]
        window = target[:, :, T - 24 * self.days :]  # (B, V, 24*days), last column = hour T-1
        by_hour = window.reshape(*window.shape[:2], self.days, 24)  # [..., day, hour offset]
        out = np.empty((*target.shape[:2], horizon, len(LEVELS)))
        for lead in range(horizon):
            h = lead % 24  # forecast step `lead` is (T + lead) ≡ column offset h of each day block
            out[:, :, lead, :] = np.moveaxis(np.nanquantile(by_hour[..., h], LEVELS, axis=-1), 0, -1)
        return out
