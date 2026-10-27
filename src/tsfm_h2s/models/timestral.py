"""Mistral Timestral adapter (https://github.com/mistralai/timestral).

Installed by ``uv sync`` (a core dependency). Runs on CPU; ``device="cuda"`` on a GPU.
Sizes: Timestral-5M / 28M / 123M / 209M. Context up to 32 768 steps; NaN = missing.
"""

from __future__ import annotations

import numpy as np


class Timestral:
    def __init__(self, size: str = "5M", device: str = "cpu"):
        from timestral import TimestralForecaster

        self.name = f"timestral_{size}"
        self.model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device)
        warmup = self.model.forecast(np.random.rand(1, 1, 64).astype(np.float32), prediction_length=1)
        self.quantile_levels = np.asarray(warmup.quantile_levels)

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        fc = self.model.forecast(
            target.astype(np.float32),
            prediction_length=horizon,
            past_covariates=None if past_covariates is None else past_covariates.astype(np.float32),
            known_covariates=None if known_covariates is None else known_covariates.astype(np.float32),
        )
        q = fc.quantiles
        return q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
