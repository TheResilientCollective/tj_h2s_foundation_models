"""Google TimesFM 3.0 adapter (https://github.com/google-research/timesfm).

Install: ``uv sync --extra timesfm``. GPU strongly recommended for full backtests
(use ``colab/run_backtest_colab.ipynb``); CPU works for small runs.

LICENSE: TimesFM 3.0 self-hosted weights are non-commercial / non-production
only. Research use here is fine; a production path would be BigQuery ML or the
Apache-2.0 TimesFM 2.5 weights.

Output is 9 deciles (0.1-0.9) only, so P(>30 ppb) relies on tail extrapolation
in ``metrics.prob_exceed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fill(x: np.ndarray) -> np.ndarray:
    """Interpolate interior NaN per row, back/forward fill the ends."""
    return pd.DataFrame(x.T).interpolate(limit_direction="both").to_numpy(dtype=np.float32).T


class TimesFM:
    def __init__(self, device: str = "cpu", batch_size: int = 16, checkpoint: str = "google/timesfm-3.0-pytorch"):
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.name = "timesfm_3.0"
        self.model = TimesFM3Evaluator(
            ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=batch_size, device=device)
        )
        self.quantile_levels = np.asarray(self.model.config.quantiles if hasattr(self.model, "config") else np.arange(1, 10) / 10)

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        contexts, past, known = [], [], []
        for i in range(len(target)):
            # drop leading columns where any target is still missing, fill the rest
            observed = ~np.isnan(target[i]).any(0)
            start = int(np.argmax(observed)) if observed.any() else target.shape[-1] - 1
            contexts.append(_fill(target[i][:, start:]))
            if past_covariates is not None:
                past.append(_fill(past_covariates[i][:, start:]))
            if known_covariates is not None:
                known.append(_fill(known_covariates[i][:, start:]))
        outs = self.model.predict_batch(
            contexts,
            horizon=horizon,
            past_only_covariates=past or None,
            past_future_covariates=known or None,
            return_quantiles=True,
            use_symmetric_averaging=False,
            make_positive=False,  # targets are log1p (non-negative) by default; keep raw-scale runs honest
        )
        return np.stack([np.asarray(o.quantiles).reshape(target.shape[1], horizon, -1) for o in outs])
