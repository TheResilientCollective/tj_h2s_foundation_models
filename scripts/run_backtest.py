"""Run one model/config through the backtest and write results/<tag>.parquet.

Examples:
  uv run python scripts/run_backtest.py --model climatology
  uv run python scripts/run_backtest.py --model timestral-5M --mode covariates
  uv run python scripts/run_backtest.py --model timesfm --mode multivariate_covariates --device cuda
  uv run python scripts/run_backtest.py --model timestral-28M --origins extended --start 2025-10-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm_h2s import backtest, data
from tsfm_h2s.models import get_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="univariate",
                    choices=["univariate", "covariates", "multivariate", "multivariate_covariates"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--transform", default="log1p", choices=["log1p", "none"])
    ap.add_argument("--seed-lag", type=int, default=0)
    ap.add_argument("--origins", default="production", choices=["production", "extended"])
    ap.add_argument("--start", default="2025-10-01", help="extended origins: first origin")
    ap.add_argument("--end", default=None, help="extended origins: last origin (default: data end - 24h)")
    ap.add_argument("--every", type=int, default=6, help="extended origins: spacing in hours")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="only the first N origins (smoke tests)")
    args = ap.parse_args()

    obs = data.load_obs()
    panel = data.fill_covariates(data.to_panel(obs))
    if args.origins == "production":
        origins = backtest.production_origins(data.load_validation_store())
    else:
        end = args.end or (obs["time"].max() - __import__("pandas").Timedelta(hours=24))
        origins = backtest.synthetic_origins(args.start, end, args.every)
    if args.limit:
        origins = origins[: args.limit]

    cfg = backtest.BacktestConfig(mode=args.mode, context_length=args.context, transform=args.transform,
                                  seed_lag_hours=args.seed_lag, batch_size=args.batch_size)
    model = get_model(args.model, device=args.device)
    out = backtest.run_backtest(model, panel, origins, cfg)

    tag = f"{model.name}__{args.mode}__ctx{args.context}__{args.transform}__lag{args.seed_lag}__{args.origins}"
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{tag}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {len(out):,} rows -> {path}")


if __name__ == "__main__":
    main()
