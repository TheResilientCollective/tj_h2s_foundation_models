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

import pandas as pd

from tsfm_h2s import backtest, data
from tsfm_h2s.models import get_model, model_name

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run_tag(model_name: str, args: argparse.Namespace) -> str:
    """Results file name: every option that changes the forecasts is in it.

    Runs can then go in parallel (separate pods or processes on one machine)
    without overwriting each other, and --skip-existing never mistakes a
    different configuration (e.g. a --limit smoke test) for this one.
    """
    weather = f"__wx{args.weather}" if "covariates" in args.mode else ""
    limit = f"__limit{args.limit}" if args.limit else ""
    origins = args.origins
    if origins == "extended":
        end = pd.Timestamp(args.end).strftime("%Y%m%d") if args.end else "dataend"
        origins += f"-{pd.Timestamp(args.start):%Y%m%d}-{end}-{args.every}h"
    return f"{model_name}__{args.mode}{weather}__ctx{args.context}__{args.transform}__lag{args.seed_lag}{limit}__{origins}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="univariate",
                    choices=["univariate", "covariates", "multivariate", "multivariate_covariates"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--transform", default="log1p", choices=["log1p", "none"])
    ap.add_argument("--seed-lag", type=int, default=0)
    ap.add_argument("--weather", default="oracle", choices=["oracle", "forecast"],
                    help="horizon weather for covariate modes: observed (oracle) or archived forecast")
    ap.add_argument("--origins", default="production", choices=["production", "extended"])
    ap.add_argument("--start", default="2025-10-01", help="extended origins: first origin")
    ap.add_argument("--end", default=None, help="extended origins: last origin (default: data end - 24h)")
    ap.add_argument("--every", type=int, default=6, help="extended origins: spacing in hours")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="only the first N origins (smoke tests)")
    ap.add_argument("--skip-existing", action="store_true", help="exit if this run's results file already exists")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    path = RESULTS / f"{run_tag(model_name(args.model), args)}.parquet"
    if args.skip_existing and path.exists():
        print(f"exists, skipping -> {path}")
        return
    model = get_model(args.model, device=args.device)
    assert model.name == model_name(args.model), (model.name, model_name(args.model))

    obs = data.load_obs()
    panel = data.fill_covariates(data.to_panel(obs))
    if args.origins == "production":
        origins = backtest.production_origins(data.load_validation_store())
    else:
        end = args.end or (obs["time"].max() - pd.Timedelta(hours=24))
        origins = backtest.synthetic_origins(args.start, end, args.every)
    if args.limit:
        origins = origins[: args.limit]

    cfg = backtest.BacktestConfig(mode=args.mode, context_length=args.context, transform=args.transform,
                                  seed_lag_hours=args.seed_lag, weather=args.weather, batch_size=args.batch_size)
    weather_forecast = None
    if args.weather == "forecast" and "covariates" in args.mode:
        weather_forecast = data.forecast_weather_for_origins(origins)
    out = backtest.run_backtest(model, panel, origins, cfg, weather_forecast=weather_forecast)

    RESULTS.mkdir(exist_ok=True)
    data.atomic_write(path, lambda tmp: out.to_parquet(tmp, index=False))
    print(f"wrote {len(out):,} rows -> {path}")


if __name__ == "__main__":
    main()
