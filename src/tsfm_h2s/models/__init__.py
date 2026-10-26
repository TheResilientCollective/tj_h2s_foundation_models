"""Forecasters sharing the ``backtest.Forecaster`` interface.

The TSFM adapters import their heavy dependencies lazily, so baselines work
without torch installed.
"""

from .baselines import HourOfDayClimatology, Persistence, SeasonalNaive


def get_model(name: str, device: str = "cpu"):
    """'persistence' | 'seasonal_naive' | 'climatology' | 'timestral-5M' (28M/123M/209M) | 'timesfm'."""
    if name == "persistence":
        return Persistence()
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "climatology":
        return HourOfDayClimatology()
    if name.startswith("timestral"):
        from .timestral import Timestral

        return Timestral(size=name.split("-", 1)[1] if "-" in name else "5M", device=device)
    if name == "timesfm":
        from .timesfm import TimesFM

        return TimesFM(device=device)
    raise ValueError(f"unknown model {name!r}")


__all__ = ["HourOfDayClimatology", "Persistence", "SeasonalNaive", "get_model"]
