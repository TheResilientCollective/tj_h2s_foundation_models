import numpy as np
import pandas as pd
import pytest

from tsfm_h2s import backtest, metrics
from tsfm_h2s.data import KNOWN_FUTURE_COVARIATES, PAST_ONLY_COVARIATES, STATIONS, TARGET
from tsfm_h2s.models import HourOfDayClimatology, Persistence, SeasonalNaive


def _panel(hours=24 * 60):
    idx = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    cols = [TARGET, *KNOWN_FUTURE_COVARIATES, *PAST_ONLY_COVARIATES]
    panel = {}
    for s in STATIONS:
        df = pd.DataFrame(rng.random((hours, len(cols))), index=idx, columns=cols).astype("float32")
        df[TARGET] = np.arange(hours, dtype="float32")  # value == position, so leakage is visible
        panel[s] = df
    return panel


class Spy:
    """Records what the model was shown."""

    name = "spy"
    quantile_levels = np.array([0.1, 0.5, 0.9])

    def predict(self, target, horizon, past_covariates=None, known_covariates=None):
        self.seen = (target, past_covariates, known_covariates)
        return np.zeros((*target.shape[:2], horizon, 3))


@pytest.mark.parametrize("mode", ["univariate", "covariates", "multivariate_covariates"])
@pytest.mark.parametrize("lag", [0, 3])
def test_no_target_leakage(mode, lag):
    panel = _panel()
    origin = panel[STATIONS[0]].index[24 * 40]
    spy = Spy()
    cfg = backtest.BacktestConfig(mode=mode, context_length=256, seed_lag_hours=lag, transform="none")
    out = backtest.run_backtest(spy, panel, pd.DatetimeIndex([origin]), cfg, progress=False)
    target, past, known = spy.seen
    origin_pos = 24 * 40
    assert np.nanmax(target) == origin_pos - 1 - lag  # last context value is origin-1h-lag
    if mode != "univariate":
        assert past.shape[-1] == 256
        assert known.shape[-1] == 256 + 24 + lag
    # lead 1 targets the origin hour, lead 24 targets origin+23h
    first = out.loc[out.lead_hour == 1, "time"].iloc[0]
    assert first == origin
    assert out.loc[out.lead_hour == 24, "actual_h2s"].iloc[0] == origin_pos + 23


def test_baselines_shapes_and_values():
    x = np.tile(np.arange(24 * 30, dtype=float), (2, 1, 1))
    assert np.all(Persistence().predict(x, 5) == x[0, 0, -1])
    sn = SeasonalNaive().predict(x, 24)
    assert sn[0, 0, 0, 0] == x[0, 0, -24] and sn[0, 0, 23, 0] == x[0, 0, -1]
    hc = HourOfDayClimatology(days=7).predict(x % 24, 24)
    assert np.allclose(hc[0, 0, :, 6], np.arange(24) + (24 * 30) % 24)  # same hour of day


def test_prob_exceed_monotone_and_tails():
    levels = np.array([0.1, 0.5, 0.9])
    q = np.array([[1.0, 3.0, 8.0], [10.0, 40.0, 100.0], [0.0, 0.0, 0.0]])
    p5, p30 = (metrics.prob_exceed(q, levels, k) for k in (5, 30))
    assert 0.1 < p5[0] < 0.5 and p30[0] < 0.1  # tail extrapolation below 1-0.9
    assert p5[1] > 0.9 and 0.5 < p30[1] < 0.9
    assert np.all(p30 <= p5) and np.all((p5 >= 0) & (p5 <= 1))
    assert p5[2] < 1e-3


def test_rubric_matches_production_semantics():
    t = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame({
        "station": "S", "time": t,
        "actual_h2s": [50, 15, 7, 1, 1, 40],
        "h2s_pred":   [1, 1, 1, 1, 40, 12],
    })
    v = metrics.annotate_verdicts(df)["verdict"].tolist()
    assert v == ["dangerous_miss", "smell_miss", "expected_low", "match", "early_warning_ok", "soft_miss"]


def test_forecast_weather_replaces_only_the_horizon():
    from tsfm_h2s.data import FORECAST_WEATHER_COVARIATES

    panel = _panel()
    index = panel[STATIONS[0]].index
    origin = index[24 * 40]
    issued = origin - pd.Timedelta(minutes=50)
    fc = pd.DataFrame(-7.0, index=index, columns=FORECAST_WEATHER_COVARIATES)  # sentinel forecast values
    by_issue = {(issued, s): fc for s in STATIONS}
    mapping = pd.Series([issued], index=pd.DatetimeIndex([origin]))

    spy = Spy()
    cfg = backtest.BacktestConfig(mode="covariates", context_length=128, weather="forecast", transform="none",
                                  stations=[STATIONS[0]])
    out = backtest.run_backtest(spy, panel, pd.DatetimeIndex([origin]), cfg, progress=False,
                                weather_forecast=(by_issue, mapping))
    known = spy.seen[2][0]
    cols = cfg.known_covariates
    obs = panel[STATIONS[0]]
    for row, col in enumerate(cols):
        ctx, hor = known[row, :128], known[row, 128:]
        assert np.allclose(ctx, obs[col].iloc[24 * 40 - 128 : 24 * 40])  # context = observed
        if col in FORECAST_WEATHER_COVARIATES:
            assert np.all(hor == -7.0)  # horizon = archived forecast
        elif col in ("surface_pressure", "cloud_cover"):
            assert np.all(hor == ctx[-1])  # not forecast: persistence
        else:  # tide / calendar: deterministic, unchanged
            assert np.allclose(hor, obs[col].iloc[24 * 40 : 24 * 40 + 24])
    assert (out["weather_issued_at"] == issued).all()
