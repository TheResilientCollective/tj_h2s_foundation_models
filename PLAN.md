# Plan: can time-series foundation models forecast Tijuana River Valley H2S?

## 1. Question

Can zero-shot (and possibly lightly fine-tuned) time-series foundation models
(TSFMs) — **Mistral Timestral** and **Google TimesFM 3.0** — match or beat the
production per-station XGBoost models (`tj_h2s_prediction`, Lean / Evidence
variants) at forecasting hourly H2S at IB_CIVIC_CTR, NESTOR-BES and SAN_YSIDRO
for leads 1–24 h?

"Valid" here means the operational bar, not a generic forecasting score:

| What matters | Metric | Why |
|---|---|---|
| Catching hazards | recall / precision / false-alarm rate of P(>k) calls at k = 5, 10, 30 ppb | the 4-tier view (green / yellow / yellow-high / orange) |
| Not missing smell events | asymmetric rubric (dangerous-miss, smell-miss, early-warning-ok within ±2 h) | mirrors `forecasting/performance_report.py` |
| Ranking risk | Spearman by lead hour | the honest framing of the 7–24 h `forecast` product |
| Magnitude | MAE (raw + log1p), by lead | comparability with the skill curves |
| Probabilistic quality | quantile (pinball) loss, Brier score at 5/10/30, PIT coverage | TSFMs are natively probabilistic — the reason to try them |

## 2. Why this is plausible (and why it might not be)

**For:** both models are probabilistic, take long context (Timestral 32 768 steps,
TimesFM 3.0 15 360), support **past-only + known-future covariates** and
**native multivariate** targets (the three stations jointly). Both topped
GIFT-Eval / fev-bench / TIME at launch. H2S has strong diurnal structure
(calm, stable nights) that long-context models should pick up without feature
engineering.

**Against:** H2S is intermittent and heavy-tailed (median ~1 ppb, max 915 ppb,
≥30 ppb is ~4% of hours at NESTOR-BES and only ~0.4–0.8% at the other two). Generic pretraining
corpora are mostly smooth business/energy series; zero-shot models tend to
regress to the median and under-call spikes — exactly the failure that matters
most here (cf. `RED_TIER_RECALL_DIAGNOSIS.md`). The XGBoost models already
encode domain drivers (wind/stability regime, SBIWTP flow anomaly, tides).

The experiment is cheap, so run it and let the backtest decide.

## 3. Data (all public, no credentials)

| Data | Source | Coverage |
|---|---|---|
| Hourly obs + covariates (`modeldata_h2s_nofill.parquet`, 58 cols) | `resilentpublic/latest/tijuana/forecast_data/` | 2023-11-20 → present (~58k rows, 3 stations, ~94–98% hourly coverage) |
| Production XGBoost forecasts joined to actuals (`validation.parquet`) | `resilentpublic/tijuana/forecast/validation_store/` | 711 runs, 2026-06-28 → present, leads 1–24, both variants, p5/p10/p30 |

Covariate roles (TSFM terms):

- **Known-future** (also known over the forecast horizon): weather
  (temperature, wind speed/dir sin-cos, RH, pressure, cloud, dewpoint, precip),
  tide height, calendar (hour/month sin-cos, is_night). In operations these come
  from forecasts; in the backtest they are *observed* values ⇒ **oracle
  weather**, an optimistic upper bound. Ablation A3 measures how much is at stake.
- **Past-only**: border flow (`Flow (m^3/s)--Border`), SBIWTP flow / anomaly /
  deficit (no forecast available).
- **Target**: `H2S` (ppb). Modelled on `log1p(clip(H2S, 0))` by default (heavy
  tail); raw scale is ablation A2.

## 4. Evaluation protocol (the important part)

1. **Same origins as production.** Each `run_ts` in the validation store is a
   forecast origin. Lead `L` targets hour `floor(run_ts) + (L-1) h`; context
   ends at `floor(run_ts) - 1 h`. This gives a **paired, head-to-head**
   comparison against the XGBoost Lean/Evidence rows on identical target hours.
2. **Extended backtest** for more hazard events: synthetic origins every 6 h
   (00/06/12/18 UTC) over 2025-10-01 → present. No XGBoost comparison there
   (its models were trained on that period), but more ≥30 ppb events for the
   TSFMs vs baselines. Zero-shot TSFMs never saw this data, so there is no
   train/test leak for them.
3. **Seed freshness.** Default assumes the last hour is observed (the
   production seed is topped up with near-real-time APCD). `seed_lag_hours`
   re-runs with 3/12/24 h stale context to mirror the pre-fix production behavior.
4. **Product slices.** Report nowcast (1–3), nearcast (4–6), forecast (7–24) separately.
5. **Probabilities from quantiles.** P(H2S > k) is read from the predicted
   quantile function (linear interpolation in log1p space; tails beyond the
   outermost quantile extrapolated log-linearly). Timestral gives 21 levels
   (0.01–0.99); TimesFM gives only 9 (0.1–0.9), so P(>30) for TimesFM relies on
   tail extrapolation. Report it but flag it.
6. **Leakage guard.** Unit test: no context value has `time >= origin`.

## 5. Model matrix

| ID | Model | Inputs | Where |
|---|---|---|---|
| B0 | Persistence (last value) | target | CPU |
| B1 | Seasonal naive (same hour yesterday) | target | CPU |
| B2 | Hour-of-day empirical quantiles over last 28 d | target | CPU |
| X-lean / X-evid | Production XGBoost products | (from validation store) | — |
| T-uni | Timestral 5M / 28M / 123M / 209M, univariate | target | CPU ok (209M slow → Colab) |
| T-cov | Timestral + known-future + past-only covariates | target + cov | CPU / Colab |
| T-mv | Timestral, 3 stations jointly (+cov) | 3 targets + cov | CPU / Colab |
| F-uni / F-cov / F-mv | TimesFM 3.0, same three configs | | Colab GPU recommended |
| (later) | TimesFM 2.5 + XReg, Chronos-2, Moirai 2, TiRex | | optional comparators |

Context length sweep: 512 / 2048 / 8192 h (≈3 wk / 3 mo / 1 yr).

## 6. Phases & deliverables

| Phase | Deliverable | Notebook |
|---|---|---|
| 0. Data + harness | loaders, backtest, metrics, leakage tests; data tour | `00_data_tour.py` |
| 1. Baselines + XGBoost reference | B0–B2 scored on production origins next to X-lean/X-evid | `01_baselines.py` |
| 2. Timestral zero-shot | T-uni size sweep → T-cov → T-mv | `02_timestral.py` |
| 3. TimesFM zero-shot | F-uni → F-cov → F-mv (Colab GPU) | `03_timesfm.py`, `colab/run_backtest_colab.ipynb` |
| 4. Leaderboard + event case studies | paired comparison, bootstrap CIs by run, top-20 events plotted | `04_leaderboard.py` |
| 5. Fine-tuning (only if Phase 2–3 is within reach) | LoRA fine-tune (TimesFM ships a PEFT example) on 2023-11 → 2025-09, test after | `05_finetune.py` |
| 6. Decision memo | go / no-go + integration sketch | `RESULTS.md` |

## 7. Decision criteria (go / no-go)

**Go** (worth a shadow product in Dagster) if a TSFM config, on the
production-origin paired set:

- has **recall@10 ≥ X-lean** at leads 1–6 with false-alarm rate no worse than +2 pp, **or**
- improves the asymmetric-rubric mean cost by ≥10% (bootstrap 95% CI excludes 0), **or**
- is a better **ensemble member**: averaging its P(>k) with X-lean improves Brier at 10/30 ppb.

**No-go** if it only wins on MAE/Spearman while losing hazard recall — that is
the median-regression failure mode and does not help residents.

## 8. Licensing & operations notes

- **Timestral**: code Apache-2.0, weights Modified MIT — check the HF model-card
  license before production; research use is fine.
- **TimesFM 3.0**: self-hosted weights are **non-commercial / non-production
  only**. Research here is fine. A production path would be BigQuery ML
  (`AI.FORECAST` with TimesFM 3.0) or the Apache-2.0 **TimesFM 2.5** weights.
  Keep TimesFM 2.5 in the matrix if 3.0 wins, so a deployable option exists.
- Integration path if "go": a `tsfm` variant emitting the same product row
  schema (`run_ts, product, station, lead_hour, time, variant, h2s_pred, p5,
  p10, p30`) so the existing validation store, heatmap and cascade work
  unchanged. Timestral-5M/28M run on CPU in well under a second per run, which
  fits the code-server.

## 9. Risks

- Oracle-weather optimism (A3 ablation; later: use archived Open-Meteo forecasts).
- Few ≥30 ppb events at IB_CIVIC_CTR / SAN_YSIDRO ⇒ wide CIs; report counts with every rate.
- The validation store overlaps only ~4 months (summer/fall); winter events
  (the main season) come from the extended backtest only.
- TimesFM decile-only output limits P(>30) fidelity.
