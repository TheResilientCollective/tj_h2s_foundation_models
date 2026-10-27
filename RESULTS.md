# Results (interim, 2026-10-26)

Zero-shot, CPU, scored **paired** on the 593 production origin hours
(2026-06-28 → 2026-10-25), against the production XGBoost forecasts for the
identical (origin, station, lead) cells. Rubric cost is the asymmetric
verdict cost from `tj_h2s_prediction` (lower is better). Δ cost is a
run-level paired bootstrap vs `xgboost_lean`, as a 95% CI.

Reproduce: `scripts/queue_cpu_runs.sh`, then `notebooks/04_leaderboard.py`.

## Headline

| | nowcast 1–3 h | nearcast 4–6 h | forecast 7–24 h |
|---|---|---|---|
| **Rubric mean cost** — XGBoost Lean | 0.067 | 0.065 | 0.069 |
| Timestral-5M (H2S only) | **0.035** | **0.038** | **0.058** |
| TimesFM 3.0 (H2S only) | 0.038 | 0.041 | 0.058 |
| Timestral-5M + forecast weather | 0.036 | 0.039 | 0.059 |
| 28-day hour-of-day climatology | 0.049 | 0.045 | 0.059 |
| **ROC-AUC, ≥10 ppb** — XGBoost Lean | 0.78 | 0.66 | 0.69 |
| Timestral-5M (H2S only) | 0.95 | 0.96 | 0.92 |
| **Spearman** — XGBoost Lean | 0.49 | 0.36 | 0.32 |
| Timestral-5M (H2S only) | 0.79 | 0.71 | 0.67 |

Every foundation-model configuration has lower rubric cost than XGBoost Lean
at every product, and every 95% CI excludes 0. Timestral-5M and TimesFM 3.0
are statistically tied.

## Findings

1. **Weather inputs add almost nothing zero-shot.** Timestral-5M scores about
   the same with no weather, with the archived Open-Meteo forecast for the
   horizon (operational), and with observed weather (oracle). Nowcast Spearman:
   0.794 / 0.791 / 0.798. The H2S history carries the signal, so the
   operational setting gives up nothing.
2. **The advantage holds with stale data.** With the newest observation 6 h or
   24 h old, nowcast skill drops (Spearman 0.79 → 0.67 → 0.63; cost 0.035 →
   0.048 → 0.049). That's about the level of the climatology baseline, and
   still significantly better than XGBoost Lean at every product. Fresh data
   is what lifts Timestral above climatology at short leads.
3. **The models rank risky hours well but under-call them at p > 0.5.**
   Recall@10 ppb at the 0.5 cutoff is 0–6% beyond lead 3, despite
   AUC ≥ 0.91. Probabilities for rare events are low but well-ordered, so the
   fix is choosing a cutoff per threshold (and lead) at an acceptable
   false-alarm rate. Retraining isn't needed. **Next analysis.**
4. **XGBoost Lean is beaten by a 28-day hour-of-day climatology** on rubric
   cost (Δ −0.010 to −0.021, CIs exclude 0) and on AUC. Its MAE is the highest
   of any non-naive model, and Evidence is worse still. In this quiet season
   the production model over-predicts. Worth a look in `tj_h2s_prediction`
   independently of this study.

## Caveats

- **Summer–fall only** (the validation store starts 2026-06-28). Hazard counts
  are small: ≥30 ppb cells number 7 (nowcast), 14 (nearcast) and about 80
  (forecast). Winter performance needs the extended backtest. That run uses
  oracle weather only, since the forecast archive starts 2026-04-10.
- Rubric cost is dominated by the many quiet hours. A model that rarely calls
  hazards scores well on it, so read it alongside AUC, Brier and recall at a
  tuned cutoff.
- The foundation-model rows use observed H2S from modeldata. The XGBoost rows
  use `actual_h2s` from the validation store. These differ by about one
  ≥10 ppb cell per product.

## Status of the go / no-go criteria (PLAN §7)

- ≥10% lower mean rubric cost with CI excluding 0: **met** at every product,
  including with 24 h stale data.
- Recall@10 ≥ XGBoost Lean at leads 1–6 with ≤ +2 pp FAR: **not met at the
  0.5 cutoff.** Pending the cutoff-calibration analysis.
- Ensemble Brier improvement: not yet tested.

## Still to run

TimesFM 3.0 with weather covariates: on CPU it didn't finish 593 origins in
2 h (the extra covariate channels make it much slower than the univariate
run), so it moves to the Colab GPU along with the larger models, longer
contexts and the winter backtest (`colab/run_backtest_colab.ipynb`, already
listed there). Given finding 1, weather is unlikely to change TimesFM's
standing much. Then the cutoff calibration, which picks
P(>k) thresholds on a tuning window and evaluates them on a held-out one.
