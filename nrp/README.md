# GPU backtests on NRP

`tsfm-backtest-job.yaml` runs the GPU part of the backtest matrix as one
Kubernetes **Indexed Job** in `ucsd-center4health`. Pod *N* runs line *N* of
the `runs.txt` ConfigMap on its own GPU (`pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`)
and uploads its result to the object store. Results then go through the same
`04_leaderboard` notebook as the CPU runs.

## Prerequisites

* `kubectl` with the `nautilus` context (same as `tijuana-dispersion-experiments/nrp/DEPLOYMENT.md`).
  The Job uses `backoffLimitPerIndex`, which needs Kubernetes ≥ 1.29 (check with `kubectl version`).
* The `object-store-credentials` secret already used by the dispersion
  workers. It must provide `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `S3_ENDPOINT_URL` and `DAGSTER_S3_BUCKET`. Results land in
  `s3://$DAGSTER_S3_BUCKET/tj_h2s_foundation_models/results/`. To use another
  bucket, add a `RESULTS_BUCKET` env var to the Job.
* **Only while the repo is private:** a read-only GitHub token so pods can clone it.
  ```bash
  kubectl create secret generic tsfm-github -n ucsd-center4health --from-literal=GH_TOKEN=<fine-grained read token>
  ```

## Run

```bash
kubectl apply -n ucsd-center4health -f nrp/tsfm-backtest-job.yaml
kubectl get pods -n ucsd-center4health -l job-name=tsfm-h2s-backtest -w
kubectl logs -n ucsd-center4health -l job-name=tsfm-h2s-backtest --prefix -f --max-log-requests 11
```

Each pod installs the package (~2–3 min), downloads model weights and the
public data, then runs one configuration. Forecast-weather runs also fetch the
archived forecasts they need (~5 min). Runs are independent: a failed index is
retried once and does not stop the others.

## Collect results

```bash
uv sync --extra nrp
uv run python scripts/nrp_results.py fetch    # -> results/*.parquet, skipping files you have
uv run marimo edit notebooks/04_leaderboard.py
```

## Change the runs

Edit `runs.txt` in the ConfigMap. Lines are `scripts/run_backtest.py`
arguments; `--device cuda` is added. Keep `completions` equal to the number of
non-comment lines. Then re-apply. A Job's spec is immutable, so delete the old
one first:

```bash
kubectl delete job tsfm-h2s-backtest -n ucsd-center4health
kubectl apply -n ucsd-center4health -f nrp/tsfm-backtest-job.yaml
```

`parallelism: 4` caps how many GPUs the job holds at once. NRP expects GPU
pods to keep their GPU busy, so don't raise it far, and don't use this job
for CPU-only runs: the baselines run fine locally.
