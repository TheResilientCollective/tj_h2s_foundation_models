"""Move backtest results between results/ and the NRP object store.

  upload: results/*.parquet -> s3://$DAGSTER_S3_BUCKET/$RESULTS_PREFIX (inside the NRP pod)
  fetch:  the reverse, skipping files you already have (on your machine)

Credentials and endpoint come from the namespace's ``object-store-credentials``
secret (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_ENDPOINT_URL /
DAGSTER_S3_BUCKET); lowercase ``access_key_id`` / ``secret_access_key`` keys are
accepted too. Override the bucket with RESULTS_BUCKET.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_PREFIX = "tj_h2s_foundation_models/results/"


def _env(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def client():
    return boto3.client(
        "s3",
        endpoint_url=_env("S3_ENDPOINT_URL"),
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID", "access_key_id"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY", "secret_access_key"),
        region_name=_env("AWS_DEFAULT_REGION") or "us-east-1",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["upload", "fetch"])
    ap.add_argument("--bucket", default=_env("RESULTS_BUCKET", "DAGSTER_S3_BUCKET"))
    ap.add_argument("--prefix", default=os.environ.get("RESULTS_PREFIX", DEFAULT_PREFIX))
    args = ap.parse_args()
    if not args.bucket:
        raise SystemExit("no bucket: set RESULTS_BUCKET or DAGSTER_S3_BUCKET, or pass --bucket")
    s3 = client()

    if args.action == "upload":
        for path in sorted(RESULTS.glob("*.parquet")):
            s3.upload_file(str(path), args.bucket, args.prefix + path.name)
            print(f"uploaded s3://{args.bucket}/{args.prefix}{path.name}")
        return

    RESULTS.mkdir(exist_ok=True)
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=args.prefix)
    for obj in (o for page in pages for o in page.get("Contents", [])):
        name = obj["Key"].rsplit("/", 1)[-1]
        dest = RESULTS / name
        if name.endswith(".parquet") and not dest.exists():
            s3.download_file(args.bucket, obj["Key"], str(dest))
            print(f"fetched {dest}")


if __name__ == "__main__":
    main()
