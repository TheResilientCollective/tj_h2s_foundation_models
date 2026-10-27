import importlib.util
import shlex
from pathlib import Path

import pytest
import yaml

from tsfm_h2s.data import atomic_write
from tsfm_h2s.models import model_name

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_backtest", ROOT / "scripts" / "run_backtest.py")
run_backtest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_backtest)


def tag(argline: str) -> str:
    args = run_backtest.build_parser().parse_args(shlex.split(argline))
    return run_backtest.run_tag(model_name(args.model), args)


def nrp_runs() -> list[str]:
    configmap = next(yaml.safe_load_all((ROOT / "nrp" / "tsfm-backtest-job.yaml").read_text()))
    return [l for l in configmap["data"]["runs.txt"].splitlines() if l.strip() and not l.strip().startswith("#")]


def test_every_nrp_run_writes_its_own_file():
    tags = [tag(line) for line in nrp_runs()]
    assert len(tags) == len(set(tags)), tags


def test_existing_result_names_are_unchanged():
    # files written before run_tag existed must still be recognised by --skip-existing
    assert tag("--model timestral-5M --seed-lag 6") == "timestral_5M__univariate__ctx2048__log1p__lag6__production"
    assert tag("--model timesfm --mode covariates --weather forecast") == (
        "timesfm_3.0__covariates__wxforecast__ctx2048__log1p__lag0__production")


def test_smoke_tests_and_windows_never_share_a_name():
    full = tag("--model climatology")
    assert tag("--model climatology --limit 3") != full
    a = tag("--model climatology --origins extended --start 2025-10-01")
    b = tag("--model climatology --origins extended --start 2025-11-01")
    c = tag("--model climatology --origins extended --start 2025-10-01 --every 3")
    assert len({a, b, c}) == 3 and a.endswith("__extended-20251001-dataend-6h")


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    target = tmp_path / "out.parquet"

    def boom(tmp):
        tmp.write_text("partial")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(target, boom)
    assert not target.exists() and list(tmp_path.iterdir()) == []
    atomic_write(target, lambda tmp: tmp.write_text("done"))
    assert target.read_text() == "done"
