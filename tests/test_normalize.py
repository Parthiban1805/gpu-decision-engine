"""P2 normalization: schema validation, runtime-estimate leakage, synthesis determinism,
and loader end-to-end tests on tiny synthetic raw files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gde.ingestion import load_alibaba2020, load_alibaba2023, load_helios, normalize
from gde.ingestion.runtime_predictor import COLD_START_S, add_runtime_estimates
from gde.ingestion.schema import COLUMNS, FLAG_COLUMNS, validate_jobs


def _frame(n: int = 3, **overrides) -> pd.DataFrame:
    base = {
        "job_id": [f"j{i}" for i in range(n)],
        "dataset": ["t"] * n,
        "submit_ts": np.arange(n, dtype=float) * 100,
        "duration_s": [100.0] * n,
        "runtime_est_s": [50.0] * n,
        "num_gpus": [1.0] * n,
        "gpu_type_req": [None] * n,
        "vram_gb": [8.0] * n,
        "cpu_cores": [4.0] * n,
        "mem_gb": [16.0] * n,
        "job_type": ["training"] * n,
        "priority": [5] * n,
        "deadline_ts": [1000.0] * n,
        "user_group": ["u"] * n,
        "final_status": ["pass"] * n,
        "vram_gb_src": ["derived"] * n,
        "job_type_src": ["derived"] * n,
        "priority_src": ["derived"] * n,
        "deadline_src": ["synthetic"] * n,
        "runtime_est_src": ["derived"] * n,
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    if "deadline_ts" not in overrides:
        df["deadline_ts"] = df["submit_ts"] + 1000.0
    return df[COLUMNS]


def test_schema_accepts_valid_frame() -> None:
    assert validate_jobs(_frame()) == []


def test_schema_catches_each_violation_class() -> None:
    bad = _frame(duration_s=[-1.0, 1.0, 1.0])
    assert any("duration_s" in v for v in validate_jobs(bad))
    bad = _frame(priority=[0, 5, 5])
    assert any("priority" in v for v in validate_jobs(bad))
    bad = _frame(final_status=["ok", "pass", "pass"])
    assert any("final_status" in v for v in validate_jobs(bad))
    bad = _frame(deadline_ts=[-1.0, 2000.0, 2000.0])
    assert any("deadline_ts" in v for v in validate_jobs(bad))
    bad = _frame()
    bad["extra_col"] = 1
    assert any("unexpected columns" in v for v in validate_jobs(bad))
    bad = _frame()
    bad.loc[0, "deadline_src"] = "guessed"
    assert any("deadline_src" in v for v in validate_jobs(bad))


def _est_frame(n: int, durations: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "job_id": [f"j{i}" for i in range(n)],
        "submit_ts": np.arange(n, dtype=float) * 100,
        "duration_s": durations,
        "user_group": ["u1"] * n,
        "job_type": ["training"] * n,
    })


def test_runtime_est_first_row_gets_cold_start_never_own_duration() -> None:
    out = add_runtime_estimates(_est_frame(3, [123.0, 100.0, 100.0]))
    assert out["runtime_est_s"].iloc[0] == COLD_START_S  # not 123


def test_runtime_est_no_leakage_from_future_rows() -> None:
    durations = [100.0, 200.0, 400.0, 800.0]
    base = add_runtime_estimates(_est_frame(4, durations))
    tampered = add_runtime_estimates(_est_frame(4, [100.0, 200.0, 400.0, 999999.0]))
    pd.testing.assert_series_equal(base["runtime_est_s"].iloc[:3], tampered["runtime_est_s"].iloc[:3])
    # and the estimate for row i never equals a function of its own duration alone
    assert base["runtime_est_s"].iloc[1] != 200.0


def test_runtime_est_fallback_levels() -> None:
    df = pd.DataFrame({
        "job_id": ["a1", "a2", "b1"],
        "submit_ts": [0.0, 10.0, 20.0],
        "duration_s": [100.0, 400.0, 50.0],
        "user_group": ["u1", "u1", "u2"],
        "job_type": ["training", "training", "training"],
    })
    out = add_runtime_estimates(df).sort_values("job_id").reset_index(drop=True)
    geo = float(np.expm1((np.log1p(100) + np.log1p(400)) / 2))
    assert out["runtime_est_s"][0] == COLD_START_S        # global first: constant prior
    assert out["runtime_est_s"][1] == pytest.approx(100.0)  # own group prior = [100]
    assert out["runtime_est_s"][2] == pytest.approx(geo)     # new user -> job_type prior


CFG = normalize.load_config()


def test_synthesis_deterministic_and_in_range() -> None:
    core = _frame(50, duration_s=[float(60 + i) for i in range(50)],
                  submit_ts=np.arange(50, dtype=float) * 100)
    core["num_gpus"] = [0.5 if i % 2 else 2.0 for i in range(50)]
    core["gpu_fractional"] = [bool(i % 2) for i in range(50)]
    one = normalize.synthesize(core.copy(), CFG, seed=123)
    two = normalize.synthesize(core.copy(), CFG, seed=123)
    pd.testing.assert_frame_equal(one, two)  # same seed -> identical
    assert "gpu_fractional" not in one.columns
    assert validate_jobs(one) == []
    tr = one[one["job_type"] == "training"]
    slack = (tr["deadline_ts"] - tr["submit_ts"]) / tr["duration_s"]
    assert slack.min() >= CFG["deadline_slack"]["training"]["min"] - 1e-9
    frac = one.iloc[1::2]   # rows built with num_gpus=0.5, gpu_fractional=True
    whole = one.iloc[::2]
    assert (frac["vram_gb_src"] == "derived").all() and (frac["vram_gb"] == frac["num_gpus"] * 16).all()
    assert (whole["vram_gb_src"] == "synthetic").all()
    assert whole["vram_gb"].max() <= whole["num_gpus"].max() * 16 + 1e-9
    assert set(one["deadline_src"]) == {"synthetic"} and set(one["runtime_est_src"]) == {"derived"}


def test_loader_alibaba2023_end_to_end(tmp_path: Path) -> None:
    d = tmp_path / "csv"
    d.mkdir()
    rows = "\n".join([
        "name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time",
        "p1,2000,8192,1,470,,LS,Succeeded,0,5000,10",        # sharing pod, no spec
        "p2,8000,16384,1,1000,V100M32|T4,LS,Succeeded,100,5100,110",
        "p3,4000,8192,1,1000,,BE,Failed,200,5200,120",
        "p4,4000,8192,1,1000,,LS,Running,300,12537496,130",  # censored -> dropped
        "p5,4000,8192,0,0,,LS,Pending,400,12537496,",        # never scheduled -> dropped
    ])
    (d / "openb_pod_list_gpuspec33.csv").write_text(rows, encoding="utf-8")
    df, drops, _ = load_alibaba2023.load(tmp_path)
    assert sorted(df["job_id"]) == ["p1", "p2", "p3"]
    assert drops["phase_not_finished (Running/Pending, censored snapshot)"] == 2
    assert df.loc[df.job_id == "p1", "num_gpus"].item() == pytest.approx(0.47)
    assert df.loc[df.job_id == "p2", "gpu_type_req"].item() == "V100M32"  # first listed
    assert df.loc[df.job_id == "p1", "gpu_type_req"].item() is None
    out = normalize.synthesize(df, CFG, CFG["seeds"]["alibaba2023"])
    assert validate_jobs(out) == []
    assert sorted(out["priority"]) == [2, 8, 8]  # p3 is BE; p1/p2 are LS


def test_loader_helios_end_to_end(tmp_path: Path) -> None:
    d = tmp_path / "data" / "Earth"
    d.mkdir(parents=True)
    rows = "\n".join([
        "job_id,user,vc,gpu_num,cpu_num,node_num,state,submit_time,start_time,end_time,duration,queue",
        "1,u1,vcA,1,4,1,COMPLETED,2020-05-01 00:00:00,2020-05-01 00:00:10,2020-05-01 00:10:10,600,10",
        "2,u1,vcA,0,2,1,COMPLETED,2020-05-01 00:01:00,2020-05-01 00:01:00,2020-05-01 00:02:00,60,0",
        "3,u2,vcB,2,8,1,FAILED,2020-05-01 00:02:00,2020-05-01 00:02:00,2020-05-01 00:12:00,600,0",
        "4,u2,vcB,1,4,1,CANCELLED,2020-05-01 00:03:00,2020-05-01 00:03:00,2020-05-01 00:03:30,30,0",
        "5,u2,vcB,1,4,1,SUSPENDED,2020-05-01 00:04:00,2020-05-01 00:04:00,2020-05-01 00:05:00,60,0",
    ])
    (d / "cluster_log.csv").write_text(rows, encoding="utf-8")
    df, drops, _ = load_helios.load("earth", tmp_path)
    assert sorted(df["job_id"]) == ["1", "3", "4"]
    assert drops["cpu_job (gpu_num == 0)"] == 1
    assert drops["state_not_finished (SUSPENDED/RUNNING)"] == 1
    assert df.loc[df.job_id == "1", "submit_ts"].item() == pytest.approx(0.0)
    assert df.loc[df.job_id == "3", "submit_ts"].item() == pytest.approx(120.0)
    assert set(df["final_status"]) == {"pass", "failed", "killed"}
    assert validate_jobs(normalize.synthesize(df, CFG, CFG["seeds"]["helios_earth"])) == []


def test_loader_alibaba2020_end_to_end(tmp_path: Path) -> None:
    tmp_path.joinpath("pai_job_table.header").write_text(
        "job_name,inst_id,user,status,start_time,end_time", encoding="utf-8")
    tmp_path.joinpath("pai_task_table.header").write_text(
        "job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type",
        encoding="utf-8")
    tmp_path.joinpath("pai_group_tag_table.header").write_text(
        "inst_id,user,gpu_type_spec,group,workload", encoding="utf-8")
    jobs = "\n".join([
        "jn1,i1,u1,Terminated,0,1000",
        "jn2,i2,u1,Failed,10,500",            # no tasks -> fallback job-table span
        "jn3,i3,u2,Running,20,",              # censored -> dropped
        "jn4,i4,u2,Terminated,30,40",         # no GPU request -> dropped
    ])
    tasks = "\n".join([
        "jn1,tensorflow,2,Terminated,100,900,600,10,50,MISC",   # 2 inst x 50% = 1 GPU, shared
        "jn1,ps,1,Terminated,100,950,100,2,,MISC",
        "jn4,tensorflow,1,Terminated,30,40,100,1,0,CPU",
    ])
    tags = "\n".join(["i1,u1,V100,gA,bert", "i2,u1,,gB,", "i3,u2,,gC,", "i4,u2,T4,gD,"])
    for name, body in [("pai_job_table", jobs), ("pai_task_table", tasks), ("pai_group_tag_table", tags)]:
        tmp_path.joinpath(f"{name}.csv").write_text(body, encoding="utf-8")
    df, drops, _ = load_alibaba2020.load(tmp_path)
    assert sorted(df["job_id"]) == ["i1"]
    assert drops["status_not_finished (Running/Waiting, censored)"] == 1
    assert drops["no_gpu_request"] == 2  # i4 asks for 0 GPUs; i2 has no task rows (NaN)
    row = df[df.job_id == "i1"].iloc[0]
    assert row["num_gpus"] == pytest.approx(1.0)          # 2 x 50%
    assert row["cpu_cores"] == pytest.approx(13.0)        # (600*2 + 100)/100
    assert row["duration_s"] == pytest.approx(850.0)      # 950 - 100
    assert bool(row["gpu_fractional"])                    # has a 50% shared instance
    assert row["gpu_type_req"] == "V100"                  # requested spec, not assigned MISC
    out = normalize.synthesize(df, CFG, CFG["seeds"]["alibaba2020"])
    assert validate_jobs(out) == []


@pytest.mark.parametrize("dataset", ["alibaba2020", "alibaba2023",
                                     "helios_earth", "helios_saturn", "helios_uranus", "helios_venus"])
def test_processed_parquets_pass_validation(dataset: str) -> None:
    path = normalize.paths.DATA / "processed" / f"{dataset}.parquet"
    if not path.is_file():
        pytest.skip(f"{path.name} not built yet (run python tasks.py normalize)")
    df = pd.read_parquet(path)
    assert validate_jobs(df) == []
    for col in FLAG_COLUMNS:
        assert df[col].isin(["measured", "derived", "synthetic"]).all()
