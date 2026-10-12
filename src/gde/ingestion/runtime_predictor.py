"""Runtime estimates from past jobs only (PLAN.md §2.2, leakage rule in §0.3).

`runtime_est_s` must look like what a scheduler could know at submit time: for each job,
the estimate is the expanding geometric mean of the durations of PRIOR jobs in its group
(user_group, job_type), falling back to (job_type) and then to all prior jobs. Implemented
with cumsum/cumcount (O(n)); the current row's own duration is excluded arithmetically
(cumsum - own value), so no job ever sees its own or any future job's duration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

FALLBACK_LEVELS: tuple[tuple[str, ...], ...] = (("user_group", "job_type"), ("job_type",), ())
# The single first job of a dataset has no history at all; using its own duration would
# leak the outcome, so it gets this constant prior instead.
COLD_START_S = 3600.0


def _log_mean_estimate(logd: pd.Series, keys: tuple[str, ...], df: pd.DataFrame) -> pd.Series:
    """Geometric mean of prior rows' logd within `keys`; NaN where no prior row exists."""
    if keys:
        grouped = logd.groupby([df[k] for k in keys], dropna=False, sort=False)
        cum = grouped.cumsum()
        prior_n = grouped.cumcount()                # 0-based position == number of prior rows
    else:                                           # global level: plain series arithmetic
        cum = logd.cumsum()
        prior_n = pd.Series(np.arange(len(logd)), index=logd.index)
    prior_sum = cum - logd                          # sum of prior rows only
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.expm1(prior_sum / prior_n.where(prior_n > 0))
    return est


def add_runtime_estimates(
    df: pd.DataFrame,
    value_col: str = "duration_s",
    out_col: str = "runtime_est_s",
    fallback_levels: Sequence[tuple[str, ...]] = FALLBACK_LEVELS,
) -> pd.DataFrame:
    """Add `out_col` (seconds) computed strictly from prior jobs. Deterministic.

    Rows are sorted by (submit_ts, job_id) with a stable sort first, so the group
    "history" is time-ordered and reproducible regardless of input order.
    """
    out = df.sort_values(["submit_ts", "job_id"], kind="mergesort").copy()
    logd = np.log1p(out[value_col].astype(float))
    est = pd.Series(np.nan, index=out.index, dtype=float)
    for keys in fallback_levels:
        if not est.isna().any():
            break
        level = _log_mean_estimate(logd, tuple(keys), out)
        est = est.fillna(level)
    est = est.fillna(COLD_START_S)                 # first job ever: constant prior, never own duration
    out[out_col] = est.clip(lower=1.0)
    return out
