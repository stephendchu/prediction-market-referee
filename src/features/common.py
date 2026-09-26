"""Point-in-time helpers shared by feature builders (spec Sec 7, 26.11).

Every lookup here answers "what was knowable at time t": the latest row whose
availability time is <= t (``pd.merge_asof``, backward, exact matches allowed).
Feature builders never look at a row any other way, which is what the leakage
tests exercise: patching ``asof`` / ``available`` to ignore time must change
the features.

Times are int64 nanoseconds since the epoch (UTC) with ``NAT`` (int64 min) for
missing. ``np.maximum`` over availability times then ignores missing inputs,
and ``<=`` is exact (float64 would round to ~256 ns). Official dates are int64
days since the epoch, also NAT-filled. By convention an int64 column is a time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..config import stable_hash

NAT = np.iinfo(np.int64).min
MIN = 60 * 10**9
DAY = 86_400 * 10**9


def ns(values) -> np.ndarray:
    """Timestamps (tz-aware, or naive = UTC) -> int64 ns; NaT -> NAT."""
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(values, dtype=object if len(values) == 0 else None), utc=True))
    return t.as_unit("ns").asi8.copy()


def days(values) -> np.ndarray:
    """Dates -> int64 days since 1970-01-01; missing -> NAT."""
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(values, dtype=object if len(values) == 0 else None)))
    if t.tz is not None:
        t = t.tz_localize(None)
    x = t.as_unit("ns").asi8
    return np.where(x == NAT, NAT, x // DAY)


def to_timestamps(t: np.ndarray) -> pd.Series:
    """int64 ns (NAT = missing) -> tz-aware UTC timestamps."""
    return pd.Series(np.asarray(t, dtype=np.int64).view("M8[ns]")).dt.tz_localize("UTC")


def tmax(*ts) -> np.ndarray:
    """Element-wise latest availability time; NAT inputs are ignored (all NAT -> NAT)."""
    out = np.asarray(ts[0], dtype=np.int64)
    for t in ts[1:]:
        out = np.maximum(out, np.asarray(t, dtype=np.int64))
    return out


def shift(t, delta: int) -> np.ndarray:
    """t + delta ns, keeping NAT (plain int64 arithmetic would wrap NAT to a real time)."""
    t = np.asarray(t, dtype=np.int64)
    return np.where(t == NAT, NAT, t + np.int64(delta))


def available(t, cutoff) -> np.ndarray:
    """True where an input with availability time `t` may be used at `cutoff`."""
    t = np.asarray(t, dtype=np.int64)
    return (t != NAT) & (t <= np.asarray(cutoff, dtype=np.int64))


def shrink(num, den, prior_rate, prior_n):
    """Empirical-Bayes style shrinkage of num/den toward prior_rate with prior_n pseudo-observations."""
    return (np.asarray(num, dtype=float) + prior_n * np.asarray(prior_rate, dtype=float)) / (
        np.asarray(den, dtype=float) + prior_n)


# ---------------------------------------------------------------------------
# As-of joins
# ---------------------------------------------------------------------------

def _codes(left: pd.DataFrame, right: pd.DataFrame, by: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Joint integer codes for the `by` key columns (-1 = missing key)."""
    nl, nr = len(left), len(right)
    if not by:
        return np.zeros(nl, dtype=np.int64), np.zeros(nr, dtype=np.int64)
    code = np.zeros(nl + nr, dtype=np.int64)
    bad = np.zeros(nl + nr, dtype=bool)
    for col in by:
        both = pd.concat([pd.Series(left[col].to_numpy()), pd.Series(right[col].to_numpy())], ignore_index=True)
        c, uniques = pd.factorize(both)
        bad |= c < 0
        code = pd.factorize(code * (len(uniques) + 1) + c)[0]
    code[bad] = -1
    return code[:nl], code[nl:]


def _fill(values: np.ndarray, j: np.ndarray) -> np.ndarray:
    """values[j], with the column's missing marker where j == -1."""
    hit = j >= 0
    if values.dtype.kind == "i":
        out = np.full(len(j), NAT, dtype=np.int64)
    elif values.dtype.kind in "fb":
        out = np.full(len(j), np.nan)
        values = values.astype(float)
    else:
        out = np.full(len(j), None, dtype=object)
    out[hit] = values[j[hit]]
    return out


def _asof_index(left: pd.DataFrame, right: pd.DataFrame, on: tuple[str, str], by: Sequence[str],
                order: Sequence[str]) -> np.ndarray:
    """Position in `right` of each left row's latest match (right[on] <= left[on]); -1 if none."""
    lon, ron = on
    j = np.full(len(left), -1, dtype=np.int64)
    if len(left) == 0 or len(right) == 0:
        return j
    lt = np.asarray(left[lon], dtype=np.int64)
    rt = np.asarray(right[ron], dtype=np.int64)
    lk, rk = _codes(left, right, by)
    lpos = np.flatnonzero((lt != NAT) & (lk >= 0))
    rpos = np.flatnonzero((rt != NAT) & (rk >= 0))
    if len(lpos) == 0 or len(rpos) == 0:
        return j
    # Ties on the time are broken by `order`: the last row after sorting wins.
    keys = [np.asarray(right[c].to_numpy()[rpos]) for c in reversed(order)]
    rpos = rpos[np.lexsort([*keys, rt[rpos]])]
    lpos = lpos[np.argsort(lt[lpos], kind="stable")]
    L = pd.DataFrame({"_t": lt[lpos], "_k": lk[lpos]})
    R = pd.DataFrame({"_t": rt[rpos], "_k": rk[rpos], "_j": rpos})
    m = pd.merge_asof(L, R, on="_t", by="_k", direction="backward", allow_exact_matches=True)
    j[lpos] = m["_j"].fillna(-1).to_numpy(dtype=np.int64)
    return j


def _asof_raw(left: pd.DataFrame, right: pd.DataFrame, *, on: tuple[str, str], by: Sequence[str] = (),
              cols: Sequence[str] = (), order: Sequence[str] = ()) -> pd.DataFrame:
    j = _asof_index(left, right, on, by, order)
    out = {c: _fill(right[c].to_numpy(), j) for c in cols}
    out["at"] = _fill(np.asarray(right[on[1]], dtype=np.int64), j)
    return pd.DataFrame(out)


def asof(left: pd.DataFrame, right: pd.DataFrame, *, on: tuple[str, str], by: Sequence[str] = (),
         cols: Sequence[str] = (), order: Sequence[str] = ()) -> pd.DataFrame:
    """For each left row, `cols` of the latest right row with right[on[1]] <= left[on[0]] and equal `by` keys.

    Aligned to left by position; "at" is the matched row's time (NAT if none). Rows
    with a missing key or time never match. This is THE availability filter.
    """
    return _asof_raw(left, right, on=on, by=by, cols=cols, order=order)


def asof_sums(queries: pd.DataFrame, events: pd.DataFrame, *, by: Sequence[str], time: str, cols: Sequence[str],
              upper: str, lower: str | None = None) -> pd.DataFrame:
    """Sums of event `cols` with `by` keys equal and time in (queries[lower], queries[upper]].

    NaN stats count as 0; "n" is the number of events and "at" the latest event
    time in the window (NAT if empty). The upper bound goes through `asof`.
    """
    by = list(by)
    ev = events.loc[np.asarray(events[time], dtype=np.int64) != NAT, [*by, time, *cols]]
    if not by:
        ev, by = ev.assign(_all=0), ["_all"]
        queries = queries.assign(_all=0)
    ev = ev.dropna(subset=by)
    agg = ev.groupby([*by, time], sort=True)[list(cols)].sum().assign(n=ev.groupby([*by, time]).size())
    stats = [*cols, "n"]
    cum = agg.astype(float).groupby(level=list(range(len(by)))).cumsum().reset_index()
    hi = asof(queries, cum, on=(upper, time), by=by, cols=stats)
    out = hi[stats].fillna(0.0)
    if lower is not None:
        lo = _asof_raw(queries, cum, on=(lower, time), by=by, cols=stats)
        out = out - lo[stats].fillna(0.0)
    out.loc[np.asarray(queries[upper], dtype=np.int64) == NAT, stats] = 0.0
    out["at"] = np.where(out["n"].to_numpy() > 0, hi["at"].to_numpy(), NAT)
    return out


def last_n_sums(frame: pd.DataFrame, *, by: str, order: Sequence[str], cols: Sequence[str], n: int) -> pd.DataFrame:
    """Per row, sums of `cols` over the last `n` rows of its `by` group (itself included) in `order`."""
    f = frame.sort_values([by, *order], kind="stable")
    cs = f.groupby(by)[list(cols)].cumsum()
    lag = cs.groupby(f[by]).shift(n).fillna(0.0)
    return (cs - lag).reindex(frame.index)


# ---------------------------------------------------------------------------
# Deterministic randomness for negative controls (26.16)
# ---------------------------------------------------------------------------

def _hash_ints(values: Sequence, salt) -> np.ndarray:
    """52-bit integer (exact in float64) from sha256 of (value, salt), via config.stable_hash."""
    uniq, inv = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    k = np.array([float(int(stable_hash([v, salt]), 16) >> 12) for v in uniq], dtype=float)
    return k[inv] if len(uniq) else np.zeros(0)


def hash_uniform(values: Sequence, salt=None) -> np.ndarray:
    """Stable uniform in [0, 1) per value; independent of pandas/numpy versions and PYTHONHASHSEED."""
    return _hash_ints(values, salt) / 2.0**52


def hash_normal(values: Sequence, salt=None) -> np.ndarray:
    """Stable standard normal per value (the uniform is offset by half a step, so never 0 or 1)."""
    return ndtri((_hash_ints(values, salt) + 0.5) / 2.0**52)
