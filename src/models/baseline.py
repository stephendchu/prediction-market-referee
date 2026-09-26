"""Layer A: the market baseline, p0 = p_market (spec Sec 7, 26.1).

The reference every model must beat out of sample. It has the same
fit / predict_raw / contributions / params interface as the residual models so
the backtest can run it through the same code path. It ignores features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..pricing import logit


def as_prob(p: Any, name: str = "p_market") -> np.ndarray:
    """1-D float array strictly inside (0, 1)."""
    arr = np.asarray(p, dtype=float).ravel()
    if not np.all((arr > 0) & (arr < 1)):
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return arr


def as_outcome(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y).ravel()
    if len(arr) != n:
        raise ValueError(f"y has {len(arr)} entries for {n} rows")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("y must be 0/1")
    return arr.astype(float)


class MarketBaseline:
    model_type = "market"

    def __init__(self) -> None:
        self.n_obs_: int | None = None

    def fit(self, X: pd.DataFrame | None, p_market: Any, y: Any) -> MarketBaseline:
        p = as_prob(p_market)
        as_outcome(y, len(p))
        self.n_obs_ = len(p)
        return self

    def predict_raw(self, X: pd.DataFrame | None, p_market: Any) -> np.ndarray:
        return as_prob(p_market).copy()

    def contributions(self, X: pd.DataFrame | None, p_market: Any) -> pd.DataFrame:
        p = as_prob(p_market)
        index = X.index if isinstance(X, pd.DataFrame) and len(X) == len(p) else None
        return pd.DataFrame({"market": logit(p), "intercept": 0.0}, index=index)

    def params(self) -> dict[str, Any]:
        return {"model_type": self.model_type}
