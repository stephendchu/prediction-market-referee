"""Market probability from executable prices (spec Sec 6).

Never last trade. The market probability is the midpoint of the executable
YES bid/ask (owner's decision, 2026-09-24); the prices a trade would actually pay
are the asks:

    executable_yes = yes_ask
    executable_no  = 1 - yes_bid
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def midpoint(yes_bid, yes_ask):
    return (np.asarray(yes_bid, dtype=float) + np.asarray(yes_ask, dtype=float)) / 2


def executable_yes(yes_ask):
    return np.asarray(yes_ask, dtype=float)


def executable_no(yes_bid):
    return 1 - np.asarray(yes_bid, dtype=float)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))
