# Review protocol

This is Section 26 of the project's specification, the review criteria and research protocol that
the referee (`quant review`, `src/reports/diagnostics.py`) applies. The rest of the specification
is not published; where code or records cite another section ("Sec 14", "spec Section 8"), that
section is not in this repository.

The text below is copied verbatim, with two exceptions, each marked with square brackets: 26.8
names a specific exchange in the original, and 26.20 is replaced by a note.

How this repository applies it:
- **The final holdout is never evaluated** (26.3). The referee reports 26.1 as INSUFFICIENT
  ("holdout not evaluated"), so no run here can reach RESEARCH-PROMISING. Beside it, an
  informational row, **26.1-wf**, applies the 26.1 test to the pooled walk-forward evaluation
  periods. It is not a substitute for the holdout test.
- **26.18** (paper trading) needs a frozen strategy and live paper trades, so it stays
  INSUFFICIENT here.
- **Everything here is synthetic.** The criteria are exercised on generated worlds with known
  answers, never on market data.

---

## 26. Review Criteria and Research Protocol

Acceptance criteria for claiming the model improves on the market baseline. MUST be
committed before examining final holdout results. The reviewer evaluates against
these rules, not criteria invented after seeing results.

### 26.1 Primary Benchmark

Benchmark: point-in-time executable market probability. Primary metric: Brier score.

```
BSS = 1 - (Brier_model / Brier_market)
```

Primary comparison: paired bootstrap over games, 10,000 resamples, game-level
resampling (the GAME is the unit, not the snapshot), 95% CI.

Acceptance: `BSS >= 0.005` AND bootstrap mean delta_brier > 0 AND 95% CI lower bound
> 0, on the final untouched holdout. Also report raw Brier delta, BSS, log-loss
delta, calibration error. Do not substitute accuracy.

### 26.2 Walk-Forward Stability

At least five chronologically distinct evaluation periods when data allows.
Candidates must beat the market on Brier in >= 4 of 5 periods, have positive
aggregate BSS, and no single period may account for > 50% of aggregate improvement.
If five periods can't be constructed, report the limitation explicitly.

### 26.3 Final Holdout

Reserve the most recent complete evaluation period as FINAL_HOLDOUT. It MUST NOT be
used for feature, hyperparameter, calibration, threshold, uncertainty,
transaction-cost, or model selection. Evaluate only after everything is frozen.
Record git commit, model config hash, feature-set hash, data snapshot/version, and
freeze timestamp before unlocking. A failed holdout remains a failed experiment.

### 26.4 Hypothesis Registration

Every feature family registered BEFORE results are examined: hypothesis_id,
description, mechanism, exact feature definitions, expected direction, minimum
sample, evaluation periods, primary metric, git commit, registration timestamp.
Exploratory discoveries MUST be labeled EXPLORATORY and cannot become confirmed
features until tested on new unseen data.

### 26.5 Multiple Hypothesis Testing

Report raw p-value and Benjamini-Hochberg FDR-adjusted q-value per experiment
family. Default FDR = 0.10. Significance alone is insufficient. Registry must
contain successful AND unsuccessful experiments.

### 26.6 Edge-Bucket Validation

Buckets: <0%, 0–2%, 2–4%, 4–6%, 6–8%, >8%. Per bucket: observations, actual trades,
mean market/predicted/conservative probability, empirical outcome frequency,
predicted edge, realized edge, gross ROI, net ROI, bootstrap 95% CI for realized
edge. Minimum for inferential claims: >= 200 independent games per bucket
(preferred >= 500). Below 200: label INSUFFICIENT SAMPLE. Do not merge buckets after
seeing performance.

### 26.7 Edge Monotonicity

Spearman rank correlation between edge-bucket rank and realized edge. Also fit
`realized_outcome ~ market_probability + predicted_edge`; predicted-edge coefficient
should be positive. Diagnostics, not standalone acceptance criteria. Failure
requires investigation even with positive P&L.

### 26.8 Transaction Costs

Reconstruct using the [venue's] fee schedule applicable at the historical timestamp.
Implement `FeeSchedule` as a versioned interface. Store per trade: fee_schedule_id,
fee_schedule_effective_date, maker_or_taker, calculated_fee. Do NOT hard-code one
schedule. Report results before fees, after fees, after fees + slippage. Must be
profitable AFTER fees and modeled slippage.

### 26.9 Slippage and Book Depth

When historical depth exists, walk the book for quantity Q and compute
volume-weighted execution price. Fallback: size <= 10% of displayed top-of-book
depth; execution price = executable ask + 25% of quoted spread (symmetric for
sells). Sensitivity at 0%, 25%, 50%, 100% spread penalty. A strategy whose
profitability disappears at 25% is execution-sensitive. Never assume midpoint
execution.

### 26.10 Snapshot Policy

ONE canonical decision snapshot per game/contract: T-30 minutes before scheduled
start. Select the most recent valid quote at or before T-30m; max quote age 5
minutes. Valid quote: two-sided, bid < ask, nonzero displayed size, spread <= 10
percentage points. No valid quote within 5 minutes of T-30m → NO OBSERVATION. Do not
reach backward indefinitely. Secondary snapshots (T-120, T-60, T-10) are sensitivity
analyses only. Never select the snapshot that produces the best result.

### 26.11 Information Cutoff

For the T-30m model: `feature_timestamp <= quote_timestamp`, not merely `<= game_start`. Applies to lineups, injuries, starters, weather, officials, consensus,
news, transactions. Features without a trustworthy availability timestamp are
excluded or given a conservative timestamp proving prior publicity. Leakage tests
must inject future information and verify rejection.

### 26.12 Uncertainty Model

The haircut may not be a constant chosen to improve ROI. Implement independently
from trading results: conformal/empirical calibration error, ensemble dispersion,
feature-distribution distance, missing-information penalties. Produce p_model,
lower_bound, upper_bound. YES: `p_conservative = lower_bound`; NO:
`p_conservative_no = 1 - upper_bound`. Evaluate coverage: nominal 90%, observed
87–93% with sufficient sample. If it improves P&L but fails coverage, reject it.

### 26.13 Calibration

>= 10 probability bins. Report predicted probability, empirical frequency,
observations, CI, ECE, MCE, calibration slope and intercept. Evaluate model
probability and conservative probability separately.

### 26.14 P&L Concentration Diagnostics

Group P&L by season, walk-forward period, month, team, market probability band,
predicted edge band, home/away, feature availability state. Report largest trade,
top 5, top 10 / total P&L; best team, best month / total. Flag top 10 trades > 40%
of positive P&L or one team > 20%. Rerun excluding the ten most profitable trades.

### 26.15 Edge Decay

Per evaluation period: predicted edge, realized edge, net ROI. Fit `realized_edge ~ time`; report slope and uncertainty. DECAY WARNING if edge falls materially across
periods or the most recent two periods both underperform the aggregate.

### 26.16 Negative Controls

Include features that should carry no information: randomized umpire assignment,
permuted weather, random feature, team-ID hash. Persistent out-of-sample improvement
from negative controls → assume leakage, selection bias, bad significance
calculation, or bugs until proven otherwise.

### 26.17 Market-Shuffle Test

Permute outcomes within constrained groups (preserving market probabilities and
features) >= 1,000 times; rerun the full evaluation. Real result should sit in the
favorable tail. Report empirical permutation p-value. Sanity test, not a substitute
for 26.1.

### 26.18 Paper-Trading Acceptance

After freeze: >= 200 qualifying paper trades AND preferably >= 60 calendar days.
Compare Brier, calibration, average predicted edge, realized edge, net ROI, trade
frequency, execution cost against frozen-backtest expectations. Paper BSS >= 50% of
backtest BSS and > 0. Realized net edge >= 50% of expected. Trade frequency within
50–150% of expected. Outside tolerance: STATUS = RESEARCH FAILURE / INVESTIGATE. Do
not loosen thresholds retrospectively.

### 26.19 Strategy Acceptance States

Exactly one: REJECTED, INCONCLUSIVE, RESEARCH-PROMISING, PROSPECTIVELY-VALIDATED.

REJECTED: holdout BSS <= 0, OR severe calibration failure, OR leakage, OR net edge
disappears under baseline costs.

INCONCLUSIVE: positive but insufficient sample, OR CI includes zero, OR stability
not met.

RESEARCH-PROMISING: BSS >= 0.005, bootstrap CI excludes zero, >= 4/5 periods
improve, buckets broadly monotonic, holdout survives costs, no material
leakage/concentration problem.

PROSPECTIVELY-VALIDATED: all of the above AND 26.18 satisfied.

No stronger claim may be made.


### 26.20 Market-Behavior Evaluator (STUB)

[Replaced for publication. The original is a placeholder for the evaluator of a second strategy
family, `market_behavior`, whose governing text is not published, followed by an amendment about a
dataset this repository does not ship. This repository implements no part of it: configuring
`strategy.family: market_behavior` raises `NotImplementedError` (`src/config.py`).]
