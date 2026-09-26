# prediction-market-referee

This is a pre-registered evaluation harness for probabilistic forecasts against a market baseline,
with leakage detection, walk-forward testing, locked holdouts, and fixed acceptance states. It takes a
model from ingest through point-in-time features, walk-forward training and a backtest at executable
prices; a referee then scores the run against a protocol fixed before any result
([docs/PROTOCOL.md](docs/PROTOCOL.md)) and assigns exactly one of four states. It ships with two
synthetic baseball worlds whose answers are known, so you can check that the harness reaches the
verdict it should.

## Quick start

Requires Python 3.12. Run everything from the repository root. Install the package editable (`-e`):
it reads its configuration from `config/` in the checkout.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expected last line (the time varies):

```
995 passed, 1 warning in 149.84s (0:02:29)
```

The warning is a `RuntimeWarning: overflow encountered in exp` from a model-fitting test on extreme
inputs.

Then run both worlds, each in its own data directory:

```bash
for W in null_v1 planted_v2; do
  export QUANT_DATA_DIR=runs/$W/data QUANT_SYNTHETIC_WORLD=$W
  python -m src.backtest.registry register mlb synthetic
  quant ingest   --sport mlb --start 2023-01-01
  quant features --sport mlb --date 2026-09-24
  quant train    --sport mlb
  quant backtest --sport mlb
  quant review   --sport mlb
done
```

- `register` records the hypotheses before anything is fitted.
- `ingest` generates the world. All data is produced here, at runtime; the repository ships none.
- `review` prints the full criteria table, then writes `runs/<world>/artifacts/mlb_review.md` and
  `mlb_review.json`.

The generated data takes about 0.6 GB per world under `runs/`, which git ignores.

Run ids, lineage ids and timestamps differ on every run. The status lines, criterion states, trades
and the tables below should match exactly. The last digits of some floating-point values in the
review can vary with the number of CPU threads (seen at the 15th significant digit).

### Expected output: null_v1

The end of `quant review`:

```
Status rules applied (26.19):
  - not RESEARCH-PROMISING: 26.1 INSUFFICIENT SAMPLE
  - not RESEARCH-PROMISING: 26.2 FAIL
  - not RESEARCH-PROMISING: 26.7 FAIL
  - not RESEARCH-PROMISING: 26.9 FAIL
  - not RESEARCH-PROMISING: 26.12 FAIL
  - not RESEARCH-PROMISING: 26.14 FAIL
Review flags (ADR 0002):
  - this review raised or cleared no flag
  - open flags: 0
╭──────────────────────────────────────────────────────────────────────────────╮
│ SYNTHETIC DATA — pipeline demonstration, not evidence about any real market  │
╰──────────────────────────────────────────────────────────────────────────────╯
STATUS: INCONCLUSIVE
```

The edge buckets (26.6) come from `runs/null_v1/artifacts/mlb_review.md`, section
`26.6: edge_buckets`, with some columns left out.
- **Predicted edge** is the model's probability minus the market's.
- **Realized edge** is the outcome minus the market's probability, with a game-level bootstrap 95% CI.
- **Trades** are T-30 trades in the baseline cost scenario.

| bucket | observations | games | trades | mean predicted edge | realized edge | CI low | CI high | net ROI | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| <0% | 10955 | 6116 | 0 | -0.01139 | -0.007702 | -0.01768 | 0.002121 | n/a | OK |
| 0-2% | 1485 | 1109 | 0 | 0.008865 | 0.006498 | -0.02355 | 0.03693 | n/a | OK |
| 2-4% | 659 | 495 | 0 | 0.02935 | -0.006692 | -0.05247 | 0.03863 | n/a | OK |
| 4-6% | 195 | 167 | 0 | 0.04713 | -0.05964 | -0.1376 | 0.01684 | n/a | INSUFFICIENT SAMPLE |
| 6-8% | 205 | 184 | 0 | 0.06777 | 0.002732 | -0.07129 | 0.07532 | n/a | INSUFFICIENT SAMPLE |
| >8% | 406 | 277 | 7 | 0.09504 | -0.03534 | -0.0986 | 0.02618 | -0.02009 | OK |

### Expected output: planted_v2

```
Status rules applied (26.19):
  - not RESEARCH-PROMISING: 26.1 INSUFFICIENT SAMPLE
Review flags (ADR 0002):
  - this review raised or cleared no flag
  - open flags: 0
╭──────────────────────────────────────────────────────────────────────────────╮
│ SYNTHETIC DATA — pipeline demonstration, not evidence about any real market  │
╰──────────────────────────────────────────────────────────────────────────────╯
STATUS: INCONCLUSIVE
```

| bucket | observations | games | trades | mean predicted edge | realized edge | CI low | CI high | net ROI | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| <0% | 5113 | 3189 | 0 | -0.01102 | 0.01694 | -0.0002318 | 0.0341 | n/a | OK |
| 0-2% | 3450 | 2551 | 0 | 0.009396 | -0.002096 | -0.02189 | 0.0181 | n/a | OK |
| 2-4% | 2070 | 1537 | 0 | 0.02885 | 0.06454 | 0.03873 | 0.09084 | n/a | OK |
| 4-6% | 1039 | 782 | 0 | 0.04943 | 0.06333 | 0.02656 | 0.1009 | n/a | OK |
| 6-8% | 891 | 653 | 0 | 0.06995 | 0.0918 | 0.05116 | 0.1319 | n/a | OK |
| >8% | 1342 | 800 | 179 | 0.103 | 0.112 | 0.07681 | 0.1472 | 0.4107 | OK |

### Runtime and memory

These were measured from a fresh clone, covering the install, the test suite and both worlds:

| Machine | Install | Tests | Both worlds | Total |
|---|---:|---:|---:|---:|
| Desktop, AMD Ryzen 9 7900X (12 cores), Linux on WSL2 | 12 s | 158 s | 195 s | 6.1 min |
| Same machine limited to 4 cores (`taskset -c 0-3`) | 12 s | 150 s | 261 s | 7.1 min |

- The 4-core row stands in for a laptop; no laptop was measured.
- The runs use the full configured schedule: four synthetic seasons, 2023 to 2026, through 2026-09-24.
  No reduction was needed to stay under 10 minutes.
- Peak resident memory is about 0.6 GB for the test suite and about 5 GB for each world run.

## What the two worlds show

Both worlds come from the same seed and the same random streams, so they have the same games,
lineups, weather, outcomes and true win probabilities (`p_true`). Only the market's prices differ.
- **null_v1** (`synthetic-mlb-0.3.0`) is the control. Its market knows every player's latent skill
  exactly, apart from four small inefficiencies fixed in the generator before any result was seen.
  The referee must not reach a positive state on it.
- **planted_v2** (`synthetic-mlb-0.3.0-planted`) adds a known, modest mispricing to the same
  market, also fixed in advance (`WorldParams` in `src/sports/synthetic.py`).

`p_true` is written to a separate, labelled truth file. No model sees it. The referee uses it only to
check interval coverage (26.12).

The figures below are the N3b record ([ADR 0003](docs/decisions/0003-generator-random-streams.md),
"Item 4"), and the Quick start reproduces them.
- **Scope:** both worlds evaluate 13,905 observations in 7,429 games at T-30, over six walk-forward
  periods (2023H2 to 2026H1).
- **The model** takes the market's log-odds as a fixed offset and learns only a penalised logistic
  residual on top of it, so with no signal in the features it collapses to the market.
- **LightGBM** is trained alongside it and is selected only if it wins at least 4 of the 6 periods.
  It did so in neither world.

| | null_v1 | planted_v2 |
|---|---|---|
| Status | INCONCLUSIVE | INCONCLUSIVE |
| Brier score, model vs market | 0.246148 vs 0.245407 | 0.246721 vs 0.249848 |
| Walk-forward BSS vs market (26.1-wf) | −0.003021 (FAIL) | +0.012516 (PASS) |
| Brier improvement, game-bootstrap 95% CI | −0.001284 to −0.000201 | +0.002010 to +0.004209 |
| Periods improved (26.2) | 0 of 6 | 5 of 6 |
| T-30 trades, baseline costs (26.8) | 7, net −14.01 on 697.58 staked | 179, net +7,347.79 on 17,888.93 staked |
| Coverage of `p_true` by the 90% interval (26.12) | 0.823 (FAIL) | 0.875 (PASS) |
| Outcome-shuffle p-value (26.17) | 0.1499 (FAIL) | 0.000999 (PASS) |

Every criterion, by state:

| Criterion | null_v1 | planted_v2 |
|---|---|---|
| 26.1 Final-holdout BSS vs market | INSUFFICIENT SAMPLE (holdout locked) | INSUFFICIENT SAMPLE (holdout locked) |
| 26.1-wf Walk-forward pooled BSS (informational) | FAIL | PASS |
| 26.2 Beats the market in most periods | FAIL | PASS |
| 26.3 Final holdout untouched | PASS | PASS |
| 26.4 Hypotheses registered before testing | PASS | PASS |
| 26.5 BH-FDR q-values per family | PASS | PASS |
| 26.6 Realized edge > 0 in every sampled bucket above the minimum | FAIL | PASS |
| 26.7 Edge buckets monotonic | FAIL | PASS |
| 26.8 Profitable after fees and slippage | FAIL | PASS |
| 26.9 Survives the spread-penalty scenarios | FAIL | PASS |
| 26.10 Canonical snapshot, quote age and spread | PASS | PASS |
| 26.11 No look-ahead | PASS | PASS |
| 26.12 Interval coverage | FAIL | PASS |
| 26.13 No severe calibration failure | PASS | PASS |
| 26.14 P&L not concentrated | FAIL | PASS |
| 26.15 No edge decay | INSUFFICIENT SAMPLE | FAIL |
| 26.16 Negative controls do not work | PASS | PASS |
| 26.17 Outcome-shuffle test | FAIL | PASS |
| 26.18 Paper trading | INSUFFICIENT SAMPLE | INSUFFICIENT SAMPLE |

### planted_v2 passes every gate except the locked holdout

- **The gates** are the criteria RESEARCH-PROMISING requires: 26.1, 26.2, 26.7, 26.9, 26.12 and 26.14.
- **26.1 is INSUFFICIENT.** It is the holdout test, and the holdout is never evaluated here. It is
  the status's only reason (INCONCLUSIVE).
- **26.1-wf passes.** This walk-forward stand-in scores a BSS of 0.0125 against a pre-registered bar
  of 0.010.
- **Not every criterion passes.**
  - 26.15 fails. Of the three periods with trades, the two most recent realized less edge than the
    aggregate.
  - 26.18 is INSUFFICIENT, because there are no paper trades.

### null_v1 fails the criteria that test for an edge

- **It fails** 26.1-wf, 26.2, 26.6 to 26.9, 26.12, 26.14 and 26.17.
- **It passes** calibration (26.13) and the negative controls (26.16).
- **It is INCONCLUSIVE rather than REJECTED.** Rejection needs a leakage, calibration or
  negative-control failure, an evaluated holdout with BSS ≤ 0, or a failed cost test on at least
  200 trades. The null made 7.
- **One thing is not yet explained.** Those 7 trades claimed a mean net edge of +0.26, measured at
  the model's own conservative probability. Against `p_true` they averaged −0.04.
- **Without the uncertainty haircut,** the model would have made 347 trades at −0.031 against
  `p_true` (95% CI −0.035 to −0.027).

### Predictions written before the runs

The predictions and their readings were fixed in advance
([ADR 0004](docs/decisions/0004-n3b-predictions.md)):

| Prediction | Reading | Measured | Result |
|---|---|---|---|
| P1: planted 26.1-wf passes the bar | BSS ≥ 0.010 | 0.012516 | held |
| P2: planted interval coverage inside the mispriced subset rises toward the rest | inside > 0.5544 and outside − inside < 0.3631 | inside 0.8768; outside − inside −0.0022 | held |
| P3: planted T-30 trades stay near zero | ≤ 23 trades | 179 | **not held** |
| P4: the null's walk-forward BSS is not in the tail of outcome redraws | redraw p ≥ 0.05 | 0.740 (740 of 1,000) | held |

P3 failed and stays failed. The readings are not revisited after the numbers are known.

## What the harness caught

An earlier version of the generator drew every random number from one shared stream.

**The correlation checks looked clean.** A report-only leakage audit correlated every model feature,
and every generator quantity that feeds one, with the residual (outcome − `p_true`). It covered
13,952 evaluation rows per world. No test survived Benjamini-Hochberg correction; the smallest
q-value was 0.10.

**The order-of-operations writeup found the problem.** The audit also required a written account of
which quantities the generator computes before each outcome is drawn, and which after.
- Each game's final-score loop drew a number of random values that depended on who won.
- The market's quotes, spreads and sizes were all drawn after that.
- So the stream position of 9 of the model's 11 features depended on earlier outcomes.

No outcome value reached a feature, and no direct leakage was claimed. But the design allowed it, so
results from that generator were relabelled as unverified.

**The fix.** Every game gets its own random stream per component: pregame state, outcome, market and
sportsbooks. Each stream is seeded from a hash of the master seed, world, game and component.

**The invariant test.** `tests/leakage/test_synthetic_outcome_independence.py` is permanent. It flips
or redraws the outcomes of half of one day's games, and requires every pregame row, quote and price
to stay bit-identical.
- **On the old generator it failed.** All 5 of its cases failed, and the market moved in all 44
  games before the test date.
- **It passes** on the fixed one (`synthetic-mlb-0.3.0`).
- **Mutants.** Deliberately broken generators showed that the first version of the test missed
  3 of 13: each made two components or two games share one random sequence. A per-game stream test
  and a stronger stream-key check were added, and all 13 are now caught.

Details are in [ADR 0003](docs/decisions/0003-generator-random-streams.md).

## Design notes

- **Point-in-time invariant.** Every feature value carries the time it became available.
  - The database refuses a row with `available_at` after its cutoff, or a cutoff after the snapshot.
    The cutoff is the quote time, not the game start (26.11).
  - The tests in `tests/leakage/` inject a future-timestamped value and check that it is rejected.
    They then check that the same row is accepted without the leak. So a missing guard fails, and so
    does a guard that rejects everything.
  - The feature tests also break the time filter on purpose, then check that the features change and
    that the output guard raises.
- **Executable prices, not midpoints.**
  - The market's forecast is the midpoint of the bid/ask quote. The database refuses an observation
    whose market probability lies outside the quote, such as a last trade.
  - A trade pays the ask on the side it buys: YES at the YES ask, NO at one minus the YES bid.
  - On top of that come a spread penalty and fees. The penalty scenarios run from 0 to a full
    spread, with a quarter spread as the baseline.
  - A trade is capped at a tenth of the displayed size.
- **Append-only evidence with commit provenance.**
  - Backtest runs, training runs, experiments, walk-forward periods and review flags go into
    append-only logs. A repeated id is refused, not overwritten.
  - Every run records the git commit it ran from, marked `-dirty` when tracked files differ from it.
  - A lineage table ties each backtest to the training invocation and experiments that fed it.
  - See [ADR 0001](docs/decisions/0001-dirty-trees-and-append-only-backtest-runs.md) and
    [ADR 0002](docs/decisions/0002-no-terminal-status-without-a-clean-commit-and-append-only-training.md).
- **Dirty-tree verdict cap.** A run without a clean commit, or fed by rows without one, gets no
  terminal status. It is capped at INCONCLUSIVE, and any reason that would have rejected it is kept
  and marked held back (`decide_status` in `src/reports/diagnostics.py`).
- **Predictions committed before runs.** The N3b predictions and their readings were committed
  before either world was run ([ADR 0004](docs/decisions/0004-n3b-predictions.md)). This repository
  is published as a single commit, so it cannot show that order itself. The record rests on the
  history it was copied from.
- **Fixed acceptance states.**
  - The referee assigns exactly one state by fixed rules (PROTOCOL 26.19): REJECTED, INCONCLUSIVE,
    RESEARCH-PROMISING or PROSPECTIVELY-VALIDATED.
  - Thresholds are configuration values in `config/mlb.yaml`, not tuned on results. They include the
    minimum edge (0.04) and the maximum spread (0.10).
  - The walk-forward periods and the final holdout (2026H2) are fixed in configuration before any
    run.

## Limitations

- **Synthetic only.**
  - Every number here comes from generated worlds with known answers.
  - No result on any real market is claimed.
  - Nothing here is trading advice, and nothing places orders.
- **No run here can reach RESEARCH-PROMISING.** The final holdout is never evaluated, and 26.18 needs
  live paper trading, which these worlds do not exercise.
- **The fee schedules are illustrative.** They are marked unverified against any venue's published
  schedule, and the review says so.
- **One seed.** Each world is one draw from its generator. None of the results have been checked
  across seeds, the null's overconfident trades included.
- **Only the protocol is published.** Code and records cite other sections of the specification
  ("Sec 14", "Sec 17"), and those sections are not in this repository.
- **Packaging.** The package is named `src`, and it must be installed editable.
- **Memory.** Each world run peaks at about 5 GB.

## Repository layout

```
config/          run configuration and the registered hypotheses
docs/PROTOCOL.md the review criteria (Section 26 of the specification)
docs/decisions/  selected decision records
src/sports/      the synthetic generator
src/features/    point-in-time features
src/models/      the models, uncertainty intervals and walk-forward training
src/backtest/    the backtest engine, execution, registries, holdout and walk-forward periods
src/portfolio/   sizing, exposure and paper trading
src/reports/     the referee (the Section 26 review)
src/storage/     the database, schemas and append-only logs
tests/unit/      unit tests
tests/leakage/   leakage tests
```

## License

MIT. See [LICENSE](LICENSE).
