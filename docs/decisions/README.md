# Decision records

Selected decision records (ADRs), copied from a private decision log, scrubbed and renumbered.
Each one says what was decided, by whom, and what was built or run. Records are not edited after
the fact: a later decision that changes an earlier one says so in its own record.

| ADR | Decision |
|---|---|
| [0001](0001-dirty-trees-and-append-only-backtest-runs.md) | Runs from a tree with uncommitted changes are marked `-dirty` and capped; backtest runs are append-only |
| [0002](0002-no-terminal-status-without-a-clean-commit-and-append-only-training.md) | A run without a clean commit gets no terminal status; a negative-control failure on such a run blocks freeze and registration; training runs are append-only |
| [0003](0003-generator-random-streams.md) | The synthetic generator draws each game's components from separate random streams (`synthetic-mlb-0.3.0`); the outcome-independence test; the N3b record |
| [0004](0004-n3b-predictions.md) | The N3b predictions, committed before the N3b runs |

## Recorded here, not in a numbered ADR

- **H_MLB_003 is registered and DISABLED** (`config/mlb_hypotheses.yaml`). Fee-band analysis is out
  of scope for this demonstration. The hypothesis stays registered and visible, so it counts
  toward the number of hypotheses tested (spec 26.5), but it is not fitted or tested. This was
  decided when the repository was published; it can be reversed by a numbered record.
