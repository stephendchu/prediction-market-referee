# ADR 0001: The exploratory family, the slope diagnostic, dirty-tree gaps, dirty rows, append-only backtest runs

- **Status:** Accepted, 2026-09-25. It was written before Q1–Q4 and N1 were built. Each item's
  state is kept under "State" below.
- **Decided by:** the owner, in chat, 2026-09-25, in a reply to the builder's second report. The
  decisions are quoted from it.
- **Answers:**
  - the builder's questions Q1–Q4;
  - the referee-v3 backlog's items 5 (dirty rows) and 12 (the exploratory family). The backlog
    rows this record uses are at the end.
- **Background:** D-A and D-B are two earlier decisions of the owner, from a record that is not
  published. D-A: "Runs from uncommitted code: record them, tag "-dirty". Dirty runs cannot be
  registered or frozen. A freeze requires a clean tree." D-B: "Missing git_commit blocks
  PROMISING. A row without provenance is not evidence."
- **Conflicts with:** nothing in the project instructions.

Published excerpt. Copied from a private decision log and scrubbed: references to private
infrastructure, data sources, unpublished records and private commit hashes are removed or replaced,
and elisions are marked […]. Item labels are kept from the original. Records, diagnostics and run
ids mentioned here that this repository does not ship belong to the private project; they are cited
as the record of what was decided and run, and cannot be checked from this repository alone.

## The owner's instructions (verbatim, scrubbed)

> GO-AHEAD
> […]
> After commit 4, rerun `quant backtest` and `quant review` once so backtest_runs carries a clean
> commit again. Paste only the status line and the referee config hash.
>
> Q1  exploratory family: not in the BH set. Own section, labeled EXPLORATORY, and never counted
> toward any acceptance criterion. 26.5 must not FAIL because exploratory rows exist. Backlog
> item 12 is closed by this.
>
> Q2  slope rule: accepted. 26.13 fails only when the whole CI is outside [0.5, 1.5]. Add one
> diagnostic line (no state change) when the CI excludes 1.0, so […] miscalibration is
> visible before it is severe.
>
> Q3  D-A gaps: close both.
>     - `quant train` on a dirty tree may train (tagged -dirty) but must not register a
>       hypothesis. Refuse the registration step with a message, not silently skip it.
>     - `holdout-register` and `holdout-lock` refuse on a dirty tree. The lock is the freeze; a
>       dirty freeze is not a freeze.
>
> Q4  -dirty rows (backlog item 5): a -dirty row is listed in the registry (successes and
> failures stay visible) but has no provenance for acceptance. Rules:
>     - 26.5 FAILs if any row in the BH set is -dirty.
>     - The run under review must carry a clean commit for RESEARCH-PROMISING; a dirty run caps
>       status at INCONCLUSIVE.
>     - Dirty rows are never dropped from the listing to make 26.5 pass. Rerun them clean
>       instead.
>
> N1  Run ids must not be reused across commits. `bt-54815509fa927a44` was overwritten and the
> clean-commit row from item 2 is gone. That is evidence destroyed by a re-run. Either fold
> commit+dirty flag into the run id, or make backtest_runs append-only like experiments, with the
> referee reading the latest run for a config. Record the choice in an ADR. Tests and mutants as
> usual.
>
> […]

## N1: the choice

**Append-only, with one run id per invocation.** The alternative, folding `<sha>[-dirty]` into
the id, was rejected for two reasons:

- Two runs from different uncommitted code at the same HEAD both carry `<sha>-dirty`, so they
  would still share an id and the second would overwrite the first.
- Q4 keeps dirty rows visible. An id that repeats per commit makes every dirty rerun overwrite
  the last one.

**What is built:**

1. **The id.** `backtest_run_id` hashes what it hashed before (the model runs, offset, threshold
   config hash, model type and quote quality) plus the run's `git_commit` and its `created_at`.
   Every invocation gets new ids, one per snapshot offset.
2. **Append-only tables.** `backtest_runs` and `backtest_decisions` are append-only.
   - The engine appends each invocation's rows in one transaction and deletes nothing.
   - A repeated id raises `ConstraintViolation`; it never overwrites.
   - `Database.replace` and `Database.replace_tables` refuse both tables, citing this record.
   - `fee_schedules` is still replaced: it is a static reference table.
   - No DDL changes, so `SCHEMA_VERSION` stays `db-0.5.0`.
3. **Which run the referee reads.** It reads the latest invocation's canonical-offset run, as
   before (latest `created_at`).
   - The sensitivity offsets come only from that same invocation: the same `created_at`,
     `git_commit`, `config_hash`, `model_type` and `quote_quality`.
   - An older invocation is never mixed into a review.
   - A named run is still selected by `load_review_inputs(backtest_run_id=...)`, and its
     sensitivity runs come from its own invocation. (Corrected: this line first said "`--run`
     still selects a named run". `quant review` has no `--run` option; the selector is only the
     library argument.)
4. **Growth.** One invocation on the synthetic sample appends about 280,000 decision rows, about
   5 MB as zstd Parquet (measured 2026-09-25). The table is a derived cache: dropping the
   database drops the history. The durable record of a result is the review artifact.

**Not changed (flagged):** `quant train` still replaces `model_runs`, `predictions` and
`walk_forward_periods` on every run (`src/models/train.py`, `replace_tables`). `model_run_id`
hashes the data fingerprint, not the commit, so a retrain on new code overwrites the old
predictions. A backtest row then points at predictions from a later training. The model runs
carry their own `git_commit`, so the mismatch is detectable, but the old predictions are gone.
The request named only `backtest_runs`, so this is left for the owner (see the backlog).

## Interpretations (the builder's, reversible)

1. **Q3, train exit code.** When registration is refused, `quant train` still trains and exits 0.
   - It prints the refusal, names the hypotheses it did not register, and adds the refusal to
     the run's notes.
   - It still validates the hypotheses file read-only, so a changed definition still stops the
     run.
   - The alternative, a non-zero exit, would break "may train".
2. **Q3, holdout commands.** `holdout-register` and `holdout-lock` check the tree before writing
   anything. So do the library calls behind them (`holdout.register`, `holdout.lock_candle`), not
   only the CLI.
3. **Q4, dirty and missing commits.** A missing commit (`unavailable: not a git repository`, or
   empty) on the run under review also caps the status at INCONCLUSIVE. It has no provenance
   either.
   - This builds backlog item 4 (D-B) for the run under review.
   - D-B for experiment rows is still not built: 26.5 FAILs on them, and 26.5 does not move the
     status.
4. **Q4, holdout rows.** "Any row in the BH set" includes experiments tested on the holdout.
   Dirty rows are counted over every FDR-family row, as the missing-commit count already is.
5. **Q1, 26.4.** 26.4 still reads exploratory experiments when it checks that hypotheses were
   registered before experiments ran. That can only make 26.4 stricter, and 26.4 is not an
   acceptance count.
6. **Q1, family name.** The exploratory family is the experiment family named `exploratory`.
   Today only one hypothesis writes it, and that hypothesis is disabled.
7. **Q1, what else fails.** 26.5 still FAILs if `review.fdr_families` names `exploratory`, as it
   does for `negative_control`.
8. **Q2, the diagnostic.** The line goes into 26.13's detail and adds `slope_ci_excludes_1` to its
   value. It says which way the model is off: a CI below 1 means overconfident, above 1 means
   underconfident. It appears in every 26.13 state that has a finite CI.
9. **N1, "the latest run for a config".** This is read as the latest invocation, whatever its
   config.
   - The alternative was the latest run matching the current config hash. That would silently
     review an older run after a config change.
   - A review of the latest run under a changed config shows the run's own `config_hash` next to
     the referee's.

[…]

12. **Q4, "the BH set".** A `-dirty` negative-control row does not FAIL 26.5: controls are
    outside the BH set (backlog item 9). 26.16 still reads them, so dirty controls could
    drive a REJECTED. Whether a dirty control may do that is not decided; it is asked, not
    invented (backlog item 14).
13. **Q4, the two caps.** A run with no clean commit returns INCONCLUSIVE before any other rule,
    as the non-executable quote cap already did. When both apply, both reasons are listed.
14. **Q1, a test renamed.** `test_an_experiment_family_outside_the_fdr_set_fails_26_5_closed`
    used `exploratory` as its example of a family outside the FDR set. Under Q1 that family has
    its own role, so the example is now `adhoc`. What the test checks is unchanged: an unknown
    family still FAILs 26.5.
15. **N1, tests that assumed one invocation.** Two existing tests relied on each backtest
    replacing the last:
    - The loader test in `tests/unit/test_review.py` now writes its T-30 and T-60 runs with one
      `created_at`, as the engine does.
    - `tests/unit/test_quote_quality.py` now reads each run's decisions by its own run ids.

## State

**Go-ahead.**

- **The rerun after commit 4** was on commit 4 (a private commit), a clean tree, at
  2026-09-26T01:13:15Z.
  - It printed `STATUS: INCONCLUSIVE`, with referee-v2 config hash `aefb3cb79dd319ef`.
  - All five backtest rows carry commit 4's full hash as `git_commit`.
- **The rerun overwrote the rows again.** The same five run ids had held the 22:54Z `-dirty`
  rows, which is N1 in action.
  - Those rows were saved first […].
  - The item 2 rows (21:24Z, from a clean commit) had already been overwritten at 22:54Z. They
    survive only in the text of an earlier record, which is not published.

**Q3, built.** Tests in `tests/unit/test_dirty_tree.py`.

- Red (tests written, code unchanged): 8 failed, 3 passed. Green: 11 passed. Control for the
  mutants (the Q3 group, unmutated): 98 passed.
- Mutants, all RED:

  | Mutant | Change | Killed by |
  |---|---|---|
  | TA | `holdout.register` without the tree check | `test_holdout_register_refuses_uncommitted_code_and_writes_nothing` |
  | TB | `holdout.register` checks after writing | the same test |
  | TC | `lock_candle` without the tree check (the CLI check stays) | `test_holdout_lock_refuses_uncommitted_code_and_writes_nothing` |
  | TD | `holdout-lock` CLI without its pre-check (the library check stays) | `test_holdout_lock_cli_refuses_before_fetching_the_schedule` |
  | TE | train registers from a dirty tree | `test_dirty_train_registers_no_hypothesis_and_no_status` |
  | TF | train skips registration silently | `test_dirty_train_says_it_refused_the_registration` |
  | TG | train notes the refusal but does not print it | the same test |
  | TH | train skips validation on the dirty path | the same test |
  | TI | the refusal omits the pending ids | the same test |
  | TJ | the pending check writes instead of reading | `test_dirty_train_registers_no_hypothesis_and_no_status` |
  | TK | train validates nothing on the dirty path | `test_dirty_train_still_refuses_an_edited_definition` |

  TB's first definition did not apply: its anchor matched nothing, and the runner reported
  `ERROR old text found 0 times`. It was fixed and rerun RED. TK was added after that run.

**Q4, Q1 and Q2, built.** Tests in `tests/unit/test_review.py`.

- Red: 31 failed, 73 passed. Green: 104 passed. Mutant control: 104 passed.
- Mutants TL–TZ and VA–VJ, 25 in all, are all RED. By area:

  | Area | Mutants |
  |---|---|
  | Q4, the run's cap | TL, TM, TN, TO, TP |
  | Q4, 26.5 on dirty rows | TQ, TR, TS, TT, TU |
  | Q1, the exploratory role and section | TV, TW, TX, TY, TZ, VA, VB, VC, VD |
  | Q2, the diagnostic | VE, VF, VG, VH, VI, VJ |

  Each mutant, its change and the test that killed it are in an evidence file that is not
  published.
- **TO needed a second run.** Its first definition was a syntax error, which gave "1 error"
  rather than a failing test, so that first run does not count. Redefined (`caps.clear();
  caps.append(`), it is RED: `test_decide_status_caps_a_run_without_a_clean_commit_at_inconclusive`.

**N1, built.** Tests in `tests/unit/test_backtest_append_only.py` (new, 25 tests).

- Red: 19 failed, 15 passed. Green: 34 passed, with the two adapted files (interpretation 15).
  Mutant control: 197 passed.
- **One test was corrected after the red run.** The engine test first asserted that every
  run id has decisions, but only T-30 and T-60 are seeded. It now asserts that the seeded offsets
  have decisions. Every red failure had already failed on the id-overlap assertion, before the
  corrected line, so the red result stands.
- Mutants VK–VZ and WA, all RED:

  | Mutant | Change |
  |---|---|
  | VK | id without the commit |
  | VL | id without the time |
  | VM | no append-only guard |
  | VN | `replace_tables` unguarded |
  | VO | `replace` unguarded |
  | VP | `append_tables` empties each table first |
  | VQ | per-table commits |
  | VR | `fee_schedules` appended |
  | VS | the pre-N1 loader |
  | VT–VX | the invocation ignores `created_at`, `git_commit`, `config_hash`, `model_type`, `quote_quality` |
  | VY | (see below) |
  | VZ | a named run takes sensitivity from the latest invocation |
  | WA | a missing commit matches any commit |

- **VY first survived.** The loader still had the old one-quote-class filter, which made the new
  invocation's `quote_quality` check redundant. The redundant filter was removed. VY is now RED
  (`test_sensitivity_never_mixes_in_another_quote_class`), and the control passed again, 197.
- `test_a_run_without_a_commit_does_not_match_one_with_a_commit` was added after the red run, to
  kill WA.

**Full suite** before the last loader change: 1274 passed. It is rerun at the end.

## Referee-v3 backlog, as of this record

Only the rows this record and [ADR 0002](0002-no-terminal-status-without-a-clean-commit-and-append-only-training.md)
use are kept. […]

| # | Item | State |
|---|---|---|
| 3 | Stricter 26.5: an experiment row without a commit FAILs | built (private commit), running as referee-v2 |
| 4 | A missing commit blocks PROMISING | for the run under review: this record, Q4 and interpretation 3. For experiment rows: not built (26.5 FAILs, but 26.5 does not move the status) |
| 5 | How 26.5 and PROMISING treat `-dirty` rows | answered by Q4; built (State) |
| 8 | The calibration slope with a bootstrap 95% CI; FAIL only when the whole CI is outside the band | committed (private commit); the Q2 CI-excludes-1 diagnostic built (State) |
| 9 | BH set = ablation, hypothesis, model; negative controls in their own section | committed (private commit) |
| 10 | 26.4 and 26.5 labels: feature families vs FDR families | committed (private commit) |
| 12 | The `exploratory` experiment family | closed by Q1; built (State) |
| 13 | `quant train` overwrites `model_runs` and `predictions` on every run | open: for the owner |
| 14 | May a `-dirty` negative control drive a 26.16 REJECTED? | open: for the owner |
