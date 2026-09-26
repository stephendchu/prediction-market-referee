# ADR 0002: A run without a clean commit gets no terminal status; a negative-control failure on one blocks freeze and registration; append-only training

- **Status:** Accepted 2026-09-26 (the owner's block "N2 close-out", items 3 onward).
  Each item below says whether it is built. Items 3 and 4 are built. Item 5 is done: the clean
  planted_v2 run `bt-b6cdaffb46a240cf` is locked as the N2 record. Item 6 was report-only, and no
  code changed. Item 7 is text only: three rows added to the referee-v3 backlog, not published
  here. […]
- **Pre-fix generator ([ADR 0003](0003-generator-random-streams.md)).** Item 5's clean reruns of
  both worlds, the N2 record (planted_v2 `bt-b6cdaffb46a240cf`) among them, came from the
  one-stream generator, `synthetic-mlb-0.1.0` / `synthetic-mlb-0.2.0-planted`, before the per-game
  streams of `synthetic-mlb-0.3.0`. They stay locked and unedited, and are unverified for spec
  decisions (the owner's decision of 2026-09-26, [ADR 0003](0003-generator-random-streams.md)).
- **Decided by:** the owner, in chat. The design choices marked "interpretation" are the
  builder's own, and each one can be reversed.
- **Supersedes:** [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) Q4's sentence
  "Neither cap hides a rejection", for the commit cap only (`src/reports/diagnostics.py`,
  `decide_status` docstring). The quote-quality cap still never hides a rejection.
- **Conflicts with:** nothing in the project instructions.

Published excerpt. Copied from a private decision log and scrubbed: references to private
infrastructure, data sources, unpublished records and private commit hashes are removed or replaced,
and elisions are marked […]. Item labels are kept from the original. Records, diagnostics and run
ids mentioned here that this repository does not ship belong to the private project; they are cited
as the record of what was decided and run, and cannot be checked from this repository alone.

## The owner's instructions (verbatim, scrubbed)

> 3. ITEM 14 (ADR): A -dirty run cannot produce any terminal status. Dirty → status capped at
>    INCONCLUSIVE, evidence retained. A negative-control failure on a dirty run raises a blocking
>    flag (no freeze/register until a clean rerun clears it). Build + tests; mutant-test per
>    standing rule. Commit.
>
> 4. ITEM 13: Make `quant train` append-only for model_runs and predictions, same pattern as N1
>    ([ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md)). Red/green + mutants. Commit.
>
> […]

## Item 3 (ITEM 14): no terminal status without a clean commit; the negative-control flag

**Built** in the commit that adds this record.

### The status cap (`decide_status`, 26.19)

- **What counts.** A run "without a clean commit" means one whose `git_commit` is `-dirty`,
  missing, empty or `unavailable…`. This is the same test (`is_clean_commit`) that
  [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) Q4's cap already used. The
  owner's text says "-dirty". A missing commit is included, because unknown is not clean (D-A,
  quoted in [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md)). *Interpretation.*
- **Terminal statuses** are REJECTED, RESEARCH-PROMISING and PROSPECTIVELY-VALIDATED: every
  status except INCONCLUSIVE. Before this record, a dirty run could still be REJECTED
  ([ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) Q4: "Neither cap hides a
  rejection"). Now it cannot. A run without a clean commit is always INCONCLUSIVE.
- **Evidence retained.** Nothing is dropped:
  - every criterion is still computed and listed;
  - the backtest rows stay (append-only,
    [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) N1);
  - each rule that would have rejected the run on a clean commit stays in the status reasons,
    as `held back by the commit cap (REJECTED on a clean commit): <reason>`. The reasons are
    the same text a clean run's REJECTED status gives.
- **The quote-quality cap is unchanged.** A clean run on degraded (non-executable) quotes can
  still be REJECTED. When both caps apply, the commit cap governs, and the run is INCONCLUSIVE.
- **Which backlog row.** The owner's "ITEM 14" is row 14 of the referee-v3 backlog in
  [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) ("May a `-dirty` negative
  control drive a 26.16 REJECTED?"). This record answers it for the run under review: no. (A
  draft of this record said the change "joins the backlog as row 13". That was wrong: row 13 is
  the owner's ITEM 13, `quant train` overwriting its tables, item 4 below. Corrected before
  commit.)
- **Not covered: dirty control rows under a clean run.** 26.16 reads every negative-control row
  in the persistent `experiments` log, and each row carries the commit of the `quant train` that
  wrote it. If `quant train` ran from a dirty tree and `quant backtest` then ran from a clean
  one, the run under review is clean, and a 26.16 FAIL driven by `-dirty` control rows still
  gives REJECTED. The owner's text speaks of "a -dirty run", so this record does not extend the
  cap to the rows. Row 14 stays open for that case, and it is listed under NOTED, NOT DONE.
- The referee string stays `referee-v2`. Under an earlier decision of the owner (not published)
  there is no v2.1. The change is built and runs as referee-v2, the way rows 8–10 were handled.

### The blocking flag

- **What raises it.** A `quant review` whose run has no clean commit and whose **26.16**
  ("Negative controls do not 'work'") is **FAIL**. 26.16 is the only criterion the referee
  calls a negative control. 26.17, the label-shuffle test, is not one of them.
  *Interpretation:* "negative-control failure" = 26.16 FAIL.
- **Where it lives.** A new persistent, append-only log, `review_flags`, is written the same way
  as `hypothesis_status` and `experiments`: Parquet under `data/logs/<sport>_<dataset>/`,
  exposed as a view. It holds two kinds of row:
  - a `raise` row per flagged run;
  - a `clear` row per flag that is cleared.
  Nothing is ever deleted. A re-ingest rebuilds only the derived tables, so it does not remove
  a flag. The schema has two CHECKs:
  - a `raise` row must name a commit that is not clean;
  - a `clear` row must name a clean one.
- **What it blocks.** While a flag is open for a dataset, two things refuse with
  `ReviewBlocked`, before anything is written:
  - `quant freeze`;
  - hypothesis registration, `python -m src.backtest.registry register` (`register_now`).
- **What it does not block** (*interpretation*):
  - […], `quant holdout-register` and `quant holdout-lock`. These are sport-level config
    registrations, and none of them belongs to one dataset's database;
  - `quant train`'s own hypothesis registration, which writes its rows with the run.
  Listed under NOTED, NOT DONE in the builder's report.
- **What clears it.** A later `quant review` in the same dataset clears a flag when all of these
  hold:
  - its run has a clean commit;
  - its 26.16 is **PASS**;
  - its run was created after the flagged run.
  26.16 INSUFFICIENT (no negative controls) does not clear, since it cannot show that the
  controls fail. A clean run whose 26.16 is FAIL is REJECTED, and the flag stays open.
  *Interpretation:* the rerun need not have the same config or data version. A fix to either is
  what a clean rerun is for. Both are recorded on the rows.
- **Scope.** A flag is per dataset database (`<sport>_<dataset>`). A separate data root has its
  own log.
- **What is reported.** The review's status reasons name the flag when it is raised. The review
  JSON, markdown and terminal output list what this review raised or cleared, and every flag
  still open.

### Tests and mutants

See "Evidence" below. Per the standing rule, the status rule and the flag logic decide evidence,
so they got mutants. The CLI and markdown text got red/green tests only.

## Evidence

Evidence files are outside the repository and are not published.

- **Red first.**
  - The first red run stopped at collection, because the module and names did not exist yet.
  - The builder then added the names only, with `negative_control_flag` returning `None` and
    nothing wired in. That gave 42 failed and 15 passed.
  - The printed-flags test was written after the print code. To show it red, the builder took
    that hunk out: 1 failed. With the hunk restored it is green.
- **Green, targeted.** 252 passed.
- **Mutants, XB–YF.** 31 mutants, all RED. The unmutated control gave 253 passed. Each run used
  `-x -m "not e2e"` over `test_review_flags`, `test_review`, `test_dirty_tree`, `test_registry`
  and `test_storage`, in four copies of the code.

  | Id | Mutant | Killed by |
  |---|---|---|
  | XB | a dirty run is REJECTED again | `test_decide_status_caps_a_run_without_a_clean_commit_at_inconclusive` |
  | XC | held-back reasons dropped | `test_a_run_without_a_clean_commit_is_never_rejected_and_keeps_the_reasons` |
  | XD | only the first held-back reason kept | same, `[all-dirty]` |
  | XE | no blocking-flag reason | `test_a_negative_control_failure_without_a_clean_commit_names_the_blocking_flag` |
  | XF | flag reason on any dirty run | `test_decide_status_caps_a_run_without_a_clean_commit_at_inconclusive` |
  | XG | raise on any 26.16 FAIL, clean or not | `test_no_flag_without_a_dirty_negative_control_failure` |
  | XH | raise on 26.16 INSUFFICIENT too | same |
  | XI | clear without a clean commit | same |
  | XJ | 26.16 INSUFFICIENT clears | same |
  | XK | the quote cap hides a rejection again | `test_decide_status_caps_non_executable_at_inconclusive` |
  | XL | held-back reasons lose their prefix | `test_a_run_without_a_clean_commit_is_never_rejected_and_keeps_the_reasons` |
  | XM | a clear row never closes a flag | `test_a_later_clean_run_with_26_16_pass_clears_every_earlier_flag` |
  | XN | any clear row closes every flag | `test_a_clean_pass_clears_only_the_flags_raised_on_earlier_runs` |
  | XO | `open_flags` reads every dataset | `test_open_flags_reads_this_dataset_only` |
  | XP | a re-review raises a second row | `test_reviewing_the_same_flagged_run_again_does_not_raise_a_second_row` |
  | XQ | a clean pass clears later runs' flags | `test_the_flag_stays_open_unless_a_later_clean_run_passes_26_16[earlier]` |
  | XR | a clean pass clears a flag of the same time | same, `[same-time]` |
  | XS | a clear is written as a raise | `test_a_later_clean_run_with_26_16_pass_clears_every_earlier_flag` |
  | XT | `record_review` never writes | `test_a_dirty_negative_control_failure_raises_one_flag` |
  | XU | `require_unblocked` never refuses | `test_flags_persist_across_a_reconnect` |
  | XV | `record_review` reads the wrong commit | `test_a_dirty_negative_control_failure_raises_one_flag` |
  | XW | `register_now`: no flag check | `test_hypothesis_registration_refuses_while_a_flag_is_open` |
  | XX | `register_now`: flag check after writing | same |
  | XY | `quant freeze`: no flag check | `test_quant_freeze_refuses_while_a_flag_is_open` |
  | XZ | `quant review` records no flag | `test_quant_review_records_the_flag_and_shows_it` |
  | YA | schema: a raise may name a clean commit | `test_the_schema_refuses_a_row_that_breaks_the_flag_rules[raise-clean]` |
  | YB | schema: a clear may name a dirty commit | same, `[clear-dirty]` |
  | YC | schema: `unavailable` counts as clean | `test_a_dirty_negative_control_failure_raises_one_flag[unavailable]` |
  | YD | schema: `''` counts as clean | same, `[empty]` |
  | YE | schema: NULL counts as clean | `test_the_schema_refuses_a_row_that_breaks_the_flag_rules[clear-none]` |
  | YF | the flag reads 26.17, not 26.16 | `test_a_dirty_negative_control_failure_raises_one_flag` |

- **Full suite.**
  - The first run failed 1 and passed 1374. The one failure was
    `test_pipeline_end_to_end_on_synthetic_data`, and it was not caused by this item: see the
    correction below.
  - After the fix: 1375 passed, 0 failed.
- **Correction: the e2e test was red from item 1's commit on.** (Item 1's commit, not published
  here, added the planted-alpha world v2 and the append-only backtest runs.)
  - The planted-world design's world marker, `synthetic_world.json`, is written into the data
    root on first ingest.
  - The end-to-end test asserts exactly which names the data root holds, so it failed.
  - Item 1's commit message reported the non-e2e suite only ("1295 passed, 1 deselected"). The
    deselected test was this one, and it had not been run.
  - Checked on clean exports: it fails at the then-current head and at item 1's commit, and
    passes at an earlier commit (private commits).
  - The fix is to the test only. The marker is the planted-world design's intended output, so
    the test now expects it and checks that it names `null_v1`. It is committed on its own,
    before this record.

## Item 4 (ITEM 13): append-only `quant train`

**Built** in the commit that adds this section. Row 13 of the
[ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md) backlog ("`quant train` replaces
`model_runs` and `predictions`") is closed by it.

### What changed

- **The two tables are append-only.** `model_runs` and `predictions` carry `append_only=True`
  in `src/storage/schemas.py`, the flag [ADR 0001](0001-dirty-trees-and-append-only-backtest-runs.md)
  N1 gave the backtest tables. `replace_tables` refuses them. `SCHEMA_VERSION` is unchanged: the
  flag is a write rule, not a column.
- **Each invocation gets its own ids.** `model_run_id` (`src/models/train.py`) now hashes the
  git commit and the invocation time along with the model type, period, feature hash, config
  hash and data fingerprint. A rerun from a new commit, or the same commit later, writes new
  rows. It never overwrites the earlier run's rows.
- **The model file is never rewritten.** If the model file for a run id already exists,
  `quant train` raises `ConstraintViolation` before saving. This is the same invocation run
  twice, which can only happen with a pinned clock. *Interpretation:* refuse, rather than
  overwrite or suffix.
- **`walk_forward_periods` is still replaced.** It is config, not a result. It is written after
  the append. See NOTED, NOT DONE below.
- **Readers take the latest invocation.** `walk_forward.latest_model_runs` returns the rows
  that share the newest row's `created_at` and `git_commit`.
  - A tie on time goes to the larger `model_run_id`, and only rows with its commit are taken.
    Two invocations are never mixed.
  - A missing commit matches a missing commit (`IS NOT DISTINCT FROM`).
  - *Interpretation:* "latest" means the last `quant train`, not a per-period or per-type
    newest row. If the latest train lacks the selected type, the readers find nothing. They do
    not fill it in from an older train.
- **Readers switched to it:**
  - the backtest engine's run lookup;
  - `explain` in `src/portfolio/paper.py`;
  - `current_manifest` in `src/backtest/holdout.py` (the freeze manifest's model pin).
- **Readers unchanged:**
  - `paper.slate_model` and the engine's row query read by run id, from ids that the latest
    train's selection names;
  - the diagnostics join by decision keys.
- **Test fix.** In `test_holdout_registry`, the stub database's `model_runs` answer gained
  `model_type` and `period_id`, since `current_manifest` now filters on them.

### Evidence

Evidence files are outside the repository and are not published.

- **Red first.**
  - The first red run stopped at collection, because `latest_model_runs` did not exist.
  - With the names only (`latest_model_runs` returning every row; the id without commit and
    time), 23 failed and 33 passed.
  - Three fixture tests passed vacuously before the change, because the old id made both trains'
    ids coincide. The seeded tests cover the same readers and were red.
- **Green, targeted.** `test_train_append_only` 32 passed. With `test_backtest_append_only`,
  57 passed.
- **Mutants, YG–ZA:** 21 mutants, all RED.
  - The first round had 19 RED and 2 survivors:
    - YY, the manifest dropping the model-type filter;
    - ZA, `explain` dropping the period filter.
  - Two tests were then added or tightened, and both mutants went RED against them. The
    unmutated control gave 102 passed in round 1 and 103 in round 2.

  | Id | Mutant | Killed by |
  |---|---|---|
  | YG | the id ignores the commit | `test_the_model_run_id_changes_with_every_part[git_commit-…]` |
  | YH | the id ignores the time | same, `[created_at-…]` |
  | YI | train passes no commit or time to the id | the `twice` fixture (train B collides with A) |
  | YJ | no model-file guard | `test_the_same_invocation_twice_raises_and_overwrites_nothing` |
  | YK | `model_runs` not append-only | `test_the_train_tables_are_append_only[model_runs]` |
  | YL | `predictions` not append-only | same, `[predictions]` |
  | YM | train replaces its tables again (flags off) | same, `[model_runs]` |
  | YN | `walk_forward_periods` appended, not replaced | the `twice` fixture |
  | YO | latest: commit not matched | `test_a_tie_on_time_takes_one_commit_not_both` |
  | YP | latest: a missing commit never matches | `test_the_backtest_reads_the_latest_train_invocation[then-no-git]` |
  | YQ | latest: the oldest invocation | `test_the_backtest_reads_only_the_latest_train` |
  | YR | latest: a tie goes to the smaller id | `test_a_tie_on_time_takes_one_commit_not_both` |
  | YS | latest: every row | the `twice` fixture |
  | YT | latest: time not matched | `test_the_backtest_reads_the_latest_train_invocation[same-commit]` |
  | YU | the engine reads every train | the `twice` fixture (duplicate decision key) |
  | YV | the engine drops the model-type filter | the `twice` fixture |
  | YW | `explain` reads every train | `test_explain_loads_the_latest_trains_model` |
  | YX | the manifest reads every train | `test_the_manifest_pins_the_latest_holdout_run` |
  | YY | the manifest drops the model-type filter | `test_the_manifest_pins_only_the_selected_types_model` (round 2) |
  | YZ | the manifest drops the period filter | `test_the_manifest_pins_the_latest_trains_holdout_model` |
  | ZA | `explain` drops the period filter | `test_explain_loads_the_latest_trains_model` (round 2) |

- **Full suite, e2e included:** 1407 passed, 1 warning, in 165 s.

### NOTED, NOT DONE (item 4)

- `walk_forward_periods` is still replaced. After a config change, an old `period_id` in
  `model_runs` could name a different window than the table now shows.
- `quant train` registers its experiments and hypotheses before it appends `model_runs`. The
  append can only fail after the model-file guard if the file was deleted by hand. Then the
  experiment rows would exist without their run rows.

## Item 5: the clean N2 reruns; planted_v2 is locked as the N2 record

**Done; summary only.** […] Both worlds were rerun from a clean clone at item 4's commit, and
the planted_v2 run `bt-b6cdaffb46a240cf` is locked as the N2 record: INCONCLUSIVE, 0 trades, and
a 26.1-wf BSS of 0.009138, below the D5 bar of BSS ≥ 0.010 (0.5 × the planted world's 0.020
design target, a fixed number). The clean runs reproduce the earlier dirty runs' numbers exactly, except the
registration-dependent 26.4 and 26.5. There are no reruns of N2 after this, except to
reproduce.

[…]
