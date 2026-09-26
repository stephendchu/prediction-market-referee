# ADR 0003: Generator random streams per game and component (synthetic-mlb-0.3.0); the outcome-independence test; N3b

- **Status:** Accepted 2026-09-26 (the owner's block "generator fix + N3 regenerated",
  items 1–5). The block was paused for unrelated work and then resumed.
  - Items 1 and 2 are built in the commit that adds this record.
  - Item 3 is [ADR 0004](0004-n3b-predictions.md), committed before item 4.
  - Item 4 (N3b) is recorded below, under "Item 4": P1, P2 and P4 held, and P3 did not.
- **Decided by:** the owner, in chat. Choices marked "interpretation" are the builder's own,
  and each one can be reversed.
- **Conflicts with:** nothing in the project instructions.

Published excerpt. Copied from a private decision log and scrubbed: references to private
infrastructure, data sources, unpublished records and private commit hashes are removed or replaced,
and elisions are marked […]. Item labels are kept from the original. Records, diagnostics and run
ids mentioned here that this repository does not ship belong to the private project; they are cited
as the record of what was decided and run, and cannot be checked from this repository alone.

## The owner's instructions (verbatim, scrubbed)

> […] BLOCK — generator fix + N3 regenerated (do only these
> items; extras → NOTED, NOT DONE; rule D: one full suite at a time)
>
> Decision on the stopped item 1: (b). N3's numbers are treated as
> unverified for spec decisions. No direct outcome leakage is
> claimed. Item 5 and row 18 are deferred to the next block.
>
> 1. GENERATOR RNG (ADR): separate deterministic RNG streams per game
>    and component, seeded from master_seed + world + game_id +
>    component. Components at minimum: pregame game state, outcome
>    and box score, […] market, sportsbooks. null_v1 and
>    planted_v2 share all underlying game state and differ only in
>    the planted mechanism. Bump the generator version. N2/N3 records
>    stay locked, labelled as produced by the pre-fix generator.
>
> 2. INVARIANT TEST: holding each game's pre-outcome state fixed,
>    flipping or redrawing its outcome must leave every pregame X
>    input and every market/book value bit-identical. Permanent
>    test; mutants (leakage-check category). Commit.
>
> 3. PREDICTIONS (ADR, commit BEFORE item 4): [N3's] three
>    predictions carry over unchanged to the regenerated worlds.
>    Add P4: if the null BSS tail (redraw p = 0.009) was a chance
>    lean in the old outcome draw, the regenerated null's fixed-
>    prediction redraw p-value is not below 0.05.
>
> 4. REGENERATE AND RERUN: both worlds from the item 3 commit, clean
>    clone, same steps as N3 item 3. Record as N3b. Report: criteria
>    table, D5, trades, 26.12 coverage by subset, the planted and
>    null 2x2 (trades, engine edge, edge vs p_true with CI), the null
>    redraw p-value, and whether P1–P4 held.
>
> 5. REPORT, then stop.

## The decision on the stopped item 1

"The stopped item 1" is item 1 of the block before this one, the owner's block "leakage check +
item 5 + row 18". Its text (verbatim):

> 1. LEAKAGE CHECK (report only, do first): on null_v1 and
>    planted_v2, T-30 eval rows. For every feature in X and every
>    generator quantity that feeds X, report correlation with
>    (outcome - p_true), with game-level bootstrap CI and
>    BH-adjusted q-values. Also write out the generator's order of
>    operations: which quantities are computed before and after
>    outcomes are drawn. Name any feature computed after outcomes.
>    If one is found, stop and report before item 2.

The check was report-only, with no code change. Its scripts and output are scratch, not
published. It ran on the T-30 eval rows of both N3 runs, 13,952 rows over 7,432 games in each
world.
- **Correlations:** none was significant after BH. The smallest q was 0.10 in null_v1 (0 of 31
  tests below 0.05) and 0.27 in planted_v2 (0 of 34).
- **Order of operations:** the generator drew everything from one shared stream.
  - Each game's pregame state and `p_true` were drawn before its outcome.
  - `_play`'s final-score loop drew a number of times that depended on `home_win`.
  - `market_frames()` then drew the market wander, spreads, sizes and jitter, and the book masks
    and timing, after every game's outcome.
- **The stop trigger:** 9 of the 11 features in X came from those late draws. The link was the
  stream position, not an outcome value.

The report offered two options:
- (a) accept the stream-position link and run items 2 and 3 as written;
- (b) fix the generator first, so each game's outcome and box score draw from their own stream,
  then regenerate both worlds and re-run N3.

The owner chose (b). What was decided in chat:
- **N3's numbers are unverified for spec decisions.** This covers every number the pre-fix
  generator produced; see "The pre-fix records" below.
- **No direct outcome leakage is claimed.** The check found an ordering link and no significant
  correlation.
- **The leakage block's item 5 (the lineage table and the dirty-cap scope) and row 18 (the
  decision rule) are deferred to the next block.** Neither is built or written.

## Item 1: per-game, per-component random streams

### Design

Every random draw comes from a stream seeded by

```python
def _stream_seed(seed: int, world: str, key: str, component: str) -> int:
    h = hashlib.sha256(f"{seed}|{world}|{key}|{component}".encode()).digest()
    return int.from_bytes(h[:16], "big")
```

`_World._stream(component, key)` returns `np.random.default_rng(_stream_seed(master seed,
rng_world, key, component))`. Each component of each key has its own stream, used for one thing
only:

| Component | Key | Draws, in order |
|---|---|---|
| `league` | none | players, teams and venues |
| `season` | the season | skill drift into the season |
| `schedule` | the date | the day's pairings |
| `pregame` | the game id | start time; starters, scratches; lineups, revisions; weather and forecasts; hidden truth |
| `outcome` | the game id | the winner, then the box score (score loop, duration, actual start, pitching, relief, batting) |
| market | the game id | liquidity noise `z`; market noise `eps_k`; in `market_frames`, the quote-wander innovations, then the home side's quote draws, then the away side's |
| `books` | the game id | the book noise `eps_b`; in `market_frames`, per book in order, the update mask, then its timing jitter |

The component key strings are hashed into the seeds, so the frozen generator keeps its original key strings; the market component's key is a legacy venue name kept only for that reason.

`_game` stores each game's market and `books` generator objects with its market state, and
`market_frames` continues them. So a game's quotes and odds depend only on its own two streams.

The outcome goes through two hooks, which the test uses:
- `_World._outcome_stream(gid)` returns the `outcome` stream;
- `_World._home_wins(r, gid, p_true)` makes the one draw that decides the game.

### Interpretations

- **I1. "world" in the seed is the stream key `WorldParams.rng_world`, not the world's name.**
  Both registered worlds use `"null_v1"`, so they share every draw. The planted mechanism
  (`market_slope`, `planted_shift_logit`, `planted_home_teams`) draws nothing. A new world that
  needs different game state gets its own key.
- **I2. There are three components beyond the four named.** `league`, `season` and `schedule`
  come from draws that belong to no single game.
- **I3. The draw order inside the market and `books` streams** is the order in the table. It
  is fixed, and the goldens pin it.
- **I4. "Every pregame X input" means every row the generator writes that is available before
  first pitch.** Features are built from those rows only. That covers:
  - schedule and contract rows, without result and settlement;
  - starters, lineups, news events and weather;
  - the truth row;
  - every quote and every sportsbook price.

  Item 2 does not check feature values.
- **I5. Bullpen fatigue is the one legitimate path from one game's outcome to another game's
  pregame values.** A game's fatigue uses its teams' relief work over the three days before, which
  is known before first pitch. The test therefore holds fixed only games whose pre-outcome state
  did not change.

### Evidence that the two worlds share game state

A check of the 0.3.0 worlds (seasons [2023] through 2023-06-30, 8 games a day, 744 games) found:
- every table but the two market tables is identical except one version column, which carries
  the generator version;
- `market_quotes` and `consensus_odds` have the same rows. Only the prices differ: `yes_bid`,
  `yes_ask`, `no_bid`, `no_ask`, `last_price` and `depth_json`, and the books' prices and de-vigged
  probabilities;
- every truth column the two worlds share is identical.

The script and its output are scratch, not published.

### The version bump

| World | Before (one stream) | Now |
|---|---|---|
| `null_v1` | `synthetic-mlb-0.1.0` | `synthetic-mlb-0.3.0` |
| `planted_v2` | `synthetic-mlb-0.2.0-planted` | `synthetic-mlb-0.3.0-planted` |

Every value drawn is new, so every synthetic number changes. No parameter changed.

The null world's pinned table fingerprints (`tests/unit/test_synthetic_planted.py`, SMALL data,
`NOW`) were retaken. The old ones, taken at an earlier private commit, are kept here:

| Table | 0.1.0 | 0.3.0 |
|---|---|---|
| consensus_odds | 4b85425b0caa05b3 | ed2ab6e443ac6a87 |
| contracts | 3ffa537d8b0b0a8f | 82072e66700d5260 |
| games | 9058d17fbc52ad8d | b804c0c13b8f402c |
| info_events | ae762868492495c2 | 739e27947264bbbd |
| market_quotes | 4c8fecde8875407c | f6940a68f9e3bc1d |
| mlb_batting_lines | b979d3aee24be825 | 67bd3ebda9433047 |
| mlb_lineups | 4542d3ab62faa870 | a5def83273d527a8 |
| mlb_pitching_lines | de805c286f32ffd7 | 1ad777fa87eb3377 |
| mlb_players | 489330d024e21bcc | 18ba6196bc720870 |
| mlb_probable_starters | 9ad72e25b929f940 | fdda7626ceb4fec7 |
| mlb_weather | 6b5e104e3fc58278 | cfb986b6c07fd6a6 |
| venues | cc02d38cbad7f16a | b11ecb77cb71d604 |
| truth | 353d9cb23f8b8cd0 | 82878d8e58d3a31b |

[Publication note. This repository's generator names the series `SYN-MLBGAME` and two columns
`market_event_id` and `market_noise_logit`, where the private one used a venue's name. That
changes the fingerprints of four tables, so the test pins contracts `d2e732e143f85c61`, games
`362ce898835f4a20`, market_quotes `f740b5da50f7d11b` and truth `bde4dc819db841db`. With the
original names restored, all 13 fingerprints match the 0.3.0 column above, and every value in
games and truth is identical (checked column by column when the repository was published).]

The version change failed 6 of the 47 tests in `test_synthetic_planted.py` and `test_paper.py`,
as expected. Only their pinned values changed:
- the goldens and the version strings;
- in `test_paper.py`, the SMALL paper day's snapshot rows went from 93 to 94, and its slate from
  20 contracts to 18.

All 47 pass after the update.

### The pre-fix records

The N2 and N3 records stay locked. Their numbers are not edited. Each now carries a Status line
saying the pre-fix generator produced it and that it is unverified for spec decisions:
- **[ADR 0002](0002-no-terminal-status-without-a-clean-commit-and-append-only-training.md),
  item 5 (N2):** the clean reruns of both worlds, among them the N2 record, planted_v2
  `bt-b6cdaffb46a240cf`.
- **The N3 record (not published), items 3 and 4 and its addendum (N3):**
  - planted_v2 `bt-2c63924b3ad2b3eb`;
  - null_v1 `bt-efeddf6d9b3a77b6`;
  - the planted 2x2;
  - the null-world BSS check.
- **The N3 close-out record (not published), items 2 and 3:** the null 2x2, and the redraw check
  (p = 0.009).

Two other records hold pre-fix numbers but are not N2 or N3 records, so they get no label
(interpretation):
- **The planted-world design record's oracle numbers** (PROPOSED, not published), measured on
  0.1.0 and 0.2.0-planted;
- **The N3 predictions record's reference figures** (not published), such as the oracle's 2,357
  T-30 trades behind P3's bound of 23. Item 3 carries those predictions over unchanged
  ([ADR 0004](0004-n3b-predictions.md)).

## Item 2: the invariant test

`tests/leakage/test_synthetic_outcome_independence.py` holds three tests.

**`test_outcomes_do_not_reach_pregame_or_market_values`** runs on each registered world, with
two interventions each:
- The data is SMALL: seasons [2023] through 2023-04-20, 4 games a day.
- It changes half of the 2023-04-10 games, the 1st and the 3rd.
- **flip:** `_home_wins` returns the other winner, and the box score follows from the same stream.
- **redraw:** `_outcome_stream` hands those games an unrelated stream.

It checks, per game, a hash of all the game's rows:
1. **Up to and including the intervention date:** every pregame row (I4) and every quote and
   price is bit-identical.
2. **The intervention took effect:** every changed game's result or box score moved (flip: its
   winner), and no other result up to that date moved.
3. **Later games whose pregame rows are unchanged:** their quotes and prices are bit-identical.
   The test fails unless at least 5 such games exist. Without them, check 3 would test nothing.

**`test_streams_are_keyed_by_seed_world_game_and_component`** checks the keys:
- every (seed, world, game, component) has its own seed and its own sequence;
- both registered worlds use one stream key, and give the same sequences;
- `_World._stream` hands out exactly the sequence `_stream_seed` names for the key asked.

**`test_each_game_asks_for_its_own_stream_of_each_component`** records every stream a SMALL
null_v1 load asks for:
- every game asks once for its `pregame`, market and `books` streams, under its own id;
- every played game asks once for its `outcome` stream, and today's games do not.

The last two tests close a gap: the interventions cannot see two components that share one
sequence, because that is a correlation, not a dependence.

### Red, then green

**Red.** The test first ran against the pre-fix generator, with only the two hooks added. All five
tests failed. The redraw case looked like this in both worlds:
- **Before 2023-04-10:** the market moved in all 44 games. The market was drawn after every
  outcome, and outcomes on and after the date changed the stream position.
- **On the date:** pregame moved in 3 of 4 games, and results moved in all 4, the two unchanged
  games included.
- **After the date:** pregame moved in all 40 games, so no later game was left for check 3.
- **Keys:** the keys test failed on the missing `STREAM_COMPONENTS`.

The flip case gave the same counts.

**Green.** On 0.3.0 all tests pass: 5 before the third test was added, 6 after.

### Mutants (leakage-check category)

- **Setup:**
  - exact-once edits to `src/sports/synthetic.py`, one at a time;
  - three code-only copies of the repository;
  - `-m "not e2e"`, with no full suite running.
- **What judges them:** the permanent test file alone. The goldens in `test_synthetic_planted.py`
  fail on any change to the data, leak or not, so they are left out (interpretation).
- **Round 1** ran the first two tests: 10 of 13 red.
  - ABS, ABT and ABU survived. Each makes two components or two games share one sequence, which
    the interventions cannot see.
  - The third test and the `_World._stream` check were then added (interpretation: closing a gap
    in the item's own test, not a new item).
- **Round 2:** 13 of 13 red. The unmutated control passes in both rounds (5 passed, then 6).

| Id | Mutant | Round 1 | Round 2 | Failing test (round 2) |
|---|---|---|---|---|
| ABM | the outcome is drawn from the game's market stream (the object the quotes continue) | RED | RED | all four interventions (market moved on the date); each-game streams |
| ABN | the outcome is drawn from the game's sportsbook stream | RED | RED | all four interventions (market moved on the date); each-game streams |
| ABO | market noise gets a term in the winner, after the outcome | RED | RED | both flips (null: market moved on the date; planted: pregame moved, through the truth row's market logit) |
| ABP | the truth row's `hidden_logit` gets a term in the winner | RED | RED | both flips (pregame moved on the date) |
| ABQ | the market stream key includes every earlier game's winner | RED | RED | both flips (pregame moved: the truth row carries the market's `thinness` and noise); each-game streams |
| ABR | the schedule stream key includes every earlier game's winner | RED | RED | all four interventions: the at-least-5 guard (0 or 2 later games kept their pregame state) |
| ABS | `_World._stream` ignores the key | SURVIVED | RED | streams keyed |
| ABT | the outcome stream is seeded as the game's market stream (a new object, same sequence) | SURVIVED | RED | each-game streams |
| ABU | the sportsbook stream is seeded as the game's market stream | SURVIVED | RED | each-game streams |
| ABV | planted_v2 gets its own stream key | RED | RED | streams keyed (`{'null_v1', 'planted_v2'} == {'null_v1'}`) |
| ABW | `_stream_seed` ignores the world | RED | RED | streams keyed (15 seeds for 16 keys) |
| ABX | `_stream_seed` ignores the component | RED | RED | streams keyed (4 seeds for 16 keys) |
| ABY | the market quote wander is keyed by earlier days' winners, so it acts only after the date | RED | RED | both flips, check 3 ("market values moved in 39 later games"); each-game streams |

The redraw interventions miss ABO, ABP, ABQ and ABY. Those mutants act on the winner, and the
redraw left both changed games' winners as they were. The flips catch all four.

### Full suite

One full `pytest`, run alone (rule D) on the tree committed with this record: 1,654 passed,
5 warnings, in 200 s, with a peak RSS of 887 MB.

## Item 4: N3b

Both worlds were run from a `git clone --no-hardlinks` of the repo, detached at the commit that
added [ADR 0004](0004-n3b-predictions.md) (private history, not published), with 0 tracked files
differing from HEAD. `src` was imported from the clone, and the runs record that commit as
`git_commit`. Each world had its own data root. The steps were the same as N3's: register, then
`quant ingest`, `features`, `train`, `backtest` and `review`. Every step exited 0.
- The script and logs are scratch, not published.
- The diagnostics are N3's scripts, changed only in their paths, their labels, and planted_v2
  keeping its own subset column. They and their outputs are scratch, not published.

| World | Generator (`synthetic_world.json`) | Backtest run | Referee | Model | Status |
|---|---|---|---|---|---|
| planted_v2 | `synthetic-mlb-0.3.0-planted` | `bt-bf08a3dbce5f3d07` | referee-v2, config hash `aefb3cb79dd319ef` | `logistic_residual` | INCONCLUSIVE |
| null_v1 | `synthetic-mlb-0.3.0` | `bt-b14c63c6ddec1c50` | same | `logistic_residual` | INCONCLUSIVE |

- **Common to both runs.**
  - T-30, with executable quotes.
  - Run config hash `02f74ee2a9c4dfd2`, the same as N2 and N3.
  - Feature-set hash `f38d7b7ec81ea7a6`: N3's 11 features.
  - The negative-control flag event is `clear`, and no flag is open.
  - Each world evaluates 13,905 observations in 7,429 games (N3: 13,952 in 7,432).
- **Model selection.**
  - planted_v2 picked `logistic_residual`: LightGBM won 1 of 6 periods, and its pooled Brier was
    0.24730 against logistic's 0.24672.
  - null_v1 also picked `logistic_residual`. LightGBM had the lower pooled Brier (0.24586 against
    0.24615), but it won only 2 of 6 periods, and 4 are needed. N3's null had picked
    `lgbm_residual`.

[Publication note. Rerunning both worlds with this repository's commands (README, Quick start)
reproduces every Sec 26 criterion state, both statuses, the backtest trades and P&L, and the
edge-bucket tables exactly. Three things differ. They come from the renames in the note under
"The version bump", and from one more: one feature's name carried a venue's name, and this
repository calls it `market_move_since_news`.
- The feature-set hash is `db8582621e758cec`, and the data versions differ.
- The LightGBM comparison model. It samples 80% of the columns by index with a fixed seed, and the
  rename moves one column. null_v1: LightGBM wins 3 of 6 periods, pooled Brier 0.24554 against
  logistic's 0.24615. planted_v2: 1 of 6, 0.24731 against 0.24672. `logistic_residual` is
  selected in both worlds, as above, and its numbers are unchanged. The LightGBM experiments'
  BH q-values move with it.
- The random negative control hashes the observation id, which contains the series name, so its
  draws differ. 26.16 still PASSes in both worlds.]

**N2 and N3 stay locked.** They are pre-fix-generator records. N3b is a separate record and
replaces neither.

### planted_v2: the N3b record

- **Status: INCONCLUSIVE.** The only reason given is: not RESEARCH-PROMISING because of 26.1
  INSUFFICIENT SAMPLE. N3 also had 26.14 FAIL. 26.15 FAILs here. It is not among the criteria
  that RESEARCH-PROMISING requires (`PROMISING_REQUIRES` is 26.1, 26.2, 26.7, 26.9, 26.12 and
  26.14), so it does not appear in the status reasons.
- **D5, against the planted world's acceptance bar (BSS ≥ 0.010; see
  [ADR 0002](0002-no-terminal-status-without-a-clean-commit-and-append-only-training.md)
  item 5): PASS.**
  - The 26.1-wf BSS is 0.012516 (N3: 0.010827; N2: 0.009138).
  - The sensitivity BSS is at or above the bar at every offset:

    | Offset | T-180 | T-120 | T-60 | T-30 | T-10 |
    |---|---|---|---|---|---|
    | BSS | 0.01271 | 0.01301 | 0.01200 | 0.01252 | 0.01181 |
    | Trades (baseline) | 162 | 155 | 162 | 179 | 190 |
- **Trade count at T-30, baseline scenario pen_25: 179** (N3: 12; N2: 0).
  - All 179 are in the planted subset and in the >8% edge bucket.
  - Pass reasons: below_min_edge 13,661, and same_game_position 65.
- **Criteria.**

  | Criterion | State | Values |
  |---|---|---|
  | 26.1 | INSUFFICIENT | holdout locked, not evaluated |
  | 26.1-wf | PASS | 7,429 games, 13,905 observations; Brier: model 0.246721, market 0.249848; delta 0.003127 (CI 0.002010–0.004209), p 0.0 |
  | 26.2 | PASS | 5 of 6 periods improved (2023H2 did not: BSS 0.00027); max share 0.306 |
  | 26.3 | PASS | LOCKED; 0 holdout rows |
  | 26.4 | PASS | 2 feature families registered before the earliest experiment |
  | 26.5 | PASS | 63 experiments: 35 in the FDR set across 3 families, plus 28 controls; 0 dirty; 24 successes |
  | 26.6 | PASS | 6 sufficient buckets; 3 at or above; min 0.0633 |
  | 26.7 | PASS | Spearman 0.886; coefficient 2.643, se 0.711, p 0.0002 |
  | 26.8 | PASS | 179 trades; net after fees and slippage 7,347.79 on stake 17,888.93 (ROI 0.411); fee schedule: the venue's general taker schedule, v1, unverified |
  | 26.9 | PASS | net P&L: pen_0 7,604.92, pen_25 7,347.79, pen_50 7,064.55, pen_100 6,507.20 |
  | 26.10 | PASS | 13,905 of 13,905; 1,035 exclusions; size verified 1.0 |
  | 26.11 | PASS | all 7 leakage counts are 0 |
  | 26.12 | PASS | coverage of p_true 0.875 (band 0.87–0.93); mean width 0.152 |
  | 26.13 | PASS | ECE 0.01587 vs the market's 0.04433; slope 0.686 (0.559–0.819) |
  | 26.14 | PASS | the top 10 trades are 18.6% of positive P&L (limit 40%); the best team is 12.4% (limit 20%); net P&L without the top ten 4,088.53 |
  | 26.15 | FAIL | 3 periods with trades: 2025H1 122 trades (realized edge 0.161), 2025H2 2 (−0.250), 2026H1 55 (0.149); both recent periods are below the aggregate 0.153; slope −0.006 (CI −2.979 to 2.967) |
  | 26.16 | PASS | 4 candidates, none working |
  | 26.17 | PASS | observed 0.003127; permutation mean 0.001707, sd 0.000225; p 0.000999 (1,000 permutations, 60 groups) |
  | 26.18 | INSUFFICIENT | no settled paper trades |

- **Edge buckets** (baseline scenario):

  | Bucket | Obs | Games | Mean predicted | Realized | CI | Label | Trades |
  |---|---|---|---|---|---|---|---|
  | <0% | 5,113 | 3,189 | −0.01102 | 0.01694 | [−0.000232, 0.03410] | OK | 0 |
  | 0–2% | 3,450 | 2,551 | 0.009396 | −0.002096 | [−0.02189, 0.01810] | OK | 0 |
  | 2–4% | 2,070 | 1,537 | 0.02885 | 0.06454 | [0.03873, 0.09084] | OK | 0 |
  | 4–6% | 1,039 | 782 | 0.04943 | 0.06333 | [0.02656, 0.1009] | OK | 0 |
  | 6–8% | 891 | 653 | 0.06995 | 0.09180 | [0.05116, 0.1319] | OK | 0 |
  | >8% | 1,342 | 800 | 0.1030 | 0.1120 | [0.07681, 0.1472] | OK | 179 |

- **26.12 coverage of p_true, split by subset.** These are the T-30 evaluation rows. The script is
  N2's coverage script, run on the N3b run. The recomputed overall coverage equals the recorded
  0.875225.

  | Rows | N | N3b coverage | N3 | N2 | p_true above upper | p_true below lower |
  |---|---|---|---|---|---|---|
  | in the planted subset | 3,662 | 0.8768 | 0.7978 | 0.5544 | 213 (0.0582) | 238 (0.0650) |
  | outside the subset | 10,243 | 0.8746 | 0.9024 | 0.9175 | 643 (0.0628) | 641 (0.0626) |
  | all | 13,905 | 0.8752 | 0.8746 | 0.8210 | 856 (0.0616) | 879 (0.0632) |

  Outside minus inside: −0.0022. Inside coverage is now slightly above outside (N3: 0.1046; N2:
  0.3631). The mean interval width is 0.15206 (N3: 0.15351).

### The [ADR 0004](0004-n3b-predictions.md) predictions, against their readings

| Prediction | Reading (ADR 0004) | Measured | Held? |
|---|---|---|---|
| P1: D5 passes | planted_v2 26.1-wf BSS ≥ 0.010 | 0.012516 | **held** |
| P2: in-subset 26.12 coverage rises toward the outside level | inside > 0.5544, and outside − inside < 0.3631 | inside 0.8768; outside − inside −0.0022 | **held** |
| P3: T-30 model trades remain near zero | at most 23 baseline T-30 trades | 179 | **not held** |
| P4: the regenerated null's redraw p-value is not below 0.05 | share of 1,000 redraws with BSS ≥ observed is ≥ 0.05 | 0.7400 (740 of 1,000); 0.7403 as (1 + count) / (1 + draws) | **held** |

As [ADR 0004](0004-n3b-predictions.md) says, P4 holding is consistent with the N3 null tail
being chance. It does not rule out leakage: redrawing outcomes cannot separate a lucky draw from
predictions that carry outcome information.

### null_v1: the control world

- **Status: INCONCLUSIVE.** The reasons: 26.1 INSUFFICIENT, 26.2 FAIL, 26.7 FAIL, 26.9 FAIL,
  26.12 FAIL and 26.14 FAIL.
- **26.1-wf FAILs.**
  - BSS −0.003021 (N3: +0.001472; N2: +0.0000289).
  - The delta Brier is −0.000741, with CI −0.001284 to −0.000201 and p 0.9965.
  - The model is worse than the market. Most of the shortfall is one period: 2024H2 has delta
    Brier −0.004416 (BSS −0.0179). The other five periods lie between −0.00028 and +0.00003.
- **26.2 FAILs:** 0 of 6 periods improved (N3: 4 of 6).
- **26.6 FAILs:** realized edge −0.0353 in the >8% bucket, the only sufficient bucket at or
  above the minimum edge.
- **26.7 FAILs:** Spearman −0.4 over 4 buckets; coefficient −0.280, p 0.771. N3 had PASS.
- **Trades.**
  - 7 at T-30 in the baseline scenario, all in the >8% bucket, net −14.01 on stake 697.58, so
    26.8 and 26.9 FAIL.
  - By offset: T-180 6, T-120 7, T-60 8, T-30 7, T-10 8. N3 had 0 at every offset.
  - The sensitivity BSS is −0.0030 to −0.0032 at every offset.
- **26.12 FAILs:** coverage 0.823; mean width 0.130 (N3: 0.818 and 0.117).
- **26.13 PASSes:** ECE 0.00920 against the market's 0.00474; slope 0.716 (0.531–0.889).
- **26.14 FAILs:** the top 10 trades are 100% of positive P&L; one team is 75.9%.
- **26.15 is INSUFFICIENT:** 1 period with trades.
- **26.16 PASSes.**
- **26.17 FAILs:** p 0.1499 (N3: PASS, p 0.000999).

### The 2x2 at T-30, both worlds (report only)

**Harness.** The harness is the one N3 used (its records are not published). Its four cells are
(a) the oracle `p_true` with no haircut; (b) the oracle `p_true` with each row's own model
half-width, in logit units, as the haircut; (c) the model with no haircut; (d) the model with the
haircut, which is the real engine. The edge is also measured against `p_true`.
- The CI uses a game-level bootstrap: 10,000 resamples, seed 7, 95% percentile interval.
- Each cell has one trade per game.
- Baseline pen_25, `minimum_edge` 0.04. 13,905 of 13,905 rows are kept in each world.
- Harness check: recomputing cell d reproduces the stored decisions (decision, pass reason and
  net edge) on every kept row, in both worlds.

**The columns.** "Engine net edge" is the engine's `net_edge` against the cell's own
`p_conservative`. "Edge vs `p_true`" is `p_true(side)` minus the all-in price.

**planted_v2** (`bt-bf08a3dbce5f3d07`):

| Cell | Trades | In / out of subset | Mean engine net edge | Mean edge vs `p_true` | 95% CI | Share > 0 | Realized net P&L (stake) | N3 trades |
|---|---:|---|---:|---:|---|---:|---|---:|
| a. oracle `p_true`, no haircut | 2,349 | 1,595 / 754 | +0.08211 | +0.08211 | [+0.08080, +0.08343] | 1.000 | 56,300.19 (234,265.22) | 2,357 |
| b. oracle `p_true`, model half-width (logit) | 480 | 442 / 38 | +0.06397 | +0.12005 | [+0.11782, +0.12235] | 1.000 | 18,487.36 (47,915.82) | 431 |
| c. N3b model, no haircut | 1,268 | 1,162 / 106 | +0.07196 | +0.08365 | [+0.08135, +0.08596] | 0.971 | 32,545.61 (126,636.09) | 978 |
| d. N3b model, with haircut (the real engine) | 179 | 179 / 0 | +0.05266 | +0.10404 | [+0.09863, +0.10927] | 0.994 | 7,347.79 (17,888.93) | 12 |

Pass reasons (below_min_edge / same_game_position):
- a: 9,850 / 1,706;
- b: 13,139 / 286;
- c: 11,742 / 895;
- d: 13,661 / 65.

**null_v1** (`bt-b14c63c6ddec1c50`; nothing is planted, so every trade is outside the subset):

| Cell | Trades | Mean engine net edge | Mean edge vs `p_true` | 95% CI | Share > 0 | Realized net P&L (stake) | N3 trades |
|---|---:|---:|---:|---|---:|---|---:|
| a. oracle `p_true`, no haircut | 434 | +0.05330 | +0.05330 | [+0.05214, +0.05451] | 1.000 | 6,319.46 (43,259.79) | 402 |
| b. oracle `p_true`, model half-width (logit) | 9 | +0.05137 | +0.09434 | [+0.08772, +0.10077] | 1.000 | −146.33 (899.44) | 4 |
| c. N3b model, no haircut | 347 | +0.07137 | −0.03118 | [−0.03530, −0.02714] | 0.228 | −2,527.58 (34,581.15) | 1 |
| d. N3b model, with haircut (the real engine) | 7 | +0.26064 | −0.04034 | [−0.07673, −0.00455] | 0.286 | −14.01 (697.58) | 0 |

Pass reasons (below_min_edge / same_game_position):
- a: 13,268 / 203;
- b: 13,892 / 4;
- c: 13,345 / 213;
- d: 13,898 / 0.

In the null world, the model's trades lose against `p_true`, with and without the haircut. Both
CIs exclude 0. Cell d rests on 7 trades. Their mean engine net edge is +0.26064 (median
+0.27306), against −0.04034 measured against `p_true`.

### The null redraw (P4)

The procedure is N3's null redraw check, as fixed in [ADR 0004](0004-n3b-predictions.md): the
same script, run on the N3b null run.
- 13,905 evaluation predictions, 7,429 games, 6 periods.
- 1,000 redraws, seed 20260926. Each redraw is one Bernoulli(`p_true_home`) per game.
- `p_model`, `p_market` and `p_true` are held fixed.
- Both checks passed: the recomputed observed BSS equals the referee's `paired_game_bootstrap`
  value, and each game's contracts agree on the recorded outcome and on `p_true`.

| Quantity | N3b | N3 (pre-fix) |
|---|---|---|
| Observed 26.1-wf BSS | −0.003021 | +0.001472 |
| Redraw mean, sd | −0.002307, 0.001171 | −0.000366, 0.000792 |
| Redraw min, max | −0.007232, +0.001171 | −0.003888, +0.002647 |
| Redraw 97.5% and 99% quantiles | −0.000007, +0.000339 | +0.001175, +0.001425 |
| Redraws at or above the observed value | 740 of 1,000 (percentile rank 0.260) | 9 of 1,000 (0.991) |
| Empirical one-sided p-value | 0.7400 (share at or above); 0.7403 as (1 + count) / (1 + draws) | 0.0090; 0.0100 |
| z against the redraws | −0.61 | +2.32 |
| Redraws at or above the 26.1 bar (`bss_min` 0.005) | 0 of 1,000 | 0 of 1,000 |
