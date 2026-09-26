"""ADR 0002 item 4 (the owner's ITEM 13): `quant train` never overwrites an earlier train.

Before this, every `quant train` replaced model_runs and predictions whole, and a model run id did
not take the commit or the time, so a rerun on a later commit silently replaced the rows (and the
model files) an earlier backtest had read. Now, as for backtest runs (ADR 0001 N1), each invocation
gets its own ids, the two tables only grow, and every reader takes the latest invocation whole.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from factories import frame, observation
from src.backtest import holdout as H
from src.backtest.engine import run_backtest
from src.backtest.registry import load_hypotheses
from src.backtest.walk_forward import holdout_period, latest_model_runs, load_periods
from src.config import load_sport_config
from src.features.mlb import write_features
from src.models import train as T
from src.models.residual import FittedModel
from src.portfolio import paper as P
from src.sports.synthetic import SyntheticSource
from src.storage.database import ConstraintViolation, connect
from src.storage.ingest import ingest
from src.storage.schemas import ALL_TABLES

TRAIN_TABLES = ("model_runs", "predictions", "train_lineage")          # train_lineage: run lineage
TABLE_SQL = {t: f"SELECT * FROM {t} ORDER BY ALL" for t in TRAIN_TABLES}
USED_SQL = "SELECT DISTINCT model_run_id FROM backtest_decisions"
WHEN = datetime(2026, 2, 1, tzinfo=timezone.utc)
CLEAN_A, CLEAN_B = "a" * 40, "b" * 40


# ---------------------------------------------------------------------------
# storage: the two train tables only grow
# ---------------------------------------------------------------------------

def run_row(rid: str, *, period: str = "2025H1", model_type: str = "logistic_residual", created=WHEN,
            commit: str | None = CLEAN_A, **over) -> dict:
    return {"model_run_id": rid, "model_type": model_type, "period_id": period, "feature_set_id": "fs",
            "feature_set_hash": "fsh", "config_hash": "c", "predict_from": pd.Timestamp("2025-01-01T00:00Z"),
            "dataset": "synthetic", "git_commit": commit, "created_at": pd.Timestamp(created),
            "is_synthetic": True} | over


def prediction(rid: str, oid: str = "SYN-NYY@T-30", p_model: float = 0.70, period: str = "2025H1") -> dict:
    return {"model_run_id": rid, "observation_id": oid, "period_id": period, "p_market": 0.56, "p_model": p_model,
            "lower_bound": p_model - 0.04, "upper_bound": p_model + 0.04, "p_conservative_yes": p_model - 0.04,
            "p_conservative_no": 1 - (p_model + 0.04), "is_synthetic": True}


@pytest.mark.parametrize("table", TRAIN_TABLES)
def test_the_train_tables_are_append_only(table):
    assert ALL_TABLES[table].append_only


@pytest.mark.parametrize("table", TRAIN_TABLES)
def test_replace_refuses_a_train_table(syn_db, table):
    syn_db.append_tables({"model_runs": frame(run_row("mr-1")), "predictions": frame(prediction("mr-1"))})
    empty = pd.DataFrame(columns=ALL_TABLES[table].column_names)
    with pytest.raises(ValueError, match=f"{table} is append-only.*ADR 0001"):
        syn_db.replace(table, empty, "model_run_id = ?", ["mr-1"])
    # a replaceable table listed first (walk_forward_periods was, until it became append-only)
    fees = pd.DataFrame(columns=ALL_TABLES["fee_schedules"].column_names)
    with pytest.raises(ValueError, match=f"{table} is append-only.*ADR 0001"):
        syn_db.replace_tables({"fee_schedules": fees, table: empty})
    assert syn_db.count("model_runs") == syn_db.count("predictions") == 1


# ---------------------------------------------------------------------------
# the id: one per invocation
# ---------------------------------------------------------------------------

ID_PARTS = dict(model_type="logistic_residual", period_id="P1", feature_hash="f", config_hash="c",
                data_fingerprint="d", git_commit=CLEAN_A, created_at=WHEN)


@pytest.mark.parametrize("field, other", [
    ("git_commit", CLEAN_B),                                     # a new commit
    ("git_commit", CLEAN_A + H.DIRTY),                           # the same HEAD with uncommitted edits
    ("git_commit", None),                                        # no git
    ("created_at", WHEN + timedelta(microseconds=1)),            # a plain rerun
    ("model_type", "lgbm_residual"), ("period_id", "P2"), ("feature_hash", "f2"), ("config_hash", "c2"),
    ("data_fingerprint", "d2"),
])
def test_the_model_run_id_changes_with_every_part(field, other):
    assert T.model_run_id(**ID_PARTS) == T.model_run_id(**ID_PARTS)
    assert T.model_run_id(**ID_PARTS | {field: other}) != T.model_run_id(**ID_PARTS)
    assert T.model_run_id(**ID_PARTS).startswith("mr-")


# ---------------------------------------------------------------------------
# quant train, twice in one data root
# ---------------------------------------------------------------------------

DAY = date(2026, 4, 6)
NOW_A = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
NOW_B = NOW_A + timedelta(hours=1)


def small_season():
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": DAY.isoformat()}
    cfg["walk_forward"]["periods"] = [
        {"id": "P0", "role": "train_only", "start": date(2026, 3, 1), "end": date(2026, 3, 29)},
        {"id": "P1", "role": "evaluation", "start": date(2026, 3, 30), "end": date(2026, 4, 2)},
        {"id": "PH", "role": "holdout", "start": date(2026, 4, 3), "end": date(2026, 4, 30)},
    ]
    cfg["model"]["min_train_games"] = 30
    cfg["uncertainty"]["ensemble_size"] = 5
    cfg["review"]["bootstrap_resamples"] = 200
    return cfg


def _tables(db) -> dict[str, pd.DataFrame]:
    return {t: db.df(TABLE_SQL[t]) for t in TRAIN_TABLES}


def _files(db) -> dict[str, tuple[int, bytes]]:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in T.model_dir(db).glob("*.pkl")}


@pytest.fixture(scope="module")
def twice(tmp_path_factory):
    """ingest -> features -> train (CLEAN_A) -> train (CLEAN_B, an hour later) -> the second train again
    (same commit, same time) -> backtest, on one small synthetic season."""
    root = tmp_path_factory.mktemp("train_append_only")
    cfg = small_season()
    out = SimpleNamespace(cfg=cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANT_DATA_DIR", str(root / "data"))
        db = connect("mlb", "synthetic", path=root / "world.duckdb", logs=root / "logs")
        out.append_calls = []
        write = db.append_tables

        def spy(frames):
            out.append_calls.append(frozenset(frames))
            return write(frames)

        mp.setattr(db, "append_tables", spy)
        try:
            ingest(db, SyntheticSource(cfg), date(2026, 1, 1), DAY)
            write_features(db, cfg, DAY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # expit overflow on synthetic extremes
                mp.setattr(H, "git_commit", lambda: CLEAN_A)
                out.a = T.run_training(db, cfg, now=NOW_A, threads=2)
                out.after_a = _tables(db)
                mp.setattr(H, "git_commit", lambda: CLEAN_B)
                out.b = T.run_training(db, cfg, now=NOW_B, threads=2)
                out.after_b, out.files_b = _tables(db), _files(db)
                out.exps_b = db.df("SELECT experiment_id, experiment_family, git_commit, created_at FROM experiments")
                out.selection_b = (T.model_dir(db) / "model_selection.json").read_text()
                try:
                    T.run_training(db, cfg, now=NOW_B, threads=2)
                    out.repeat_error = None
                except ConstraintViolation as exc:
                    out.repeat_error = exc
                out.after_repeat, out.files_after_repeat = _tables(db), _files(db)
                out.selection_after_repeat = (T.model_dir(db) / "model_selection.json").read_text()
                bt = run_backtest(db, cfg, dict(out.b.selection), now=NOW_B + timedelta(minutes=5))
                out.bt_model_runs = set(db.df(USED_SQL)["model_run_id"])
                out.bt_config_ids = {i for c in bt.runs["config_json"] for i in json.loads(c)["model_run_ids"]}
                out.manifest = H.current_manifest(db, cfg, dict(out.b.selection), NOW_B)
                out.latest = latest_model_runs(db)
        finally:
            db.close()
    out.ids_a, out.ids_b = set(out.a.model_runs["model_run_id"]), set(out.b.model_runs["model_run_id"])
    return out


def test_each_train_gets_its_own_ids(twice):
    assert len(twice.ids_a) == len(twice.b.model_runs) == len(twice.ids_b) > 0
    assert not twice.ids_a & twice.ids_b


def test_a_second_train_keeps_the_first_trains_rows(twice):
    runs, preds = twice.after_b["model_runs"], twice.after_b["predictions"]
    assert set(runs["model_run_id"]) == twice.ids_a | twice.ids_b
    assert dict(zip(runs["model_run_id"], runs["git_commit"])) == (
        {i: CLEAN_A for i in twice.ids_a} | {i: CLEAN_B for i in twice.ids_b})
    assert len(preds) == len(twice.after_a["predictions"]) + twice.b.n_predictions
    key = ["model_run_id", "observation_id"]
    first = preds[preds["model_run_id"].isin(twice.ids_a)].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, twice.after_a["predictions"].sort_values(key).reset_index(drop=True))
    kept = runs[runs["model_run_id"].isin(twice.ids_a)].sort_values("model_run_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, twice.after_a["model_runs"].sort_values("model_run_id").reset_index(drop=True))


def test_each_train_keeps_its_own_model_files(twice):
    runs = twice.after_b["model_runs"]
    paths = dict(zip(runs["model_run_id"], runs["artifact_path"]))
    assert len(set(paths.values())) == len(paths) == len(twice.a.model_runs) + len(twice.b.model_runs)
    for rid, path in paths.items():
        assert path.endswith(f"{rid}.pkl") and f"{rid}.pkl" in twice.files_b


def test_the_same_invocation_twice_raises_and_overwrites_nothing(twice):
    """Same commit and same time: the ids repeat, and the rerun is refused before it writes a table or
    rewrites a model file; model_selection.json stays the second train's."""
    assert isinstance(twice.repeat_error, ConstraintViolation), twice.repeat_error
    for t in TRAIN_TABLES:
        pd.testing.assert_frame_equal(twice.after_repeat[t], twice.after_b[t])
    assert twice.files_after_repeat == twice.files_b
    assert twice.selection_after_repeat == twice.selection_b


def test_the_backtest_reads_only_the_latest_train(twice):
    assert twice.bt_model_runs and twice.bt_model_runs <= twice.ids_b
    assert twice.bt_config_ids and twice.bt_config_ids <= twice.ids_b


def test_the_manifest_pins_the_latest_trains_holdout_model(twice):
    hold = holdout_period(load_periods(twice.cfg, "synthetic")).period_id
    runs = twice.after_b["model_runs"]
    new = runs[runs["model_run_id"].isin(twice.ids_b) & (runs["period_id"] == hold)
               & (runs["model_type"] == twice.b.selection["selected"])]
    assert len(new) == 1
    assert twice.manifest["model_artifact_hash"] == H.artifact_hash(new["artifact_path"].iloc[0])
    assert twice.manifest["model_config_hash"] == new["config_hash"].iloc[0]


def test_latest_model_runs_is_the_second_train(twice):
    assert set(twice.latest["model_run_id"]) == twice.ids_b



# Each train records its lineage (invocation -> model runs, experiments, hypotheses), in the
# transaction that writes its model runs

def test_each_train_records_its_own_lineage(twice):
    lin, exps = twice.after_b["train_lineage"], twice.exps_b
    hyp_ids = {h["hypothesis_id"] for h in load_hypotheses(T.hypotheses_path("mlb"))}
    invs = {}
    for name, ids, commit, now in (("a", twice.ids_a, CLEAN_A, NOW_A), ("b", twice.ids_b, CLEAN_B, NOW_B)):
        inv = set(lin.loc[(lin["kind"] == "model_run") & lin["row_id"].isin(ids), "invocation_id"])
        assert len(inv) == 1, name
        mine = lin[lin["invocation_id"].isin(inv)]
        logged = exps[(exps["git_commit"] == commit) & (exps["created_at"] == pd.Timestamp(now))]
        assert (logged["experiment_family"] == "negative_control").any(), name
        assert {k: set(g["row_id"]) for k, g in mine.groupby("kind")} == {
            "model_run": ids, "experiment": set(logged["experiment_id"]), "hypothesis": hyp_ids}, name
        assert set(mine["git_commit"]) == {commit} and set(mine["created_at"]) == {pd.Timestamp(now)}, name
        invs[name] = inv
    assert not invs["a"] & invs["b"] and set(lin["invocation_id"]) == invs["a"] | invs["b"]


def test_a_second_train_logs_its_own_experiments(twice):
    """Experiment ids take the commit and the time: the second train's rows are new, the first train's stay."""
    exps = twice.exps_b
    a, b = (set(exps.loc[exps["git_commit"] == c, "experiment_id"]) for c in (CLEAN_A, CLEAN_B))
    assert a and b and not a & b and len(a) == len(b) == twice.a.experiments_total == twice.b.experiments_total
    assert twice.b.experiments_logged == twice.b.experiments_total


def test_lineage_is_written_with_the_model_runs(twice):
    trains = [c for c in twice.append_calls if "model_runs" in c]
    assert len(trains) == 2 and all("train_lineage" in c and "predictions" in c for c in trains)
    assert all("model_runs" in c for c in twice.append_calls if "train_lineage" in c)


def test_the_invocation_id_takes_the_commit_and_the_time():
    """Two trains at one commit, a microsecond apart, are two invocations; the same inputs are one."""
    db, runs = SimpleNamespace(dataset="synthetic", is_synthetic=True), pd.DataFrame({"model_run_id": ["mr-1"]})

    def invocation(commit, now):
        rows = T._lineage_rows(db, runs, [], {"h1": {}}, commit=commit, now=now, run_hash="rh",
                               quote_quality="executable")
        assert rows["invocation_id"].nunique() == 1
        return rows["invocation_id"].iloc[0]

    base = invocation(CLEAN_A, NOW_A)
    assert invocation(CLEAN_A, NOW_A) == base
    others = [invocation(CLEAN_A, NOW_A + timedelta(microseconds=1)), invocation(CLEAN_B, NOW_A),
              invocation(CLEAN_A + "-dirty", NOW_A), invocation(None, NOW_A)]
    assert base not in others and len(set(others)) == len(others)


# ---------------------------------------------------------------------------
# readers: the latest invocation, never mixed with another (hand-made rows)
# ---------------------------------------------------------------------------

def seed(db, runs: list[dict], p_model: dict[str, float] | None = None) -> None:
    db.insert("observations", frame(observation(outcome=1), observation(offset=60, outcome=1, is_canonical=False)))
    db.append_tables({
        "model_runs": frame(*runs),
        "predictions": frame(*[prediction(r["model_run_id"], oid, (p_model or {}).get(r["model_run_id"], 0.70),
                                          r["period_id"])
                               for r in runs for oid in ("SYN-NYY@T-30", "SYN-NYY@T-60")])})


def backtest(db, selected: str = "logistic_residual"):
    return run_backtest(db, load_sport_config("mlb"), {"selected": selected}, now=WHEN + timedelta(days=1))


def used(db) -> set[str]:
    return set(db.df(USED_SQL)["model_run_id"])


@pytest.mark.parametrize("old, new", [
    (dict(created=WHEN, commit=CLEAN_A), dict(created=WHEN + timedelta(seconds=1), commit=CLEAN_B)),
    (dict(created=WHEN, commit=CLEAN_A), dict(created=WHEN + timedelta(seconds=1), commit=CLEAN_A)),
    (dict(created=WHEN, commit=CLEAN_A), dict(created=WHEN + timedelta(seconds=1), commit=None)),
    (dict(created=WHEN, commit=None), dict(created=WHEN + timedelta(seconds=1), commit=None)),
], ids=["new-commit", "same-commit", "then-no-git", "no-git-twice"])
def test_the_backtest_reads_the_latest_train_invocation(syn_db, old, new):
    seed(syn_db, [run_row("mr-old", **old), run_row("mr-new", **new)], {"mr-old": 0.90, "mr-new": 0.60})
    res = backtest(syn_db)
    assert used(syn_db) == {"mr-new"}
    assert {i for c in res.runs["config_json"] for i in json.loads(c)["model_run_ids"]} == {"mr-new"}


def test_a_tie_on_time_takes_one_commit_not_both(syn_db):
    """Two invocations at the same instant (possible only with an injected clock): the one ordered first
    by (created_at, model_run_id) descending, and only the rows with its commit."""
    seed(syn_db, [run_row("mr-a", commit=CLEAN_A), run_row("mr-b", commit=CLEAN_B)])
    backtest(syn_db)
    assert used(syn_db) == {"mr-b"}


@pytest.mark.parametrize("commit", [CLEAN_A, None], ids=["commit", "no-git"])
def test_one_invocations_rows_are_all_taken(syn_db, commit):
    """Rows sharing created_at and commit are one invocation (a missing commit matches a missing one)."""
    seed(syn_db, [run_row("mr-1", commit=commit), run_row("mr-2", period="2025H2", commit=commit),
                  run_row("mr-0", created=WHEN - timedelta(seconds=1), commit=commit)])
    assert set(latest_model_runs(syn_db)["model_run_id"]) == {"mr-1", "mr-2"}


def test_latest_model_runs_on_an_empty_table(syn_db):
    assert latest_model_runs(syn_db).empty


def test_the_latest_train_without_the_selected_type_is_not_filled_from_an_older_one(syn_db):
    seed(syn_db, [run_row("mr-old-lgbm", model_type="lgbm_residual"),
                  run_row("mr-new-logit", created=WHEN + timedelta(seconds=1), commit=CLEAN_B)])
    with pytest.raises(RuntimeError, match="no lgbm_residual model runs"):
        backtest(syn_db, "lgbm_residual")
    assert syn_db.count("backtest_decisions") == 0


class Loaded(Exception):
    """Raised by the patched FittedModel.load: carries the artifact path a reader chose."""


def test_explain_loads_the_latest_trains_model(syn_db, monkeypatch):
    """A settled evaluation-period contract is explained with the latest train's model for its period."""
    def load(path):
        raise Loaded(str(path))

    monkeypatch.setattr(FittedModel, "load", staticmethod(load))
    latest = dict(created=WHEN + timedelta(seconds=1), commit=CLEAN_B)
    seed(syn_db, [run_row("mr-old", artifact_path="models/mr-old.pkl"),
                  run_row("mr-new", artifact_path="models/mr-new.pkl", **latest),
                  # the same train's model for an earlier period, which sorts first: never this contract's
                  run_row("mr-new-2024H2", period="2024H2", artifact_path="models/mr-new-2024H2.pkl", **latest)])
    sel = {"selected": "logistic_residual"}
    with pytest.raises(Loaded, match=r"models/mr-new\.pkl$"):
        P.explain(syn_db, load_sport_config("mlb"), sel, "SYN-NYY", now=WHEN + timedelta(days=1))


def test_the_manifest_pins_the_latest_holdout_run(syn_db, tmp_path):
    cfg = load_sport_config("mlb")
    hold = holdout_period(load_periods(cfg, "synthetic")).period_id
    old, new = tmp_path / "mr-old.pkl", tmp_path / "mr-new.pkl"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    syn_db.append_tables({"model_runs": frame(
        run_row("mr-old", period=hold, config_hash="c-old", artifact_path=str(old)),
        run_row("mr-new", period=hold, config_hash="c-new", artifact_path=str(new),
                created=WHEN + timedelta(seconds=1), commit=CLEAN_B))})
    m = H.current_manifest(syn_db, cfg, {"selected": "logistic_residual", "feature_set_hash": "f"}, WHEN)
    assert m["model_config_hash"] == "c-new"
    assert m["model_artifact_hash"] == H.artifact_hash(new)


def test_the_manifest_pins_only_the_selected_types_model(syn_db, tmp_path):
    """The latest train's holdout rows are the market model and one residual model; a selection naming
    another type pins nothing, rather than whichever row sorts first."""
    cfg = load_sport_config("mlb")
    hold = holdout_period(load_periods(cfg, "synthetic")).period_id
    path = tmp_path / "mr-logit.pkl"
    path.write_bytes(b"logit")
    syn_db.append_tables({"model_runs": frame(
        run_row("mr-market", period=hold, model_type="market", config_hash="c-market", artifact_path=str(path)),
        run_row("mr-logit", period=hold, config_hash="c-logit", artifact_path=str(path)))})
    m = H.current_manifest(syn_db, cfg, {"selected": "lgbm_residual", "feature_set_hash": "f"}, WHEN)
    assert m["model_config_hash"] == "unavailable: no final-period model"
    assert m["model_artifact_hash"] == "unavailable: no final-period model"
    m = H.current_manifest(syn_db, cfg, {"selected": "logistic_residual", "feature_set_hash": "f"}, WHEN)
    assert m["model_config_hash"] == "c-logit"
