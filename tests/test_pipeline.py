"""Verify S1/S2 data isolation, actual CV execution and persisted results."""

import csv
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from src import experiment


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_labels_split_and_reproducibility():
    config = experiment.ExperimentConfig()
    data = experiment.load_and_split(config)
    again = experiment.load_and_split(config)
    original = load_breast_cancer()
    assert data.X_train.shape == (455, 30)
    assert data.X_test.shape == (114, 30)
    assert np.isfinite(data.X_train).all() and np.isfinite(data.X_test).all()
    assert set(data.train_indices).isdisjoint(data.test_indices)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([data.train_indices, data.test_indices])), np.arange(569)
    )
    np.testing.assert_array_equal(data.train_indices, again.train_indices)
    np.testing.assert_array_equal(data.test_indices, again.test_indices)
    for indices, X, y in (
        (data.train_indices, data.X_train, data.y_train),
        (data.test_indices, data.X_test, data.y_test),
    ):
        np.testing.assert_array_equal(X, original.data[indices])
        np.testing.assert_array_equal(y, (original.target[indices] == 0).astype(int))
    assert experiment.class_counts(data.y_train) == {"benign_0": 285, "malignant_1": 170}
    assert experiment.class_counts(data.y_test) == {"benign_0": 72, "malignant_1": 42}


def test_fixed_stratified_folds_never_include_holdout():
    config = experiment.ExperimentConfig()
    data = experiment.load_and_split(config)
    folds = experiment.make_cv_splits(data.X_train, data.y_train, config)
    repeated = experiment.make_cv_splits(data.X_train, data.y_train, config)
    assert len(folds) == 3
    all_valid = []
    for (train, valid), (train_again, valid_again) in zip(folds, repeated):
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(valid, valid_again)
        assert set(train).isdisjoint(valid)
        assert set(train) | set(valid) == set(range(len(data.y_train)))
        assert set(data.train_indices[train]).isdisjoint(data.test_indices)
        assert set(data.train_indices[valid]).isdisjoint(data.test_indices)
        assert abs(data.y_train[valid].mean() - data.y_train.mean()) < 0.01
        all_valid.extend(valid)
    np.testing.assert_array_equal(np.sort(all_valid), np.arange(len(data.y_train)))


def test_baseline_preserves_library_defaults():
    defaults = RandomForestClassifier().get_params()
    actual = experiment.make_baseline(experiment.ExperimentConfig()).get_params()
    assert {key for key in defaults if defaults[key] != actual[key]} == {
        "random_state", "n_jobs"
    }
    assert actual["random_state"] == 42
    assert actual["n_jobs"] == 1
    assert actual["n_estimators"] == 100
    assert actual["max_depth"] is None
    assert actual["min_samples_leaf"] == 1


def test_real_cv_receives_only_training_data(monkeypatch, tmp_path):
    config = experiment.ExperimentConfig()
    expected = experiment.load_and_split(config)
    real_cross_validate = experiment.cross_validate
    calls = []

    def checked_cross_validate(model, X, y, **kwargs):
        np.testing.assert_array_equal(X, expected.X_train)
        np.testing.assert_array_equal(y, expected.y_train)
        assert kwargs["scoring"] == "roc_auc"
        assert kwargs["error_score"] == "raise"
        assert kwargs["n_jobs"] == 1
        calls.append(len(X))
        return real_cross_validate(model, X, y, **kwargs)

    monkeypatch.setattr(experiment, "cross_validate", checked_cross_validate)
    result = experiment.run_baseline(config, tmp_path)
    assert calls == [455]
    assert result["holdout_evaluated"] is False
    assert result["n_trials"] == 0
    assert result["cv"]["data_scope"] == "train_only"
    assert json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8")) == result


def test_cli_runs_twice_with_identical_metrics_and_separate_outputs(tmp_path):
    runs = []
    for name in ("smoke", "repro"):
        output = tmp_path / name
        completed = subprocess.run(
            [sys.executable, "-m", "src.experiment", "--baseline-only", "--seed", "42",
             "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        assert "Holdout not evaluated; Optuna trials: 0" in completed.stdout
        assert [path.name for path in output.iterdir()] == ["baseline.json"]
        result = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
        scores = np.array(result["cv_auc_folds"])
        assert scores.shape == (3,)
        assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        assert result["cv_auc_mean"] == pytest.approx(scores.mean())
        assert result["cv_auc_std"] == pytest.approx(scores.std(ddof=0))
        assert result["duration_sec"] > 0
        assert len(result["fit_time_sec_per_fold"]) == 3
        assert len(result["score_time_sec_per_fold"]) == 3
        assert result["versions"]["scikit-learn"]
        assert result["environment"]["machine"]
        assert result["seeds"] == {"split": 42, "cv": 42, "random_forest": 42}
        runs.append(result)
    for field in ("params", "cv_auc_folds", "cv_auc_mean", "cv_auc_std", "split", "cv"):
        assert runs[0][field] == runs[1][field]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_seed_is_rejected(seed, tmp_path):
    with pytest.raises(ValueError, match="seed"):
        experiment.ExperimentConfig(seed=seed)
    with pytest.raises(SystemExit) as error:
        experiment.parse_args(["--seed", str(seed), "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert not (tmp_path / "baseline.json").exists()


def read_trials(output_dir):
    with (output_dir / "trials.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_search_splits_once_and_shares_train_and_folds(monkeypatch, tmp_path):
    real_load = experiment.load_and_split
    real_folds = experiment.make_cv_splits
    real_cv = experiment.cross_validate
    observed = {}
    split_calls, fold_calls, evaluated_models = [], [], []

    def load(config):
        split_calls.append(1)
        observed["data"] = real_load(config)
        return observed["data"]

    def folds(X, y, config):
        fold_calls.append(1)
        observed["folds"] = real_folds(X, y, config)
        return observed["folds"]

    def cv(model, X, y, **kwargs):
        data = observed["data"]
        assert X is data.X_train and y is data.y_train
        assert kwargs["cv"] is observed["folds"]
        assert kwargs["scoring"] == "roc_auc" and kwargs["n_jobs"] == 1
        params = model.get_params()
        baseline = experiment.make_baseline(experiment.ExperimentConfig()).get_params()
        assert all(params[key] == value for key, value in baseline.items()
                   if key not in experiment.SEARCH_SPACE)
        evaluated_models.append(model)
        return real_cv(model, X, y, **kwargs)

    monkeypatch.setattr(experiment, "load_and_split", load)
    monkeypatch.setattr(experiment, "make_cv_splits", folds)
    monkeypatch.setattr(experiment, "cross_validate", cv)
    config = experiment.ExperimentConfig()
    result = experiment.run_experiment(config, 3, tmp_path)
    assert len(split_calls) == len(fold_calls) == 1
    assert len(evaluated_models) == 4  # Baseline + three fresh candidate models.
    assert len({id(model) for model in evaluated_models}) == 4
    data = observed["data"]
    objective = experiment.make_objective(data.X_train, data.y_train, observed["folds"], config)
    captured = inspect.getclosurevars(objective).nonlocals
    assert set(captured) == {"X_train", "y_train", "cv_splits", "config"}
    assert captured["X_train"] is data.X_train
    assert captured["y_train"] is data.y_train
    assert result["holdout_evaluated"] is False
    assert result["status"] == "complete" and result["n_complete"] == 3


def test_search_cli_logging_and_reproducibility(tmp_path):
    runs, journals = [], []
    for name in ("smoke", "repro"):
        output = tmp_path / name
        subprocess.run(
            [sys.executable, "-m", "src.experiment", "--trials", "3", "--seed", "42",
             "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        assert {p.name for p in output.iterdir()} == {"baseline.json", "trials.csv", "best_config.json"}
        baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
        best = json.loads((output / "best_config.json").read_text(encoding="utf-8"))
        rows = read_trials(output)
        assert len(rows) == best["n_trials"] == best["n_trials_recorded"] == best["n_complete"] == 3
        assert best["n_failed"] == 0 and best["status"] == "complete"
        assert [int(row["trial_number"]) for row in rows] == list(range(3))
        for row in rows:
            assert row["state"] == "COMPLETE" and row["error"] == ""
            assert int(row["n_estimators"]) in [50, 100, 150, 200, 250]
            assert row["max_depth"] in {"None", "4", "6", "8", "12", "16", "24"}
            assert 1 <= int(row["min_samples_leaf"]) <= 8
            scores = np.array(json.loads(row["cv_auc_folds"]))
            assert scores.shape == (3,) and np.isfinite(scores).all()
            assert ((0 <= scores) & (scores <= 1)).all()
            assert float(row["cv_auc_mean"]) == pytest.approx(scores.mean())
            assert float(row["cv_auc_std"]) == pytest.approx(scores.std(ddof=0))
            assert float(row["duration_sec"]) >= float(row["cv_duration_sec"]) > 0
            assert int(row["seed"]) == 42
        winner = max(rows, key=lambda row: float(row["cv_auc_mean"]))
        assert best["best_trial_number"] == int(winner["trial_number"])
        assert best["best_cv_auc_mean"] == float(winner["cv_auc_mean"])
        assert best["best_cv_auc_std"] == float(winner["cv_auc_std"])
        assert best["best_params"] == {
            "n_estimators": int(winner["n_estimators"]),
            "max_depth": None if winner["max_depth"] == "None" else int(winner["max_depth"]),
            "min_samples_leaf": int(winner["min_samples_leaf"]),
        }
        assert best["cv_auc_improvement"] == pytest.approx(best["best_cv_auc_mean"] - baseline["cv_auc_mean"])
        for key in ("split", "cv", "versions", "environment", "seeds"):
            assert baseline[key] == best[key]
        assert best["seeds"]["optuna"] == 42
        assert best["sampler"]["name"] == "TPESampler" and best["direction"] == "maximize"
        assert best["sampler"]["n_startup_trials"] == 10
        assert best["search_duration_sec"] >= sum(float(row["duration_sec"]) for row in rows)
        assert baseline["baseline_search_trials"] == 0
        assert not best["holdout_evaluated"] and not baseline["holdout_evaluated"]
        runs.append(best)
        journals.append([{k: v for k, v in row.items() if k not in {"duration_sec", "cv_duration_sec"}} for row in rows])
    assert journals[0] == journals[1]
    for key in ("best_params", "best_cv_auc_mean", "best_cv_auc_std", "best_cv_auc_folds"):
        assert runs[0][key] == runs[1][key]


@pytest.mark.parametrize("fail_on_call", [2, 3])
def test_failed_trials_are_logged_without_valid_scores(monkeypatch, tmp_path, fail_on_call):
    real_evaluate = experiment.evaluate_model
    calls = []

    def evaluate(*args):
        calls.append(1)
        if len(calls) == fail_on_call:
            raise ValueError("deliberate test failure")
        return real_evaluate(*args)

    monkeypatch.setattr(experiment, "evaluate_model", evaluate)
    with pytest.raises(ValueError, match="deliberate test failure"):
        experiment.run_experiment(experiment.ExperimentConfig(), 3, tmp_path)
    rows = read_trials(tmp_path)
    assert len(rows) == fail_on_call - 1
    assert rows[-1]["state"] == "FAIL"
    assert rows[-1]["cv_auc_mean"] == rows[-1]["cv_auc_std"] == rows[-1]["cv_auc_folds"] == ""
    assert rows[-1]["error"] == "ValueError: deliberate test failure"
    assert float(rows[-1]["duration_sec"]) > 0
    best = json.loads((tmp_path / "best_config.json").read_text(encoding="utf-8"))
    assert best["status"] == "incomplete" and best["n_failed"] == 1
    assert best["n_complete"] == fail_on_call - 2
    assert best["error"] == rows[-1]["error"]
    if fail_on_call == 2:
        assert best["best_params"] is None and best["best_cv_auc_mean"] is None
    else:
        assert best["best_trial_number"] == 0


@pytest.mark.parametrize("trials", [0, -1, 31])
def test_invalid_budget_is_rejected(trials, tmp_path):
    with pytest.raises(SystemExit) as error:
        experiment.parse_args(["--trials", str(trials), "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_smoke_cannot_overwrite_main_results():
    path = ROOT / "results" / "baseline.json"
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        experiment.parse_args(["--trials", "3", "--output-dir", str(path.parent)])
    assert error.value.code == 2
    assert path.read_bytes() == before
    assert experiment.parse_args(["--trials", "3"]).output_dir == Path("results/smoke")
    assert experiment.parse_args([]).trials == 30


def test_baseline_only_cannot_mix_with_search_outputs(tmp_path):
    (tmp_path / "trials.csv").write_text("existing results", encoding="utf-8")
    with pytest.raises(ValueError, match="separate output directory"):
        experiment.run_baseline(experiment.ExperimentConfig(), tmp_path)
    assert not (tmp_path / "baseline.json").exists()
