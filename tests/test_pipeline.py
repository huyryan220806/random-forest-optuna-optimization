"""Verify the S1 protocol, actual CV execution and command-line outputs."""

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
            [sys.executable, "-m", "src.experiment", "--seed", "42",
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
