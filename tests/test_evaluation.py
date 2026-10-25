"""S3 checks use synthetic holdout predictions; official holdout is run separately."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src import experiment, plots


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_search(tmp_path):
    for name in ("baseline.json", "best_config.json", "trials.csv"):
        shutil.copyfile(ROOT / "results" / name, tmp_path / name)
    return tmp_path


def test_holdout_metrics_fit_train_once_and_predict_test_once(monkeypatch):
    train = np.ones((6, 2))
    holdout = np.zeros((4, 2))
    data = experiment.DataSplit(train, holdout, np.array([0, 1, 0, 1, 0, 1]),
                                np.array([0, 1, 1, 0]), np.arange(6), np.arange(6, 10))
    calls = []

    class Model:
        classes_ = np.array([0, 1])

        def fit(self, X, y):
            assert X is train and y is data.y_train
            calls.append("fit")

        def predict_proba(self, X):
            assert X is holdout
            calls.append("predict")
            return np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]])

    monkeypatch.setattr(experiment, "RandomForestClassifier", lambda **kwargs: Model())
    metrics = experiment.fit_and_evaluate({}, data)
    assert calls == ["fit", "predict"]
    assert metrics["test_roc_auc"] == pytest.approx(0.75)
    assert metrics["test_f1"] == pytest.approx(0.5)
    assert metrics["test_accuracy"] == metrics["test_balanced_accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["prediction_calls"] == 1
    assert metrics["fit_duration_sec"] >= 0 and metrics["inference_duration_sec"] >= 0


def synthetic_metrics(params, data):
    # Test fixture, never used for official results.
    return {"params": params, "test_roc_auc": 0.75, "test_f1": 0.5,
            "test_accuracy": 0.5, "test_balanced_accuracy": 0.5,
            "confusion_matrix": [[36, 36], [21, 21]], "fit_duration_sec": 0.1,
            "inference_duration_sec": 0.01, "inference_sec_per_sample": 0.01 / 114,
            "prediction_calls": 1}


def test_final_resume_and_idempotence_without_second_prediction(monkeypatch, saved_search):
    calls = []

    def evaluate(params, data):
        checkpoint = json.loads((saved_search / "final_metrics.json").read_text())
        assert checkpoint["selection"]["model"] == "optimized"
        assert checkpoint["selection"]["uses_holdout"] is False
        calls.append(params["n_estimators"])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return synthetic_metrics(params, data)

    monkeypatch.setattr(experiment, "fit_and_evaluate", evaluate)
    before = experiment.source_hashes(saved_search)
    with pytest.raises(RuntimeError, match="interruption"):
        experiment.run_final_evaluation(saved_search, saved_search)
    partial = json.loads((saved_search / "final_metrics.json").read_text())
    assert partial["status"] == "in_progress"
    assert set(partial["models"]) == {"baseline"}
    result = experiment.run_final_evaluation(saved_search, saved_search)
    assert calls == [100, 50, 50]  # Saved baseline is never repeated.
    assert result["status"] == "complete" and result["holdout_evaluated"]
    path = saved_search / "final_metrics.json"
    original_bytes = path.read_bytes()
    assert experiment.run_final_evaluation(saved_search, saved_search) == result
    assert calls == [100, 50, 50] and path.read_bytes() == original_bytes
    assert experiment.source_hashes(saved_search) == before
    with pytest.raises(ValueError, match="Final evaluation exists"):
        experiment.run_experiment(experiment.ExperimentConfig(), 30, saved_search)


@pytest.mark.parametrize("corruption", ["winner", "metadata", "scores"])
def test_invalid_logs_rejected_before_holdout(monkeypatch, saved_search, corruption):
    def never_evaluate(*args):
        pytest.fail("Invalid logs must not reach the holdout")

    monkeypatch.setattr(experiment, "fit_and_evaluate", never_evaluate)
    path = saved_search / "best_config.json"
    best = json.loads(path.read_text())
    if corruption == "winner":
        best["best_params"]["n_estimators"] = 250
    elif corruption == "metadata":
        best["split"]["random_state"] = 7
    else:
        best["best_cv_auc_mean"] = 0.5
    path.write_text(json.dumps(best))
    with pytest.raises(ValueError):
        experiment.run_final_evaluation(saved_search, saved_search)
    assert not (saved_search / "final_metrics.json").exists()


def test_plots_and_comparison_are_derived_from_logs(monkeypatch, saved_search):
    monkeypatch.setattr(experiment, "fit_and_evaluate", synthetic_metrics)
    final = experiment.run_final_evaluation(saved_search, saved_search)
    output = saved_search / "figures"
    before = experiment.source_hashes(saved_search)
    subprocess.run([sys.executable, "-m", "src.plots", "--input-dir", str(saved_search),
                    "--output-dir", str(output)], cwd=ROOT, check=True, capture_output=True)
    for name in ("best_so_far.png", "auc_vs_runtime.png"):
        with Image.open(output / name) as image:
            assert image.width > 1000 and image.height > 600
            assert np.asarray(image).std() > 0  # Not an empty canvas.
    with (output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2
    for row in table:
        metrics = final["models"][row["model"]]
        for key in ("cv_auc_mean", "cv_auc_std", "test_roc_auc", "test_f1", "fit_duration_sec", "inference_duration_sec"):
            assert float(row[key]) == metrics[key]
    assert experiment.source_hashes(saved_search) == before
    _, _, rows = experiment.load_search_results(saved_search)
    numbers, scores, best_so_far, _ = plots.plot_series(rows)
    assert numbers.tolist() == list(range(1, 31))
    assert np.all(np.diff(best_so_far) >= 0)
    assert best_so_far[-1] == max(scores)
    assert all(best_so_far[i] == max(scores[:i + 1]) for i in range(len(scores)))
    assert (output / "comparison.md").exists()


def test_changed_source_cannot_reuse_final_results(monkeypatch, saved_search):
    monkeypatch.setattr(experiment, "fit_and_evaluate", synthetic_metrics)
    experiment.run_final_evaluation(saved_search, saved_search)
    with (saved_search / "trials.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")  # Same parsed data, different source fingerprint.
    with pytest.raises(ValueError, match="Source logs changed"):
        experiment.run_final_evaluation(saved_search, saved_search)


def test_provenance_survives_git_line_ending_conversion(saved_search):
    before = experiment.source_hashes(saved_search)
    for name in before:
        path = saved_search / name
        text = path.read_text(encoding="utf-8")
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert experiment.source_hashes(saved_search) == before
