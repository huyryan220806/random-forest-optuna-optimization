"""Run the S1 Random Forest baseline on training folds only.

Optuna search and final holdout evaluation belong to later sessions.
Run from the repository root: python -m src.experiment --seed 42
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


@dataclass(frozen=True)
class ExperimentConfig:
    """Single source of protocol settings; the main run uses seed 42."""

    seed: int = 42
    test_size: float = 0.2
    n_splits: int = 3
    n_jobs: int = 1

    def __post_init__(self):
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")


@dataclass
class DataSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_and_split(config: ExperimentConfig) -> DataSplit:
    """Load bundled data and split once, retaining row IDs for leakage checks."""
    dataset = load_breast_cancer()
    y = (dataset.target == 0).astype(int)  # Original malignant=0 -> positive=1.
    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=config.test_size,
        stratify=y,
        shuffle=True,
        random_state=config.seed,
    )
    return DataSplit(
        X_train=dataset.data[train_indices],
        X_test=dataset.data[test_indices],
        y_train=y[train_indices],
        y_test=y[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
    )


def make_cv_splits(X_train, y_train, config: ExperimentConfig):
    """Materialize folds once so later S2 trials can reuse these exact splits."""
    cv = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    return list(cv.split(X_train, y_train))


def make_baseline(config: ExperimentConfig) -> RandomForestClassifier:
    """Keep all library defaults except the seed and worker count."""
    return RandomForestClassifier(random_state=config.seed, n_jobs=config.n_jobs)


def evaluate_baseline(X_train, y_train, cv_splits, config: ExperimentConfig) -> dict:
    """Only training data enters CV; errors must not turn into valid scores."""
    model = make_baseline(config)
    started = perf_counter()
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_splits,
        scoring="roc_auc",
        n_jobs=config.n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    duration = perf_counter() - started
    # sklearn's test_score means CV validation folds, NOT our holdout test set.
    aucs = scores["test_score"]
    if not np.isfinite(aucs).all():
        raise ValueError("Baseline CV returned a non-finite ROC-AUC")
    return {
        "params": model.get_params(),
        "cv_auc_folds": aucs.tolist(),
        "cv_auc_mean": float(aucs.mean()),
        "cv_auc_std": float(aucs.std(ddof=0)),
        "duration_sec": duration,
        "fit_time_sec_per_fold": scores["fit_time"].tolist(),
        "score_time_sec_per_fold": scores["score_time"].tolist(),
    }


def class_counts(y) -> dict:
    return {"benign_0": int(np.sum(y == 0)), "malignant_1": int(np.sum(y == 1))}


def run_baseline(config: ExperimentConfig, output_dir: Path) -> dict:
    """Save CV results and enough protocol/environment metadata to rerun them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_and_split(config)
    cv_splits = make_cv_splits(data.X_train, data.y_train, config)
    metrics = evaluate_baseline(data.X_train, data.y_train, cv_splits, config)
    result = {
        "schema_version": 1,
        "stage": "S1-baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "seeds": {
            "split": config.seed,
            "cv": config.seed,
            "random_forest": config.seed,
        },
        "n_trials": 0,
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "loader": "sklearn.datasets.load_breast_cancer",
            "source_doi": "https://doi.org/10.24432/C5DW2B",
            "n_samples": len(data.train_indices) + len(data.test_indices),
            "n_features": data.X_train.shape[1],
            "target_transform": "y = (target == 0).astype(int)",
            "positive_class": "malignant",
            "labels": {"0": "benign", "1": "malignant"},
        },
        "split": {
            "method": "train_test_split",
            "test_size": config.test_size,
            "stratify": True,
            "shuffle": True,
            "random_state": config.seed,
            "n_train": len(data.y_train),
            "n_test": len(data.y_test),
            "train_class_counts": class_counts(data.y_train),
            "test_class_counts": class_counts(data.y_test),
        },
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.seed,
            "scoring": "roc_auc",
            "data_scope": "train_only",
            "std_ddof": 0,
            "n_jobs": config.n_jobs,
            "fold_sizes": [
                {"train": len(train), "validation": len(valid)}
                for train, valid in cv_splits
            ],
        },
        "holdout_evaluated": False,
        **metrics,
        "timing": {
            "clock": "time.perf_counter",
            "duration_sec_scope": "3-fold CV wall time: fitting, scoring and CV overhead",
            "run_duration_sec": perf_counter() - started,
            "run_duration_sec_scope": "data loading, split, folds and baseline CV; excludes JSON write",
        },
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    # Refuse NaN/Infinity in persisted scientific results.
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    (output_dir / "baseline.json").write_text(payload + "\n", encoding="utf-8")
    return result


def parse_args(argv=None):
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="S1 only: Random Forest baseline CV; holdout is not evaluated."
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run_baseline(ExperimentConfig(seed=args.seed), args.output_dir)
    print(
        f"Baseline CV ROC-AUC: {result['cv_auc_mean']:.6f} "
        f"+/- {result['cv_auc_std']:.6f} "
        f"({result['duration_sec']:.3f}s)"
    )
    print(f"Saved: {args.output_dir / 'baseline.json'}")
    print("S1 complete. Holdout not evaluated; Optuna trials: 0.")


if __name__ == "__main__":
    main()
