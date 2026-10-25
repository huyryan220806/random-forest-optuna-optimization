"""Run baseline and Optuna search on training folds only (S1/S2).

Final holdout evaluation belongs to S3.
Run: python -m src.experiment --trials 30 --seed 42 --output-dir results
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


DEFAULT_N_TRIALS = 30
TPE_STARTUP_TRIALS = 10
SEARCH_SPACE = {
    "n_estimators": {"low": 50, "high": 250, "step": 50},
    "max_depth": [None, 4, 6, 8, 12, 16, 24],
    "min_samples_leaf": {"low": 1, "high": 8},
}
TRIAL_FIELDS = [
    "trial_number", "n_estimators", "max_depth", "min_samples_leaf",
    "cv_auc_mean", "cv_auc_std", "duration_sec", "state", "seed",
    "cv_duration_sec", "cv_auc_folds", "error",
]


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
    return evaluate_model(make_baseline(config), X_train, y_train, cv_splits, config)


def evaluate_model(model, X_train, y_train, cv_splits, config: ExperimentConfig) -> dict:
    """One evaluation path for baseline/trials; no holdout arguments."""
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
        raise ValueError("CV returned a non-finite ROC-AUC")
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


def build_metadata(config: ExperimentConfig, data: DataSplit, cv_splits) -> dict:
    """Common provenance for baseline and search, without test evaluation."""
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "seeds": {
            "split": config.seed,
            "cv": config.seed,
            "random_forest": config.seed,
        },
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
        "timing": {
            "clock": "time.perf_counter",
            "duration_sec_scope": "3-fold CV wall time: fitting, scoring and CV overhead",
        },
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "optuna")
            },
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }


def write_json(path: Path, result: dict):
    # Refuse NaN/Infinity in persisted scientific results.
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def save_baseline(config, output_dir, data, cv_splits, metadata, n_trials=0):
    result = {
        **metadata,
        "stage": "S2-baseline" if n_trials else "S1-baseline",
        "n_trials": n_trials,
        "baseline_search_trials": 0,
        **evaluate_baseline(data.X_train, data.y_train, cv_splits, config),
    }
    write_json(output_dir / "baseline.json", result)
    return result


def run_baseline(config: ExperimentConfig, output_dir: Path) -> dict:
    """Retain a baseline-only command for S1 reproduction."""
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("trials.csv", "best_config.json")):
        raise ValueError("Use a separate output directory for a baseline-only run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_split(config)
    cv_splits = make_cv_splits(data.X_train, data.y_train, config)
    metadata = build_metadata(config, data, cv_splits)
    return save_baseline(config, output_dir, data, cv_splits, metadata)


def make_objective(X_train, y_train, cv_splits, config: ExperimentConfig):
    """Capture only train arrays, fixed folds and settings, never DataSplit/test."""
    def objective(trial):
        started = perf_counter()
        try:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", **SEARCH_SPACE["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth", SEARCH_SPACE["max_depth"]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", **SEARCH_SPACE["min_samples_leaf"]),
            }
            model = RandomForestClassifier(
                **params, random_state=config.seed, n_jobs=config.n_jobs
            )
            metrics = evaluate_model(model, X_train, y_train, cv_splits, config)
            for key in ("cv_auc_folds", "cv_auc_std"):
                trial.set_user_attr(key, metrics[key])
            trial.set_user_attr("cv_duration_sec", metrics["duration_sec"])
            return metrics["cv_auc_mean"]
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            raise
        finally:
            trial.set_user_attr("duration_sec", perf_counter() - started)

    return objective


def write_trials(study, path: Path, seed: int):
    """Persist the journal after each trial, including failed/partial runs."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for trial in study.trials:
            valid = trial.state == optuna.trial.TrialState.COMPLETE
            params = trial.params.copy()
            if "max_depth" in params and params["max_depth"] is None:
                params["max_depth"] = "None"
            writer.writerow({
                "trial_number": trial.number,
                **params,
                "cv_auc_mean": trial.value if valid else None,
                "cv_auc_std": trial.user_attrs.get("cv_auc_std") if valid else None,
                "duration_sec": trial.user_attrs.get("duration_sec"),
                "state": trial.state.name,
                "seed": seed,
                "cv_duration_sec": trial.user_attrs.get("cv_duration_sec"),
                "cv_auc_folds": json.dumps(trial.user_attrs["cv_auc_folds"]) if valid else "",
                "error": trial.user_attrs.get("error", ""),
            })
    temporary.replace(path)


def validate_search_output(config, n_trials, output_dir):
    if not 1 <= n_trials <= DEFAULT_N_TRIALS:
        raise ValueError(f"--trials must be between 1 and {DEFAULT_N_TRIALS}")
    main_dir = Path(__file__).resolve().parents[1] / "results"
    if output_dir.resolve() == main_dir and (
        n_trials != DEFAULT_N_TRIALS or config.seed != ExperimentConfig().seed
    ):
        raise ValueError("Main results require 30 trials and seed 42; use results/smoke or another directory.")


def run_experiment(config: ExperimentConfig, n_trials: int, output_dir: Path) -> dict:
    """One split and one fold list shared by baseline and the entire study."""
    output_dir = Path(output_dir)
    validate_search_output(config, n_trials, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_split(config)
    cv_splits = make_cv_splits(data.X_train, data.y_train, config)
    metadata = build_metadata(config, data, cv_splits)
    metadata["seeds"]["optuna"] = config.seed
    baseline = save_baseline(config, output_dir, data, cv_splits, metadata, n_trials)
    sampler = optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=TPE_STARTUP_TRIALS)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=optuna.pruners.NopPruner()
    )
    objective = make_objective(data.X_train, data.y_train, cv_splits, config)
    started = perf_counter()
    error_message = None
    try:
        study.optimize(
            objective, n_trials=n_trials, n_jobs=1,
            callbacks=[lambda study, trial: write_trials(study, output_dir / "trials.csv", config.seed)],
        )
    except BaseException as error:
        # Preserve diagnostics on failure/interruption, then return a nonzero CLI exit.
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        search_duration = perf_counter() - started
        write_trials(study, output_dir / "trials.csv", config.seed)
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        best = study.best_trial if completed else None
        result = {
            **metadata,
            "stage": "S2-search",
            "status": "complete" if error_message is None and len(completed) == n_trials else "incomplete",
            "error": error_message,
            "n_trials": n_trials,
            "n_trials_recorded": len(study.trials),
            "n_complete": len(completed),
            "n_failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
            "search_space": SEARCH_SPACE,
            "sampler": {"name": "TPESampler", "seed": config.seed, "n_startup_trials": TPE_STARTUP_TRIALS},
            "direction": "maximize",
            "pruner": "NopPruner",
            "study_n_jobs": 1,
            "best_trial_number": best.number if best else None,
            "best_params": best.params if best else None,
            "model_params": {**make_baseline(config).get_params(), **best.params} if best else None,
            "best_cv_auc_mean": best.value if best else None,
            "best_cv_auc_std": best.user_attrs["cv_auc_std"] if best else None,
            "best_cv_auc_folds": best.user_attrs["cv_auc_folds"] if best else None,
            "best_trial_duration_sec": best.user_attrs["duration_sec"] if best else None,
            "baseline_cv_auc_mean": baseline["cv_auc_mean"],
            "cv_auc_improvement": best.value - baseline["cv_auc_mean"] if best else None,
            "search_duration_sec": search_duration,
            "timing": {
                **metadata["timing"],
                "duration_sec_scope": "trial objective wall time: sampling, model creation, CV and metric logging",
                "cv_duration_sec_scope": "3-fold CV wall time, measured identically to baseline duration_sec",
                "search_duration_sec_scope": "study.optimize wall time, including trial CSV checkpoints; excludes baseline",
            },
        }
        write_json(output_dir / "best_config.json", result)
    return result


def parse_args(argv=None):
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="S1/S2: baseline and Optuna CV search; holdout is not evaluated."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS)
    mode.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.output_dir is None:
        args.output_dir = Path(
            "results/baseline-only" if args.baseline_only else
            "results" if args.trials == DEFAULT_N_TRIALS else "results/smoke"
        )
    if not args.baseline_only:
        try:
            validate_search_output(ExperimentConfig(seed=args.seed), args.trials, args.output_dir)
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    config = ExperimentConfig(seed=args.seed)
    if args.baseline_only:
        result = run_baseline(config, args.output_dir)
        print(f"Baseline CV ROC-AUC: {result['cv_auc_mean']:.6f} +/- {result['cv_auc_std']:.6f}")
        print("Holdout not evaluated; Optuna trials: 0.")
    else:
        result = run_experiment(config, args.trials, args.output_dir)
        print(f"Baseline CV ROC-AUC: {result['baseline_cv_auc_mean']:.6f}")
        print(f"Best CV ROC-AUC: {result['best_cv_auc_mean']:.6f} +/- {result['best_cv_auc_std']:.6f}")
        print(f"Best params: {result['best_params']}")
        print(f"Search time: {result['search_duration_sec']:.3f}s; completed trials: {result['n_complete']}")
        print("S2 complete. Holdout not evaluated.")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
