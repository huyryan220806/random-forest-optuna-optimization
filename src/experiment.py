"""Run baseline/search on training folds; evaluate frozen results separately.

Run: python -m src.experiment --trials 30 --seed 42 --output-dir results
Then: python -m src.experiment --evaluate-only --input-dir results
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
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
    if any((output_dir / name).exists() for name in ("trials.csv", "best_config.json", "final_metrics.json")):
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
    if (output_dir / "final_metrics.json").exists():
        raise ValueError("Final evaluation exists; use a separate directory for a new search.")
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


def load_search_results(input_dir: Path):
    """Validate a completed study before reporting or touching the holdout."""
    input_dir = Path(input_dir)
    baseline = json.loads((input_dir / "baseline.json").read_text(encoding="utf-8"))
    best = json.loads((input_dir / "best_config.json").read_text(encoding="utf-8"))
    with (input_dir / "trials.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (best["status"] != "complete" or not rows or
            len(rows) != best["n_trials"] or best["n_complete"] != len(rows) or best["n_failed"] != 0):
        raise ValueError("A complete, consistent search log is required.")
    for key in ("seed", "n_trials", "dataset", "split", "cv", "seeds", "versions"):
        if baseline[key] != best[key]:
            raise ValueError(f"Baseline/search metadata mismatch: {key}")
    baseline_scores = np.asarray(baseline["cv_auc_folds"], dtype=float)
    if (baseline_scores.shape != (best["cv"]["n_splits"],) or not np.isfinite(baseline_scores).all()
            or not np.isclose(baseline["cv_auc_mean"], baseline_scores.mean(), rtol=0, atol=1e-12)
            or not np.isclose(baseline["cv_auc_std"], baseline_scores.std(ddof=0), rtol=0, atol=1e-12)
            or best["baseline_cv_auc_mean"] != baseline["cv_auc_mean"]
            or best["search_space"] != SEARCH_SPACE):
        raise ValueError("Baseline metrics or search space do not match the frozen study.")
    for number, row in enumerate(rows):
        if row["state"] != "COMPLETE" or int(row["trial_number"]) != number or int(row["seed"]) != best["seed"]:
            raise ValueError("Invalid trial state, order or seed.")
        scores = np.asarray(json.loads(row["cv_auc_folds"]), dtype=float)
        if (scores.shape != (best["cv"]["n_splits"],) or not np.isfinite(scores).all()
                or not ((0 <= scores) & (scores <= 1)).all()
                or not np.isclose(float(row["cv_auc_mean"]), scores.mean(), rtol=0, atol=1e-12)
                or not np.isclose(float(row["cv_auc_std"]), scores.std(ddof=0), rtol=0, atol=1e-12)):
            raise ValueError("Trial scores do not match their fold metrics.")
        for key in ("duration_sec", "cv_duration_sec"):
            if not np.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise ValueError("Invalid trial duration.")
    winner = max(rows, key=lambda row: float(row["cv_auc_mean"]))
    params = {
        "n_estimators": int(winner["n_estimators"]),
        "max_depth": None if winner["max_depth"] == "None" else int(winner["max_depth"]),
        "min_samples_leaf": int(winner["min_samples_leaf"]),
    }
    if (best["best_trial_number"] != int(winner["trial_number"]) or best["best_params"] != params
            or best["best_cv_auc_mean"] != float(winner["cv_auc_mean"])
            or best["best_cv_auc_std"] != float(winner["cv_auc_std"])
            or best["best_cv_auc_folds"] != json.loads(winner["cv_auc_folds"])):
        raise ValueError("Best config does not match the CV winner in trials.csv.")
    expected = make_baseline(ExperimentConfig(seed=best["seed"])).get_params()
    if baseline["params"] != expected or best["model_params"] != {**expected, **params}:
        raise ValueError("Saved model parameters differ from the declared baseline/search.")
    return baseline, best, rows


def source_hashes(input_dir: Path):
    # Normalize text newlines so a Git checkout on Windows/Linux keeps provenance.
    return {
        name: hashlib.sha256((input_dir / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for name in ("baseline.json", "best_config.json", "trials.csv")
    }


def fit_and_evaluate(params: dict, data: DataSplit) -> dict:
    """Fit once on full train; obtain holdout probabilities once for all metrics."""
    model = RandomForestClassifier(**params)
    started = perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_sec = perf_counter() - started
    started = perf_counter()
    probabilities = model.predict_proba(data.X_test)
    inference_sec = perf_counter() - started
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    positive_probabilities = probabilities[:, positive_column]
    # Matches sklearn predict: argmax selects class 0 when probabilities tie.
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    return {
        "params": params,
        "test_roc_auc": float(roc_auc_score(data.y_test, positive_probabilities)),
        "test_f1": float(f1_score(data.y_test, predictions, pos_label=1, zero_division=0)),
        "test_accuracy": float(accuracy_score(data.y_test, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(data.y_test, predictions)),
        "confusion_matrix": confusion_matrix(data.y_test, predictions, labels=[0, 1]).tolist(),
        "fit_duration_sec": fit_sec,
        "inference_duration_sec": inference_sec,
        "inference_sec_per_sample": inference_sec / len(data.y_test),
        "prediction_calls": 1,
    }


def run_final_evaluation(input_dir: Path, output_dir: Path) -> dict:
    """Evaluate the frozen main study, checkpointing after each model for resume."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    baseline, best, _ = load_search_results(input_dir)
    if best["n_trials"] != DEFAULT_N_TRIALS:
        raise ValueError("Holdout evaluation requires the completed 30-trial main study, not smoke.")
    config = ExperimentConfig(seed=best["seed"])
    fingerprints = source_hashes(input_dir)
    path = output_dir / "final_metrics.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if existing and existing["source_sha256"] != fingerprints:
        raise ValueError("Source logs changed after evaluation; refusing to mix results.")
    if existing and existing["status"] == "complete":
        return existing  # Idempotent: no second fit or holdout prediction.
    data = load_and_split(config)  # Reconstruct exactly the saved S2 split.
    folds = make_cv_splits(data.X_train, data.y_train, config)
    metadata = build_metadata(config, data, folds)
    metadata["seeds"]["optuna"] = config.seed
    for key in ("dataset", "split", "cv", "seeds", "versions"):
        if metadata[key] != best[key]:
            raise ValueError(f"Current evaluation differs from frozen S2 metadata: {key}")
    selected = "optimized" if best["best_cv_auc_mean"] > baseline["cv_auc_mean"] else "baseline"
    result = existing or {
        **metadata,
        "stage": "S3-holdout",
        "status": "in_progress",
        "n_trials": best["n_trials"],
        "source_sha256": fingerprints,
        "source_hash_scope": "UTF-8 text with CRLF normalized to LF",
        "best_trial_number": best["best_trial_number"],
        "selection": {
            "model": selected,
            "basis": "Higher train CV ROC-AUC; ties keep baseline. Fixed before holdout evaluation.",
            "uses_holdout": False,
        },
        "threshold": 0.5,
        "threshold_rule": "malignant if probability > 0.5; ties select benign (sklearn argmax)",
        "confusion_matrix_order": "rows=true, columns=predicted; labels=[0 benign, 1 malignant]",
        "search_duration_sec": best["search_duration_sec"],
        "timing": {
            "clock": "time.perf_counter",
            "fit_scope": "one fit on all training samples",
            "inference_scope": "one predict_proba batch over all holdout samples, excluding metrics",
            "repetitions": 1,
            "per_sample_scope": "batch time divided by sample count, not isolated request latency",
        },
        "models": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, result)  # Selection is persisted before seeing test predictions.
    for name, params, cv_mean, cv_std in (
        ("baseline", baseline["params"], baseline["cv_auc_mean"], baseline["cv_auc_std"]),
        ("optimized", best["model_params"], best["best_cv_auc_mean"], best["best_cv_auc_std"]),
    ):
        if name not in result["models"]:
            result["models"][name] = {
                **fit_and_evaluate(params, data),
                "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
            }
            result["holdout_evaluated"] = True
            write_json(path, result)
    result["status"] = "complete"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, result)
    return result


def parse_args(argv=None):
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Baseline/search CV, or --evaluate-only for frozen S2 holdout evaluation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS)
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, help="Saved S2 logs; only with --evaluate-only")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.evaluate_only:
        if args.seed != defaults.seed:
            parser.error("Evaluation uses the seed saved in S2 logs; do not override --seed.")
        args.input_dir = args.input_dir or Path("results")
        args.output_dir = args.output_dir or args.input_dir
    elif args.input_dir is not None:
        parser.error("--input-dir requires --evaluate-only")
    elif args.output_dir is None:
        args.output_dir = Path(
            "results/baseline-only" if args.baseline_only else
            "results" if args.trials == DEFAULT_N_TRIALS else "results/smoke"
        )
    if not args.baseline_only and not args.evaluate_only:
        try:
            validate_search_output(ExperimentConfig(seed=args.seed), args.trials, args.output_dir)
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    config = ExperimentConfig(seed=args.seed)
    if args.evaluate_only:
        result = run_final_evaluation(args.input_dir, args.output_dir)
        for name, metrics in result["models"].items():
            print(f"{name}: test AUC={metrics['test_roc_auc']:.6f}, F1={metrics['test_f1']:.6f}")
        print(f"Selection fixed by CV: {result['selection']['model']}; status: {result['status']}")
    elif args.baseline_only:
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
