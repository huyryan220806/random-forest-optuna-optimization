"""Regenerate S3 figures/tables from validated logs, without fitting models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import load_search_results, source_hashes


def plot_series(rows):
    trials = np.array([int(row["trial_number"]) + 1 for row in rows])
    aucs = np.array([float(row["cv_auc_mean"]) for row in rows])
    durations = np.array([float(row["duration_sec"]) for row in rows])
    return trials, aucs, np.maximum.accumulate(aucs), durations


def write_comparison(final: dict, output_dir: Path):
    fields = [
        "model", "n_estimators", "max_depth", "min_samples_leaf", "cv_auc_mean",
        "cv_auc_std", "test_roc_auc", "test_f1", "test_accuracy", "test_balanced_accuracy",
        "fit_duration_sec", "inference_duration_sec", "search_duration_sec",
    ]
    rows = []
    for name in ("baseline", "optimized"):
        metrics = final["models"][name]
        rows.append({
            "model": name,
            **{key: "None" if metrics["params"][key] is None else metrics["params"][key]
               for key in ("n_estimators", "max_depth", "min_samples_leaf")},
            **{key: metrics[key] for key in fields[4:-1]},
            "search_duration_sec": 0.0 if name == "baseline" else final["search_duration_sec"],
        })
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Model | CV AUC mean ± std | Test AUC | F1 | Accuracy | Balanced accuracy | Fit (s) | Predict batch (ms) | Search (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['cv_auc_mean']:.6f} ± {row['cv_auc_std']:.6f} "
            f"| {row['test_roc_auc']:.6f} | {row['test_f1']:.6f} | {row['test_accuracy']:.6f} "
            f"| {row['test_balanced_accuracy']:.6f} | {row['fit_duration_sec']:.6f} "
            f"| {row['inference_duration_sec'] * 1000:.3f} | {row['search_duration_sec']:.6f} |"
        )
    lines.extend([
        "", "Source: final_metrics.json. Positive class: malignant=1.",
        "Timing: one fit and one predict_proba batch; search excludes baseline.",
        "Single measurements depend on machine load; they are not latency guarantees.",
    ])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots(input_dir: Path, output_dir: Path):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    baseline, best, rows = load_search_results(input_dir)
    final_path = input_dir / "final_metrics.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
    if final and (final["status"] != "complete" or final["source_sha256"] != source_hashes(input_dir)):
        raise ValueError("Final evaluation is incomplete or belongs to different search logs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    trials, aucs, best_so_far, durations = plot_series(rows)
    best_index = best["best_trial_number"]
    baseline_auc = baseline["cv_auc_mean"]
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
             "axes.spines.right": False, "axes.titleweight": "bold"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
        ax.step(trials, best_so_far, where="post", color="#147D92", linewidth=2.2,
                label="Best-so-far Optuna CV AUC")
        ax.axhline(baseline_auc, color="#B75B20", linestyle="--", linewidth=1.6,
                   label=f"Baseline CV AUC ({baseline_auc:.6f})")
        ax.scatter(trials[best_index], aucs[best_index], marker="*", s=180, color="#263D63",
                   zorder=4, label=f"First best: trial #{best_index} (zero-based)")
        ax.set(xlabel="Number of completed trials", ylabel="Mean 3-fold CV ROC-AUC",
               title=f"Random Forest tuning: convergence | seed {best['seed']}",
               xlim=(0.5, len(rows) + 0.5))
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=True)
        fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
        fig.text(0.5, 0.015, "Training CV only. Holdout scores do not enter the search.",
                 ha="center", fontsize=9, color="#555555")
        fig.savefig(output_dir / "best_so_far.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
        ax.scatter(durations, aucs, s=45, color="#147D92", alpha=0.7, label="Completed trials")
        ax.scatter(durations[best_index], aucs[best_index], s=220, marker="*", color="#263D63",
                   edgecolor="white", linewidth=0.6, zorder=4, label=f"Selected trial #{best_index}")
        ax.axhline(baseline_auc, color="#B75B20", linestyle="--", linewidth=1.6,
                   label="Baseline CV AUC (score reference)")
        ax.set(xlabel="Trial objective duration (seconds)", ylabel="Mean 3-fold CV ROC-AUC",
               title=f"Random Forest tuning: score vs. search cost | seed {best['seed']}")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=True)
        fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
        fig.text(0.5, 0.015, "Trial time includes sampling + CV. Baseline line compares scores, not runtime.",
                 ha="center", fontsize=9, color="#555555")
        fig.savefig(output_dir / "auc_vs_runtime.png", dpi=180)
        plt.close(fig)
    if final:
        write_comparison(final, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    generate_plots(args.input_dir, args.output_dir)
    print(f"Figures saved to {args.output_dir}; comparison exported if final metrics exist.")


if __name__ == "__main__":
    main()
