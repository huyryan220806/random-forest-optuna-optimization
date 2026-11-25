"""Verify the three runs and regenerate the comparison plot and table."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.repeat_experiment import SEEDS
from src.experiment import load_search_results, source_hashes


def read_run(batch, seed):
    folder = batch / f"seed-{seed}"
    baseline, best, trials = load_search_results(folder)
    final = json.loads((folder / "final_metrics.json").read_text(encoding="utf-8"))
    if (len(trials) != 30 or best["seed"] != seed or final["seed"] != seed
            or final["status"] != "complete" or final["selection"]["uses_holdout"]
            or final["source_sha256"] != source_hashes(folder)):
        raise ValueError(f"Inconsistent evidence in {folder}")
    models = final["models"]
    baseline_test = models["baseline"]["test_roc_auc"]
    optimized_test = models["optimized"]["test_roc_auc"]
    return {
        "seed": seed, "search_created_utc": best["created_at_utc"],
        "holdout_completed_utc": final["completed_at_utc"], "complete_trials": len(trials),
        "best_trial": best["best_trial_number"], **best["best_params"],
        "baseline_cv_auc": baseline["cv_auc_mean"], "optimized_cv_auc": best["best_cv_auc_mean"],
        "delta_cv_auc": best["cv_auc_improvement"], "baseline_test_auc": baseline_test,
        "optimized_test_auc": optimized_test, "delta_test_auc": optimized_test - baseline_test,
        "baseline_test_f1": models["baseline"]["test_f1"],
        "optimized_test_f1": models["optimized"]["test_f1"],
        "search_sec": best["search_duration_sec"],
    }


def plot_differences(rows, output):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for offset, field, label, color, marker in [
        (-0.12, "delta_cv_auc", "Cross-validation", "#1A526C", "o"),
        (0.12, "delta_test_auc", "Holdout", "#B65418", "D"),
    ]:
        ax.scatter([row[field] for row in rows], [i + offset for i in range(len(rows))],
                   color=color, s=65, marker=marker, label=label, zorder=3)
    ax.set_yticks(range(len(rows)), [f"Seed {row['seed']}" for row in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Chênh lệch ROC-AUC (Optuna trừ baseline)")
    ax.set_title("Mức thay đổi ROC-AUC qua ba lần chạy", loc="left", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.spines[["right", "top", "left"]].set_visible(False)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    batch = parser.parse_args().directory
    rows = [read_run(batch, seed) for seed in SEEDS]
    with (batch / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_differences(rows, batch / "auc-by-seed.png")
    for field in ("delta_cv_auc", "delta_test_auc"):
        values = [row[field] for row in rows]
        print(f"{field}: mean={mean(values):+.6f}, sample SD={stdev(values):.6f}")


if __name__ == "__main__":
    main()
