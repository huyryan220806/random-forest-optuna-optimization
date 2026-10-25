| Model | CV AUC mean ± std | Test AUC | F1 | Accuracy | Balanced accuracy | Fit (s) | Predict batch (ms) | Search (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.989538 ± 0.003436 | 0.992890 | 0.962963 | 0.973684 | 0.964286 | 0.298146 | 9.779 | 0.000000 |
| optimized | 0.989934 ± 0.004250 | 0.994048 | 0.962963 | 0.973684 | 0.964286 | 0.147775 | 4.134 | 24.978321 |

Source: final_metrics.json. Positive class: malignant=1.
Timing: one fit and one predict_proba batch; search excludes baseline.
Single measurements depend on machine load; they are not latency guarantees.
