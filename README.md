# Random Forest + Optuna Optimization

**Tối ưu siêu tham số Random Forest bằng Optuna cho bài toán phân loại** —
bài tập lớn cá nhân học phần Nhập môn Tối ưu hóa.

Repository: [huyryan220806/random-forest-optuna-optimization](https://github.com/huyryan220806/random-forest-optuna-optimization).
Tiến độ hiện tại: **S1 — khởi tạo và baseline**. Kế hoạch và nhật ký nghiệm thu
ở [PROJECT_PLAN.md](PROJECT_PLAN.md). Optuna (S2), đánh giá holdout/biểu đồ
(S3), báo cáo/demo (S4) sẽ được triển khai khi được yêu cầu.

## Problem

Với cùng dữ liệu và các folds đánh giá, Optuna TPE có tìm được cấu hình
Random Forest có ROC-AUC CV cao hơn cấu hình mặc định không? Nếu có,
mức cải thiện đổi lấy bao nhiêu chi phí tìm kiếm, huấn luyện và suy luận?

MVP gồm một dataset, một họ mô hình và ba siêu tham số. Baseline mặc định
là mốc so sánh trung thực; kết quả tối ưu không bắt buộc phải thắng baseline.

## Optimization Formulation

Biến quyết định: `x = (n_estimators, max_depth, min_samples_leaf)`.

| Biến | Miền tìm kiếm đã chốt cho S2 | Baseline |
|---|---|---|
| `n_estimators` | 50, 100, 150, 200, 250 | 100 |
| `max_depth` | `None`, 4, 6, 8, 12, 16, 24 | `None` |
| `min_samples_leaf` | số nguyên 1–8 | 1 |

Hàm mục tiêu:

$$
\min_{x \in \Omega} f(x) = -\frac{1}{3}\sum_{k=1}^{3}\mathrm{ROC\!\!\!-AUC}_{k}(x)
$$

Tương đương tối đa hóa ROC-AUC trung bình qua 3 folds trên **train**.
Giữ nguyên các tham số khác theo mặc định của scikit-learn, chỉ đặt
`random_state=42`, `n_jobs=1`. Các mặc định baseline đều thuộc miền tìm kiếm.

Budget dự kiến S2: **30 trials**, `TPESampler(seed=42)`; smoke test 3 trials
lưu riêng. S1 có **0 trials**. Thời gian là chi phí cần đo/phân tích, không
phải ràng buộc cứng. Baseline không có chi phí tìm kiếm; sau này cần báo riêng
chi phí search của Optuna. Không chọn cấu hình, seed hay threshold bằng test score.

## Dataset và Method

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
được nạp bằng `sklearn.datasets.load_breast_cancer()`; 569 mẫu, 30 đặc trưng.
Nguồn, citation, giấy phép và cách lấy dữ liệu nằm trong [data/README.md](data/README.md).

1. Đảo nhãn: `y = (target == 0).astype(int)` → **malignant=1**, benign=0.
2. Split duy nhất bằng `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
3. Dùng `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` trên train.
4. Chạy `RandomForestClassifier(random_state=42, n_jobs=1)` bằng CV tuần tự,
   ROC-AUC từ xác suất lớp malignant. Lưu từng fold, mean và std (`ddof=0`).
5. Giữ holdout cho S3, sau khi cấu hình tối ưu được chốt từ CV ở S2.

`ExperimentConfig` trong `src/experiment.py` là nguồn cấu hình cho split/CV/RF.
Hàm CV chỉ nhận `X_train/y_train`; danh sách folds được tạo một lần trong mỗi run
để S2 có thể tái sử dụng. Không có preprocessing học từ toàn bộ dataset.

## Key Results — S1

Kết quả chạy thật ngày 25/10/2026, seed 42, từ
[results/baseline.json](results/baseline.json):

| Đại lượng | Kết quả |
|---|---:|
| Train / holdout | 455 / 114 mẫu |
| CV ROC-AUC fold 1 | 0.992059 |
| CV ROC-AUC fold 2 | 0.991874 |
| CV ROC-AUC fold 3 | 0.984680 |
| CV ROC-AUC mean ± std | **0.989538 ± 0.003436** |
| Tổng thời gian 3-fold CV | 0.691996 giây |
| Optuna trials | 0 |

Run chính và run kiểm chứng riêng cho tham số, split/CV và điểm số giống hệt nhau.
`python -m pytest -q`: **7 passed**; `pip check`: không có dependencies bị lỗi.
Bảng trên được điền từ JSON run chính; cần cập nhật từ log nếu chạy lại kết quả chính.
Chưa có kết quả tối ưu, điểm holdout hay kết luận so sánh.

## How to Run

Yêu cầu Python **3.10+**. Môi trường kiểm chứng S1 dùng **Python 3.12.10** trên
Windows; nên dùng Python 3.12 để đối chiếu kết quả. Các dependencies runtime
được ghim phiên bản trong `requirements.txt`, gồm scikit-learn 1.7.2.
Chỉ cài thư viện cần cho baseline và pytest; Optuna/pandas/matplotlib sẽ được
bổ sung ở phiên sử dụng chúng.

Tại thư mục gốc repo, dùng Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# S1: baseline chính, tạo results/baseline.json
.\.venv\Scripts\python.exe -m src.experiment --seed 42 --output-dir results

# Kiểm chứng riêng, không ghi đè kết quả chính
.\.venv\Scripts\python.exe -m src.experiment --seed 42 --output-dir results/smoke
.\.venv\Scripts\python.exe -m pytest -q
```

Nếu Python 3.10+ khác đang là mặc định, có thể dùng `py -m venv .venv`.
Các lệnh trên gọi thẳng Python của virtualenv nên không cần sửa ExecutionPolicy.
Sau khi activate, lệnh baseline tương đương là:

```powershell
python -m src.experiment --seed 42 --output-dir results
```

S1 chưa hỗ trợ `--trials` hay `src.plots`; các lệnh này trong kế hoạch là mục tiêu
của S2/S3. Chạy lại cùng thư mục sẽ thay `baseline.json` ở thư mục đó;
dùng `results/smoke` hoặc `results/repro` để kiểm chứng riêng.

JSON ghi đầy đủ tham số RF, seed, quy tắc nhãn, split/CV, điểm từng fold,
mean/std, phiên bản và môi trường. `duration_sec` là wall time cho toàn bộ
3-fold CV (fit + scoring + overhead), dùng `time.perf_counter()`.
`fit_time_sec_per_fold`/`score_time_sec_per_fold` đo riêng từng fold;
đây chưa phải thời gian fit cuối hoặc suy luận holdout. Timing phụ thuộc máy/tải
hệ thống và không yêu cầu giống hệt khi chạy lại. JSON không chấp nhận NaN/Infinity.

## Repository Structure

```text
.
├── PROJECT_PLAN.md          # Đặc tả và tiến độ làm việc
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── src/
│   ├── __init__.py
│   └── experiment.py        # S1: load/split, CV, baseline JSON
├── data/README.md           # Nguồn và quy tắc nhãn
├── results/baseline.json    # Kết quả baseline chính
├── tests/test_pipeline.py   # Nhãn, leakage, mặc định RF, CLI/tái lập
├── figures/                # S3
├── report/README.md         # S4
└── plans/                  # Tài liệu đầu vào; PDF giữ cục bộ
```

## Limitations

Hiện chỉ có baseline với một seed và một dataset nhỏ; std giữa các folds
không phải độ bất định qua nhiều lần chạy độc lập. Chưa đủ bằng chứng kết luận
Optuna cải thiện chất lượng. Dataset phục vụ benchmark học thuật, không chứng
minh khả năng chẩn đoán lâm sàng. Không triển khai ứng dụng web hoặc mô hình khác.

## AI assistance

Codex hỗ trợ tạo cấu trúc, code baseline, tài liệu và chạy các kiểm tra tự động
ghi trong nhật ký S1. Đây là kiểm chứng do công cụ thực hiện; phần người học tự
chạy, đọc hiểu code và giải thích kết quả cần được người học bổ sung sau khi làm.
