# Kiểm chứng tái lập S4

Ngày **30/10/2026**, do Codex thực hiện; không thay cho sinh viên tự kiểm chứng.
Không mở rộng search space, seed, dataset hoặc phương pháp.

## Phạm vi và kết quả

Clone mới từ GitHub tại commit `6ac99d74d6f57ae7d771bcbd8d2a3dca6dab09cc`,
tạo `.venv` mới bằng Python 3.12.10, cài `requirements.txt`. Commit này chứa code
và kết quả S3; S4 bổ sung tài liệu, không sửa `src/`, tests hoặc requirements.

| Kiểm tra thực tế | Kết quả |
|---|---|
| `pip check` workspace và clone mới | Không có dependency lỗi |
| Tests workspace | 24 passed, 14 warnings, 37.84 giây |
| Tests clone/venv mới | 24 passed, 14 warnings, 59.84 giây |
| Smoke 3 trials trên clone | Thành công, lưu riêng |
| Tái lập 30 trials seed 42 | 30 COMPLETE, 0 FAIL |
| Chuỗi 30 cấu hình, state, seed, từng fold, CV mean/std | Khớp chính xác các trường CSV không phải thời gian |
| Baseline/best params, folds/CV và protocol | Khớp; tolerance số thực 1e-12 |
| Holdout AUC/F1/accuracy/balanced accuracy/confusion matrix | Khớp kết quả chính; tolerance 1e-12 |
| Đọc lại `--evaluate-only` complete | SHA-256 byte của final JSON không đổi |
| Nạp log chính sau Git checkout Windows | Thành công, fingerprint chuẩn hóa CRLF/LF hợp lệ |
| Sinh hình/bảng trên clone | Đủ 2 PNG, comparison.csv và comparison.md |

14 warnings là deprecation giữa matplotlib 3.10.6 và pyparsing, không phải test
thất bại. Dependencies trực tiếp đã ghim; đây không phải lockfile toàn bộ dependency bắc cầu.

Không so timing, timestamp hoặc fingerprint giữa hai run độc lập. PNG trade-off
cũng có thể khác vì tọa độ thời gian được đo lại. Kết quả gốc trong `results/` và
`figures/` giữ nguyên, không thay bằng số của lần kiểm chứng.

## Lệnh chạy từ clone mới

```powershell
git clone https://github.com/huyryan220806/random-forest-optuna-optimization.git
cd random-forest-optuna-optimization
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.experiment --trials 3 --seed 42 --output-dir results/smoke

# Dùng tên thư mục mới nếu "kiem-chung-1" đã được đánh giá holdout.
.\.venv\Scripts\python.exe -m src.experiment --trials 30 --seed 42 --output-dir results/repro/kiem-chung-1
.\.venv\Scripts\python.exe -m src.experiment --evaluate-only --input-dir results/repro/kiem-chung-1
.\.venv\Scripts\python.exe -m src.plots --input-dir results/repro/kiem-chung-1 --output-dir figures/repro/kiem-chung-1

# Đọc main đã đóng băng, không fit/predict thêm.
.\.venv\Scripts\python.exe -m src.experiment --evaluate-only --input-dir results
```

Kỳ vọng cùng môi trường:

- Baseline CV AUC **0.98953798971112**.
- Best trial **#10**, **50 cây / max_depth 16 / min_samples_leaf 1**,
  CV AUC **0.9899337158686189**.
- Test AUC baseline/optimized **0.9928902116402116 / 0.9940476190476191**.
- F1 cả hai **0.9629629629629629**, confusion matrix **[[72, 0], [3, 39]]**.

## Bằng chứng và giới hạn

Log cục bộ ở `.build/s4/` (gitignored): `install.log`, `clone-tests.log`,
`clone-smoke.log`, `clone-search.log`, `clone-holdout.log`, `clone-plots.log`,
`repro-verification.json` và bản clone. Người clone repo có thể chạy lại theo lệnh trên.

Đã kiểm chứng Windows/Python 3.12.10; chưa xác minh độc lập Linux/macOS hoặc Python
khác. Tái lập một seed không chứng minh ổn định thống kê. Holdout trên clone chỉ
đối chiếu cấu hình đã cố định, không dùng để chọn lại tham số/seed/ngưỡng.
