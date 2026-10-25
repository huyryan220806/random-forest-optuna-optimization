# PROJECT PLAN — Random Forest + Optuna

> **Tài liệu làm việc cho Codex** | Học phần: **Nhập môn Tối ưu hóa** | Bài tập lớn **cá nhân** | Ngôn ngữ thực hiện: **Python**
>
> **Trạng thái (25/10/2026):** S1/S2 đã hoàn tất; **S3 đang triển khai** theo yêu cầu người dùng. Tiến độ và hướng dẫn tiếp tục nếu gián đoạn được lưu tại mục 12. Cấu hình S2 đã chốt, chưa đánh giá holdout tại thời điểm bắt đầu S3.
>
> **Quỹ thời gian thực tế:** 2–3 tuần tập trung, **1–2 ngày làm việc mỗi tuần** (dự kiến 4 ngày làm việc cốt lõi, tối đa 6 ngày nếu có thể). Hạn chế phát sinh việc ngoài phạm vi.

## 0. Mục tiêu của tài liệu này

Tài liệu này là **đặc tả duy nhất để giao việc cho Codex**. Codex cần đọc toàn bộ trước khi tạo mã nguồn, làm đúng phạm vi và cập nhật tiến độ sau mỗi phiên làm việc. **Không tự ý mở rộng sang ứng dụng web, mô hình khác hay nhiều dataset.**

**Tên đề tài tiếng Việt:** Tối ưu siêu tham số Random Forest bằng Optuna cho bài toán phân loại.

**Tên đề tài tiếng Anh:** *Hyperparameter Optimization of Random Forest Using Optuna for Classification*.

**Tên GitHub repository đề xuất:** `random-forest-optuna-optimization`.

**Câu hỏi nghiên cứu:** Với cùng dữ liệu và quy trình đánh giá, Optuna (TPE) có tìm được cấu hình Random Forest cho **ROC-AUC cross-validation tốt hơn** cấu hình mặc định không? Mức cải thiện (nếu có) đổi lấy bao nhiêu chi phí tìm kiếm, huấn luyện và suy luận?

**Điều kiện thành công:** Một người khác có thể clone repo, cài dependencies, chạy lại script, tạo ra log/bảng/biểu đồ và hiểu được **biến quyết định → hàm mục tiêu → budget/ràng buộc → baseline → quá trình tối ưu → kết quả → quyết định cuối**. Không cần có kết quả Optuna *luôn* thắng baseline; báo cáo trung thực mới là yêu cầu.

---

## 1. Phạm vi được CHỐT

### Bắt buộc (MVP)

1. **Một dataset:** Breast Cancer Wisconsin Diagnostic từ `sklearn.datasets.load_breast_cancer()` (569 bản ghi, 30 đặc trưng). Nêu rõ nguồn gốc UCI/scikit-learn và ý nghĩa nhãn.
2. **Một họ mô hình:** `sklearn.ensemble.RandomForestClassifier`.
3. **Baseline:** Random Forest với siêu tham số mặc định của thư viện; đặt `random_state` và `n_jobs` thống nhất để tái lập/đo thời gian.
4. **Một phương pháp tối ưu:** `optuna.samplers.TPESampler(seed=...)`, tối ưu qua 3-fold stratified CV trên tập train.
5. **Ba siêu tham số được thay đổi:** `n_estimators`, `max_depth`, `min_samples_leaf`.
6. **Ngân sách:** mặc định **30 trials** Optuna (phù hợp khuyến nghị HPO 30–50 trials/phương pháp trong hướng dẫn); chế độ smoke test dùng 3 trials và **lưu kết quả riêng**.
7. **Metric tối ưu:** trung bình **ROC-AUC trên 3 folds train/CV**; test set hoàn toàn không tham gia tuning.
8. Lưu `CSV/JSON` cho từng trial, kết quả baseline, cấu hình tốt nhất, thời gian và seed.
9. Có **ít nhất 2 hình**: (i) best-so-far ROC-AUC theo trial; (ii) ROC-AUC theo thời gian trial hoặc chi phí huấn luyện.
10. Tạo `README.md`, báo cáo **8–12 trang** (Word/PDF theo đề bài), slide **5–8 trang** hoặc 1 trang tóm tắt, demo **5–7 phút**, repo công khai.

### Chỉ làm nếu đã hoàn thành MVP và còn thời gian

- Lặp lại Optuna với 3 seed để đo độ ổn định; **không thay thế kết quả run chính**, không dùng test để chọn seed.
- Random Search cùng search space và số lần đánh giá (30) để đối chứng bổ sung.
- Phân tích thêm feature importance, calibration hoặc đo latency nhiều lần.

### KHÔNG làm trong dự án này

- Streamlit/web app/API/deployment; XGBoost/CNN/LLM; data scraping; nhiều dataset; Optuna nhiều mục tiêu; hệ thống theo dõi experiment phức tạp; Docker/CI nâng cao nếu không có thời gian.

---

## 2. Mô hình hóa bài toán tối ưu (bắt buộc trình bày rõ trong báo cáo)

### 2.1. Biến quyết định

\[
x = (n_{\text{estimators}},\ \text{max\_depth},\ \text{min\_samples\_leaf})
\]

Đề xuất không gian tìm kiếm **nhỏ, đủ so sánh**:

| Biến | Miền giá trị được phép | Ghi chú |
|---|---|---|
| `n_estimators` | 50, 100, 150, 200, 250 | 100 của baseline được bao gồm |
| `max_depth` | `None`, 4, 6, 8, 12, 16, 24 | `None` của baseline được bao gồm |
| `min_samples_leaf` | số nguyên 1–8 | 1 của baseline được bao gồm |

Giữ nguyên các tham số khác ở giá trị mặc định, ngoại trừ `random_state=seed`, `n_jobs=1` để kết quả và timing nhất quán. **Không cố ý làm baseline bất lợi.** Nếu thay đổi không gian tìm kiếm phải ghi lý do trước khi chạy kết quả chính.

### 2.2. Hàm mục tiêu

\[
\min_{x \in \Omega} f(x) = -\frac{1}{3} \sum_{k=1}^{3}\operatorname{ROC\!\!\!-AUC}_{k}(x)
\]

Tương đương với **max ROC-AUC trung bình qua 3-fold CV**. Giữ CV folds cố định khi so sánh baseline và Optuna. Báo cả độ lệch chuẩn ROC-AUC giữa các fold.

### 2.3. Ràng buộc / budget / trade-off

- `x` phải thuộc miền tìm kiếm `Ω` ở bảng trên.
- Số lần đánh giá mô hình tối đa: `N_trials = 30` trong run chính.
- Chi phí tính toán là một **trade-off để đo và phân tích**: thời gian mỗi trial, tổng thời gian tìm kiếm, thời gian fit cuối, thời gian suy luận.
- **Không** mô tả latency là *ràng buộc cứng* nếu code không thật sự loại nghiệm vi phạm ngưỡng. Không đưa ra ngưỡng thời gian tùy tiện không được kiểm chứng.
- CPU đủ để chạy; **không yêu cầu GPU**.

### 2.4. Quyết định cuối cùng

Chọn baseline hoặc cấu hình Optuna dựa trên **CV metric, sai khác trên holdout test, độ phức tạp và chi phí tính toán**. Nếu chênh lệch ROC-AUC rất nhỏ hoặc không ổn định, có thể hợp lý khi chọn baseline đơn giản hơn. Không lựa chọn cấu hình chỉ vì nó cho test score cao hơn.

---

## 3. Dataset và protocol thực nghiệm

**Dataset:** `sklearn.datasets.load_breast_cancer`; đây là dataset benchmark công khai, không phải dữ liệu bệnh nhân cá nhân do người thực hiện thu thập. Ghi citation/URL chính thức và license/điều kiện tái sử dụng của nguồn trong `data/README.md`. Không đặt kết luận theo hướng sử dụng mô hình để chẩn đoán lâm sàng.

**Nhãn:** Dữ liệu scikit-learn gốc quy ước `0 = malignant` và `1 = benign`. Để `positive class` thể hiện **malignant**, trong code định nghĩa `y = (target == 0).astype(int)` và nêu rõ quy tắc này trong tài liệu. Khi dùng ROC-AUC/F1 phải nhất quán nhãn dương mới.

**Chia dữ liệu và bảo vệ test set:**

1. Tách dữ liệu một lần: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
2. Tạo `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` **chỉ trên train**.
3. Baseline: tính ROC-AUC CV trên **cùng** train/CV folds.
4. Optuna: mỗi trial xây dựng một Random Forest mới, thực hiện CV trên cùng train/CV folds và trả về ROC-AUC trung bình.
5. Sau khi chốt `best_params` **dựa trên CV**, fit baseline và model tối ưu trên toàn bộ train, rồi đánh giá **một lần** trên test với ROC-AUC, F1, accuracy/balanced accuracy, confusion matrix (nếu cần).
6. Không dùng test score để điều chỉnh search space, số trials, threshold, seed hoặc chọn winner. Có thể chạy lại toàn bộ pipeline để xác minh tái lập, nhưng không dùng vòng lặp test đó để tuning.

**Tính công bằng:** Chốt cách chia dữ liệu/CV/metrics trước khi tối ưu. So sánh baseline mặc định (không có search budget) với phương pháp Optuna (30 trials) và **báo rõ phần chi phí tìm kiếm tăng thêm**; không tuyên bố đây là so sánh 2 phương pháp HPO cùng budget. Nếu thêm Random Search thì bắt buộc cùng `30 trials`, cùng search space, cùng folds.

**Randomness:** cố định các seed chính cho split, CV, RF, Optuna; log lại seed. Kết quả 1 seed có giới hạn về độ ổn định, nêu trong báo cáo. Nếu còn thời gian, bổ sung nhiều seed và báo mean ± std.

---

## 4. Kỹ thuật, cấu trúc repo và đầu ra

**Ngôn ngữ:** Python 3.10+; ưu tiên code đơn giản, dễ giải thích hơn framework hóa.

**Thư viện chính:** `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`, `pytest` (nếu có tests); dùng `time.perf_counter()` để đo thời gian. Tránh thêm dependencies không cần thiết.

### Cấu trúc tối giản

```text
random-forest-optuna-optimization/
├── README.md                       # Tóm tắt dự án, công thức, kết quả, cách chạy
├── PROJECT_PLAN.md                 # File này (copy từ tài liệu được cung cấp)
├── requirements.txt                # Dependencies/phiên bản có thể tái lập
├── .gitignore                      # Bỏ qua .venv, __pycache__, cache, secrets...
├── src/
│   ├── __init__.py
│   ├── experiment.py               # Load/split, baseline, Optuna, final evaluation, logging
│   └── plots.py                    # Sinh biểu đồ TỪ log, không hard-code số liệu
├── data/
│   └── README.md                   # Nguồn/citation/cách lấy dữ liệu
├── results/
│   ├── trials.csv                  # Mỗi dòng = một trial
│   ├── baseline.json               # Metric CV, timing, params
│   ├── best_config.json            # Best params, best CV metric, seed
│   └── final_metrics.json          # So sánh trên holdout + timing
├── figures/
│   ├── best_so_far.png
│   └── auc_vs_runtime.png
├── tests/
│   └── test_pipeline.py            # Smoke test, shape, file output, data leakage checks
└── report/
    └── README.md                   # Báo cáo Word/PDF + slide khi đã hoàn thành
```

*Không bắt buộc có notebook; repo chạy được bằng script Python là đủ. Nếu tạo notebook để trình bày, tránh nhân bản toàn bộ logic.*

### Giao diện chạy mong muốn (Codex phải triển khai đúng)

Trên Windows PowerShell, ở thư mục gốc dự án:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Chạy thử nhanh, không ghi đè kết quả chính
python -m src.experiment --trials 3 --seed 42 --output-dir results/smoke

# Chạy thực nghiệm chính
python -m src.experiment --trials 30 --seed 42 --output-dir results

# Sinh lại hình từ log đã lưu
python -m src.plots --input-dir results --output-dir figures

# Kiểm tra cơ bản
python -m pytest -q
```

Nếu chính sách PowerShell chặn activate, dùng trực tiếp `.\.venv\Scripts\python.exe -m ...` mà không sửa chính sách hệ thống. Đây là **lệnh mục tiêu**, chỉ chạy được sau khi Codex tạo mã nguồn tương ứng.

### File logging phải có

`results/trials.csv` nên có các cột:

`trial_number, n_estimators, max_depth, min_samples_leaf, cv_auc_mean, cv_auc_std, duration_sec, state, seed`

Lưu `max_depth=None` nhất quán bằng giá trị rỗng hoặc chuỗi có tài liệu mô tả. Khi thử thất bại, log `state/error` và không coi score là hợp lệ.

`baseline.json`, `best_config.json`, `final_metrics.json` cần có: cấu hình, metric, seed, split/CV, số trials, phiên bản thư viện quan trọng, runtime và thông tin môi trường cơ bản. Không bịa số liệu hay tự điền thành tích giả.

### Hai biểu đồ bắt buộc

1. **Search/convergence plot:** trục x = số trial; trục y = `best-so-far CV ROC-AUC`; có đường ngang thể hiện baseline CV ROC-AUC.
2. **Trade-off plot:** scatter `cv_auc_mean` vs `duration_sec` của từng trial; làm nổi bật nghiệm tốt nhất và baseline nếu cùng cách đo thời gian được giải thích rõ.

Có thể bổ sung bảng so sánh baseline / optimized (CV mean±std; test ROC-AUC/F1; training/inference time). Biểu đồ và bảng **phải sinh từ code/log**, không sao chép số liệu minh họa.

---

## 5. Kế hoạch gọn theo 1–2 ngày/tuần

**Ưu tiên:** hoàn thành MVP trong **4 phiên làm việc** thuộc 2 tuần; tuần 3 dành để xử lý trục trặc/hoàn thiện nếu có thời gian. Không gán số giờ cố định cho một ngày vì lịch deadline cá nhân thay đổi.

| Phiên | Khi nào | Việc chính | Điều kiện hoàn thành |
|---|---|---|---|
| **S1 — Khởi tạo & baseline** | Tuần tập trung 1, ngày 1 | Tạo repo; môi trường; tải dataset từ sklearn; xử lý nhãn; train/test split; baseline CV; lưu baseline JSON | Chạy lại baseline bằng 1 lệnh; README có phần Problem/Formulation; commit có ý nghĩa |
| **S2 — Optuna & logging** | Tuần tập trung 1, ngày 2 (nếu có) hoặc tuần 2 | Cài objective CV; TPE; 3-trial smoke test; chạy 30-trial main; xuất trials CSV + best config | Không sử dụng test trong objective; 30 trials được ghi; commit có ý nghĩa |
| **S3 — Đánh giá & hình** | Tuần tập trung 2, ngày 1 | Final evaluation trên holdout; đo runtime; sinh 2 hình; bảng kết quả; smoke tests | Bảng/hình tái lập từ log; quyết định dựa trên số liệu thật; commit có ý nghĩa |
| **S4 — Bàn giao** | Tuần tập trung 2, ngày 2 | README đầy đủ; báo cáo 8–12 trang; slide hoặc 1 trang tóm tắt; hướng dẫn demo 5–7 phút; kiểm tra clone/run | Repo sạch, công khai khi nộp; README/đường dẫn trong báo cáo; commit/tag khi sẵn sàng |
| **S5 — Dự phòng** | Tuần tập trung 3, ngày 1 (nếu có) | Sửa lỗi reproducibility; nhiều seed hoặc Random Search (chỉ khi MVP xong) | Không phát sinh scope mới nếu còn lỗi chính |
| **S6 — Dự phòng** | Tuần tập trung 3, ngày 2 (nếu có) | Luyện demo/bảo vệ, rà soát rubric và trình bày | Chạy lại từ môi trường sạch, kết quả khớp báo cáo |

### Lưu ý quan trọng: quy định môn học vẫn là 5 tuần

Đề chính thức yêu cầu **checkpoint 5 tuần** và **lịch sử commit phản ánh tiến độ**, không chỉ upload toàn bộ dự án vào một commit cuối. Kế hoạch ở trên **nén thời gian lao động thực tế**, **không tự động thay thế** lịch checkpoint của giảng viên.

- Tạo repo và commit thật ngay phiên S1; mỗi phiên/đầu việc có commit rõ ràng.
- Nếu checkpoint của giảng viên kéo dài đến tuần 5, vẫn cần **nộp/cập nhật đúng lịch chính thức**, dù code lõi đã hoàn thành sớm. Dành các lần cập nhật ngắn cho sửa README, báo cáo, giải thích kết quả, demo.
- **Không giả mạo ngày commit hay tạo lịch sử tiến độ không có thật.** Nếu hoàn thành sớm mà không thể tham gia đủ checkpoint theo từng tuần, hỏi giảng viên xem có chấp nhận nghiệm thu sớm không.

### Thứ tự ưu tiên khi bị thiếu thời gian

1. **Không được bỏ:** formulation, baseline, Optuna thực sự chạy, holdout test đúng, CSV log, 2 plots, README, báo cáo và repo.
2. **Có thể tối giản:** EDA, số biểu đồ bổ sung, số lượng file/code module, tests phụ, slide (nếu chọn 1 trang tóm tắt được giảng viên chấp nhận).
3. **Cắt trước:** Random Search, chạy nhiều seed bổ sung, feature importance, giao diện và deploy.

---

## 6. Checklist nghiệm thu (Definition of Done)

### Kỹ thuật

- [x] `python -m src.experiment --trials 3 --seed 42 --output-dir results/smoke` chạy thành công.
- [x] `python -m src.experiment --trials 30 --seed 42 --output-dir results` chạy thành công.
- [x] Tất cả dữ liệu chỉ split **một lần**; đúng stratify và random seed.
- [x] CV của baseline và Optuna dùng **cùng train set, 3 folds, metric**.
- [x] Trong objective Optuna **không đọc X_test/y_test**; nhãn `malignant=1` nhất quán.
- [x] Các tham số baseline nằm trong search space; không sửa baseline để tạo chênh lệch giả.
- [x] `trials.csv` có đủ 30 trial và metric/params/duration/trial state.
- [ ] `best_config.json`, `baseline.json`, `final_metrics.json` có thật, không điền số giả.
- [ ] Sinh 2 hình qua `src.plots` từ log; không hard-code performance.
- [ ] Kết quả cuối có phân tích cả **mặt tốt, mặt hạn chế và chi phí search**.
- [x] `requirements.txt` có phiên bản hợp lý, ghi Python version và cách tái lập.
- [x] Không có secret, dữ liệu nhạy cảm, file mô hình quá lớn, `.venv` trong Git.

### Báo cáo/Portfolio

- [x] Problem statement + tên nguồn dataset + lý do chọn.
- [x] Viết rõ **x, f(x), Ω, budget/trade-off**; code trùng với công thức.
- [x] Baseline và nguyên lý TPE; không chỉ ghi `study.optimize(...)`.
- [x] Protocol train/CV/test; seed, budget, môi trường; giới hạn nghiên cứu.
- [ ] Bảng và ≥2 hình có thể tái lập; chọn nghiệm cuối bằng lý do và số liệu.
- [ ] Báo cáo Word/PDF 8–12 trang, repo link; slide 5–8 trang **hoặc** 1 trang tóm tắt; demo 5–7 phút.
- [ ] GitHub public; README có Problem, Optimization Formulation, Dataset, Method, Key Results, How to Run, Repository Structure, Limitations.
- [ ] Có ≥5 mốc commit có ý nghĩa; đối chiếu thêm yêu cầu **checkpoint theo tuần** nếu giảng viên kiểm tra.
- [x] Có mục **AI assistance** mô tả Codex hỗ trợ những gì và người làm đã tự kiểm chứng phần nào.

---

## 7. Nội dung mẫu đăng ký đề (điền theo bảng ở file hướng dẫn)

| Trường trong mẫu | Nội dung đề xuất |
|---|---|
| Tên đề tài | Tối ưu siêu tham số Random Forest bằng Optuna cho bài toán phân loại |
| Bối cảnh/vấn đề | Cấu hình Random Forest mặc định chưa chắc đạt ROC-AUC CV tốt nhất trong budget thực nghiệm; cần đánh giá lợi ích của HPO |
| Dataset/nguồn | Breast Cancer Wisconsin Diagnostic, qua scikit-learn (nguồn benchmark công khai; bổ sung citation/điều kiện sử dụng) |
| Decision variables `x` | `n_estimators`, `max_depth`, `min_samples_leaf` |
| Objective `f(x)` | Min âm trung bình ROC-AUC qua 3-fold Stratified CV trên train |
| Constraints/trade-off | Miền tham số hữu hạn; 30 trials; thời gian train/search/inference được đo làm trade-off |
| Baseline | `RandomForestClassifier` mặc định, seed/n_jobs nhất quán |
| Phương pháp tối ưu | Optuna TPE |
| Metric | CV ROC-AUC (chính), test ROC-AUC/F1 (báo cáo), runtime |
| Budget/điều kiện dừng | 30 trials; 3 folds/trial; seed cố định; dừng sau đủ trials hoặc lỗi được ghi nhận |
| Sản phẩm tuần cuối | GitHub repo công khai; Python code; CSV/JSON log; ≥2 hình; báo cáo 8–12 trang; slide/tóm tắt + demo |
| GitHub repository | `random-forest-optuna-optimization` (đề xuất; đường dẫn thực tế điền sau) |

---

## 8. Quy tắc dành riêng cho Codex

1. **Đọc file này trước khi code** và bám đúng phạm vi MVP.
2. Làm theo **S1 → S2 → S3 → S4**, mỗi bước chạy test/smoke test phù hợp; nếu lỗi thì sửa trước khi triển khai phần tiếp.
3. **Không viết code quá mức cần thiết:** ưu tiên 2 module `src/experiment.py`, `src/plots.py`, thêm helpers chỉ khi hợp lý.
4. Trước khi sửa đổi lớn, nêu ngắn gọn: file định sửa, lý do, đầu ra dự kiến. Sau bước hoàn tất, tóm tắt: đã làm gì, lệnh đã kiểm tra, kết quả thực tế, việc còn lại.
5. Giữ **một nguồn cấu hình** cho seed/split/CV/search space; không dùng các giá trị khác nhau giữa các file.
6. Không dùng/test-test/test leakage; không in kết quả mô phỏng như kết quả chính thức; không tự viết câu “đạt AUC 99%” nếu code chưa sinh ra log.
7. Nếu package/API thay đổi, ưu tiên giải pháp tối giản, cập nhật `requirements.txt` và README. Không tự ý thêm tính năng ngoài phạm vi.
8. Không commit/công khai secrets; **không tự tạo GitHub repo public, push hoặc phát hành release khi chưa được người dùng yêu cầu**.
9. Bảo đảm mọi bảng/hình trong báo cáo dẫn được đến CSV/JSON tạo ra từ code.
10. Nếu thiếu thời gian: ưu tiên **Definition of Done** và hoàn thiện sản phẩm nộp; bỏ phần mở rộng.

### Prompt để dán vào Codex (phiên đầu tiên)

```text
Hãy đọc kỹ PROJECT_PLAN.md và thực hiện CHỈ phiên S1 theo đúng phạm vi MVP.
Trước khi code, tóm tắt các file cần tạo và rủi ro chính. Tạo cấu trúc repo,
requirements.txt, data/README.md, phần problem/formulation trong README.md,
và code baseline có train/test split stratified + 3-fold CV ROC-AUC; lưu JSON.
Không triển khai Optuna cho đến khi baseline chạy được và tôi yêu cầu sang S2.
Chạy smoke test/kiểm tra lệnh thực tế. Không bịa kết quả, không push GitHub.
Cuối phiên: liệt kê file đã tạo, câu lệnh đã chạy, kết quả và bước S2 tiếp theo.
```

### Prompt cho các phiên sau

- **S2:** “Đọc `PROJECT_PLAN.md`. Chỉ triển khai S2: Optuna TPESampler, 30 trials, logging đầy đủ, smoke test 3 trials ra thư mục riêng. Chứng minh test set không vào objective; chạy thử và báo kết quả thực tế.”
- **S3:** “Đọc `PROJECT_PLAN.md`. Chỉ triển khai S3: final holdout evaluation một lần sau khi chốt best CV config, tạo hai biểu đồ từ log, thêm checks và bảng so sánh.”
- **S4:** “Đọc `PROJECT_PLAN.md`. Chỉ triển khai S4: hoàn thiện README đúng rubric, dàn ý báo cáo 8–12 trang, slide/tóm tắt demo, checklist tái lập. Không bịa số liệu và không tự push public.”

---

## 9. Nguồn đối chiếu yêu cầu

1. `DE_BAI_TAP_LON_CA_NHAN_5_TUAN_GITHUB_PORTFOLIO.pdf`: mục 2–3 (yêu cầu x/objective/constraints/baseline/log), mục 4 (5-week checkpoints), mục 5–8 (sản phẩm, GitHub, báo cáo, rubric), mục 9–10 (lỗi trừ điểm/công cụ AI).
2. `HUONG_DAN_PHAM_VI_DE_BAI_TAP_LON_GITHUB_PORTFOLIO.pdf`: mục 2–5 (scope HPO, mức xanh, benchmark), mục 7–10 (protocol, portfolio, mẫu đăng ký, checklist duyệt đề).
3. Nguồn dataset/API để ghi vào `data/README.md`:
   - scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
   - UCI Wisconsin Diagnostic Breast Cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
   - Optuna documentation: https://optuna.readthedocs.io/

**Lưu ý:** Các URL là tài liệu tra cứu dự kiến; khi thực hiện cần kiểm tra phiên bản thư viện/điều kiện sử dụng hiện hành. Mọi con số thực nghiệm trong báo cáo phải do chính code sinh ra.

---

## 10. Nhật ký triển khai S1 — 25/10/2026

**Phạm vi:** chỉ khởi tạo dự án và baseline. S2 phải chờ yêu cầu tiếp theo của người dùng.
Tài liệu đầu vào thực tế là `plans/PROJECT_PLAN_RANDOM_FOREST_OPTUNA_CODEX.md`;
bản đó được giữ nguyên, còn file ở gốc này là kế hoạch làm việc được cập nhật.
Khi bắt đầu, workspace chỉ có tài liệu và remote GitHub chưa có commit.

### Đã triển khai

- Khởi tạo Git nhánh `main`, cấu hình `origin` theo GitHub người dùng cung cấp.
- Tạo `.venv` với Python 3.12.10; ghim dependencies runtime và pytest trong `requirements.txt`.
- Tạo `src/__init__.py`, `src/experiment.py`: nạp dataset, đảo nhãn malignant=1,
  split stratified 80/20 một lần, 3-fold stratified CV trên train, baseline RF mặc định
  ngoại trừ `random_state=42`, `n_jobs=1`.
- Lưu `results/baseline.json`: tham số, ROC-AUC từng fold/mean/std, timing,
  seed, split/CV, phiên bản và môi trường; `n_trials=0`, `holdout_evaluated=false`.
- Tạo README với Problem/Formulation/protocol/cách chạy/giới hạn/AI assistance,
  `data/README.md` với nguồn/citation/giấy phép, `report/README.md`, thư mục figures,
  `.gitignore`, `.gitattributes` và `tests/test_pipeline.py`.
- Smoke baseline được lưu riêng tại `results/smoke/baseline.json` và bỏ qua trong Git.
  Đây là kiểm chứng baseline S1, không phải smoke 3 trials của S2.

### Lệnh kiểm chứng đã chạy thành công

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.experiment --seed 42 --output-dir results
.\.venv\Scripts\python.exe -m src.experiment --seed 42 --output-dir results/smoke
```

- `pip check`: không có dependencies bị lỗi.
- `pytest`: **7 passed**. Kiểm tra nhãn/shape, phân lớp stratified,
  train/test và CV không giao nhau, folds tái lập, baseline giữ mặc định,
  CV thực nhận đúng train, CLI xuất JSON và tái lập điểm số, từ chối seed sai.
- Đã đọc lại code thực tế và JSON đầu ra; đối chiếu main/smoke bằng assertion:
  tham số, điểm từng fold/mean/std và split/CV giống hệt nhau.
- Train: **455** mẫu (285 benign, 170 malignant); holdout: **114** mẫu
  (72 benign, 42 malignant); **30** đặc trưng.
- ROC-AUC từng fold: **0.9920590951061865**, **0.9918744228993536**,
  **0.9846804511278195**.
- ROC-AUC CV mean ± std (`ddof=0`): **0.98953798971112 ± 0.0034356257821230238**.
- Thời gian CV run chính: **0.6919962000101805 giây**; bao gồm 3 lần fit,
  scoring và overhead CV. Thời gian không phải đại lượng cần khớp khi tái lập.

### Điểm dừng và phần chưa làm

S1 hoàn tất ở baseline CV và commit cục bộ; không push GitHub.
Chưa có objective Optuna, `--trials`, trials CSV, best config, final metrics,
module plots, báo cáo hoặc slide. Các checkbox nghiệm thu toàn MVP ở mục 6
chưa được đánh dấu hoàn tất thay cho S1.

Khi người dùng yêu cầu S2: tái sử dụng cấu hình/folds train, triển khai objective
ROC-AUC và TPE với search space đã chốt, chạy 3-trial smoke riêng rồi 30-trial main,
lưu trials CSV và best config. Holdout vẫn dành cho S3.


---

## 11. Nhật ký triển khai S2 — 25/10/2026

**Phạm vi:** Optuna và logging, theo yêu cầu người dùng. S3 chưa triển khai.
Giữ nguyên dataset, nhãn, split/CV, seed 42, baseline và search space đã chốt.

### Thay đổi

- Ghim Optuna 4.5.0; thêm `TPESampler(seed=42, n_startup_trials=10)`, maximize
  ROC-AUC CV, 30 trials tuần tự, không pruning. Smoke 3 trials chỉ kiểm tra pipeline.
- `run_experiment` split một lần và tạo một danh sách folds dùng chung;
  baseline và mỗi RF mới của trial đi qua cùng hàm `evaluate_model`.
- Objective chỉ nhận train/folds/config, không nhận holdout hoặc toàn bộ DataSplit.
  Search space có đúng ba biến; tham số RF khác giữ mặc định ngoài seed/n_jobs.
- CSV cập nhật sau từng trial, gồm params, mean/std, điểm từng fold, timing, state,
  seed và error. `max_depth=None` được lưu bằng chuỗi `None` trong CSV, `null` trong JSON.
- Trial lỗi được ghi `FAIL` với điểm để trống rồi dừng; best config ghi trạng thái
  `incomplete`, không chọn trial lỗi. Kiểm chứng cả lỗi ngay trial đầu và sau trial thành công.
- Best config chứa cấu hình chọn từ CV, toàn bộ tham số RF, protocol, phiên bản/môi trường,
  sampler, số trials và thời gian search. JSON/CSV được ghi qua tệp tạm rồi thay thế.
- Lệnh mặc định hiện chạy 30 trials; muốn chạy lại S1 dùng `--baseline-only` và
  thư mục riêng. Smoke/budget khác 30 không được ghi vào thư mục kết quả chính.
- Cập nhật README, data/report README, .gitignore và tests. Không thêm pandas,
  matplotlib hoặc framework theo dõi thí nghiệm.

### Lệnh đã chạy và kiểm chứng

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.experiment --trials 3 --seed 42 --output-dir results/smoke
.\.venv\Scripts\python.exe -m src.experiment --trials 30 --seed 42 --output-dir results
.\.venv\Scripts\python.exe -m src.experiment --trials 30 --seed 42 --output-dir results/repro
```

- **16 tests passed**; dependencies không có xung đột.
- Test theo dõi việc split/folds: mỗi thao tác đúng một lần; baseline + các trial
  nhận chính cùng đối tượng X_train/y_train và folds, metric ROC-AUC.
  Kiểm tra closure objective chỉ giữ train arrays/folds/config.
- Đã đọc lại code/diff, CSV và JSON thực tế. Assertion xác nhận đủ **30/30 COMPLETE**,
  số trial 0–29, params đúng miền, tất cả điểm hữu hạn, mean/std tính lại từ từng fold
  khớp log, best config đúng dòng có CV cao nhất và metadata nhất quán.
- Chạy lại 30 trials tại `results/repro`: toàn bộ params, điểm từng fold/mean/std,
  state và seed khớp tuyệt đối với run chính; timing được loại khỏi phép so sánh.
  Ba trial smoke cũng khớp ba trial đầu run chính. Không dùng holdout để kiểm chứng.
- Smoke best CV: **0.987947940025502**, 3 trials; chỉ giữ cục bộ.
- Baseline main CV: **0.989537989711120 ± 0.003435625782123**.
- Best trial **#10**: `{'n_estimators': 50, 'max_depth': 16, 'min_samples_leaf': 1}`;
  CV **0.989933715868619 ± 0.004250459350418**.
- Chênh CV: **+0.000395726157499 AUC**; mức chênh nhỏ, chưa kết luận
  khả năng tổng quát hóa hoặc chọn mô hình cuối.
- Thời gian CV baseline main: **0.715908 giây**;
  search main: **24.978321 giây** (loại trừ baseline).

### Điểm dừng

S2 hoàn tất với `results/baseline.json`, `results/trials.csv` và
`results/best_config.json`; commit cục bộ, không push GitHub.
Holdout chưa được đánh giá (`holdout_evaluated=false`), chưa tạo final_metrics,
plots hay báo cáo. Chỉ triển khai S3 sau khi người dùng yêu cầu.

---

## 12. Tiến độ S3 và điểm tiếp tục — 25/10/2026

Người dùng yêu cầu triển khai S3 và lưu tiến độ phòng khi hết quota.
Commit nền: `90803c6`. Chỉ S3; chưa làm S4, chưa push GitHub.

### Mốc hiện tại

- [x] Đọc kế hoạch/code/log S2; Git sạch khi bắt đầu.
- [ ] Thêm final evaluation từ cấu hình S2 đã lưu, không chạy lại tuning.
- [ ] Thêm plots từ log và bảng so sánh, dependencies, kiểm tra S3.
- [ ] Chạy kiểm tra, đánh giá holdout chính, lưu `results/final_metrics.json`.
- [ ] Sinh hai hình, xem ảnh thực tế, cập nhật kết luận/README và commit.

### Quy tắc tiếp tục nếu phiên bị gián đoạn

1. Đọc mục này, `git status`, code thực tế và các file results trước khi làm tiếp.
2. Giữ nguyên ba file S2: baseline.json, trials.csv, best_config.json. Best params:
   n_estimators=50, max_depth=16, min_samples_leaf=1; seed=42, n_jobs=1.
3. Holdout chỉ đánh giá sau khi chốt cấu hình; threshold cố định 0.5 (ties chọn lớp 0,
   phù hợp argmax của RandomForest). Không tuning lại seed/search/threshold theo test.
4. Nếu `final_metrics.json` đã tồn tại, đọc/kiểm tra trước, dùng lại cho bảng/hình;
   không chạy lại holdout chỉ để lấy số đẹp hơn. Các log xuất phải ghi phiên bản và timing.
5. Chỉ báo hoàn tất sau khi kiểm tra lệnh thực, nội dung JSON/CSV và hai ảnh.

Mốc này được ghi trước khi viết code S3. Chưa có kết quả holdout.
