# Demo và bảo vệ trong 5–7 phút

Mục tiêu: trình bày chuỗi **bài toán → formulation → baseline → TPE → log → kết quả → quyết định**.
Kịch bản dự kiến **6 phút 30 giây**, chưa gồm hỏi đáp. Đây là tài liệu để luyện,
không phải xác nhận sinh viên đã trình bày.

## Chuẩn bị

- Điền họ tên, MSSV, lớp, giảng viên trong Word/slide; hiện các mục là **(tự điền)**.
- Mở slide, README và PowerShell tại gốc repo. Cài dependencies trước buổi demo.
- Mở sẵn `src/experiment.py` tại `make_objective`, `results/trials.csv`,
  `best_config.json`, `final_metrics.json` và hai hình trong `figures/`.
- Nếu không có mạng, repo đã clone và môi trường đã cài vẫn chạy được vì dataset
  có sẵn trong scikit-learn.

## Nhịp trình bày

| Thời gian | Slide/thao tác | Ý cần nói |
|---|---|---|
| 0:00–0:25 | Slide 1 | TPE cải thiện CV AUC so với mặc định bao nhiêu, với chi phí nào? |
| 0:25–1:05 | Slide 2 | 569 mẫu/30 features, malignant=1, train 455/test 114, CV chỉ trên train. |
| 1:05–1:50 | Slide 3 | Ba biến, Ω có 280 cấu hình, maximize mean AUC 3 folds, budget 30 trials. |
| 1:50–2:35 | Slide 4 | Baseline giữ mặc định; 10 startup trials rồi TPE dùng l(x)/g(x); logging. |
| 2:35–3:20 | Slide 5 | Best #10 là lần thử 11; CV tăng 0.000396; std không phải khoảng tin cậy. |
| 3:20–4:00 | Slide 6 | Search thêm 24.978 giây; trial lâu hơn không bảo đảm tốt hơn. |
| 4:00–4:45 | Slide 7 | Test AUC 0.992890 → 0.994048; F1 0.962963 không đổi; mỗi model bỏ sót 3 malignant. |
| 4:45–5:20 | Slide 8 | Giữ cấu hình 50/depth 16/leaf 1 theo CV; một seed/holdout nhỏ chưa chứng minh ưu thế ổn định. |
| 5:20–6:20 | Terminal + code | Smoke 3 trials, đọc cached holdout, sinh hình riêng; objective không nhận test. |
| 6:20–6:30 | README/repo | Chỉ đường dẫn code/log/báo cáo/lệnh chạy lại; kết luận về lợi ích nhỏ. |

## Lệnh demo

PowerShell tại gốc repo, dùng virtualenv đã cài. Nếu `results/smoke` đã có
`final_metrics.json`, đổi sang thư mục mới như `results/smoke/demo-2`.

```powershell
# Smoke chưa đi qua 10 trials khởi tạo; không thay kết quả chính.
.\.venv\Scripts\python.exe -m src.experiment --trials 3 --seed 42 --output-dir results/smoke

# Đọc S3 complete, không fit/predict thêm.
.\.venv\Scripts\python.exe -m src.experiment --evaluate-only --input-dir results

# Sinh hình/bảng vào thư mục demo được gitignore.
.\.venv\Scripts\python.exe -m src.plots --input-dir results --output-dir figures/repro/demo
Get-Content figures/repro/demo/comparison.md
```

Chỉ trong code: `make_objective(X_train, y_train, cv_splits, config)`; baseline và
trial cùng dùng `evaluate_model`; `selection.uses_holdout=false` trong final JSON.
CSV đếm trial từ 0, trục hình đếm từ 1.

Không chạy lại search trong `results` đã đóng băng sau S3. Nếu cần chứng minh toàn
pipeline, dùng thư mục mới theo [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Nếu terminal/projector trục trặc, mở log/hình có sẵn và nói rõ đó là kết quả đã lưu;
không đổi seed/tham số để tìm lần chạy đẹp hơn.

## Câu hỏi bảo vệ

**Vì sao AUC?** Objective đánh giá xếp hạng xác suất qua nhiều ngưỡng. F1 vẫn được
báo cho malignant tại ngưỡng cố định.

**Vì sao không dùng test trong tuning?** Nếu chỉnh cấu hình theo test, nó không
còn là kiểm tra độc lập cho quá trình chọn cấu hình đó.

**Optuna có chắc tốt hơn?** Run này CV tăng nhẹ, ít cây hơn; test AUC tăng nhẹ nhưng
F1 và confusion matrix bằng nhau. Chưa có kiểm định, nhiều seed hay Random Search
cùng budget để kết luận mạnh hơn.

**Confusion matrix bằng nhau có nghĩa cùng mẫu sai?** Không thể suy ra từ ma trận
tổng hợp. Log xác nhận số lượng TN/FP/FN/TP giống nhau.

**Đã tìm hết Ω?** Chưa. 30 trials so với 280 cấu hình và có thể lặp; đường nằm ngang
không chứng minh cực trị toàn cục.

**Smoke có chứng minh TPE chưa?** Smoke 3 chỉ kiểm tra đường chạy. 10 trials đầu là
khởi tạo ngẫu nhiên; main 30 mới đi qua giai đoạn dùng lịch sử.

**Tại sao timing khác khi chạy lại?** Khác máy/tải/cache; timing đo một lần. Đối chiếu
cấu hình, split/folds và metrics với cùng phiên bản, không ép timing giống nhau.

**AI và bạn tự làm gì?** Codex hỗ trợ code, tài liệu và kiểm tra tự động. Phần người
học trực tiếp kiểm chứng phải mô tả đúng thực tế; hiện là **(tự điền)**.

## Việc cá nhân trước khi nộp

- [ ] Điền bìa/slide và phần tự kiểm chứng; xuất lại PDF sau khi sửa Word.
- [ ] Tự chạy lệnh, đọc code, giải thích formulation/TPE/bảng/hình.
- [ ] Luyện demo và đo thời lượng thực tế 5–7 phút.
- [ ] Nộp repo public và báo cáo lên LMS đúng lịch.
- [ ] Đối chiếu checkpoint 5 tuần với giảng viên; không giả ngày commit.
