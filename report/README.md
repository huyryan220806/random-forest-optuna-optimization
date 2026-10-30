# Tài liệu bàn giao S4

Tài liệu dùng số liệu thật từ log S2/S3. Họ tên, MSSV, lớp và giảng viên đều để
**(tự điền)** theo yêu cầu.

| Tài liệu | Nội dung |
|---|---|
| [Báo cáo Word](Bao_cao_Random_Forest_Optuna.docx) | 11 trang, chỉnh sửa được; công thức native Word, bảng/hình và nguồn tham khảo |
| [Báo cáo PDF](Bao_cao_Random_Forest_Optuna.pdf) | Bản xuất từ Word để đọc/nộp |
| [Slide](Trinh_bay_Random_Forest_Optuna.pptx) | 8 slide, 3 bảng và 2 biểu đồ native chỉnh sửa được, có speaker notes |
| [Kịch bản demo](DEMO.md) | Dự kiến 6 phút 30 giây, lệnh chạy và câu hỏi bảo vệ |
| [Kiểm chứng tái lập](REPRODUCIBILITY.md) | Clone/venv mới, 24 tests, chạy lại 30 trials/holdout/plots |

## Nguồn và phạm vi

Số liệu từ `results/baseline.json`, `trials.csv`, `best_config.json` và
`final_metrics.json`. Hình báo cáo lấy từ `figures/`, sinh bằng `src.plots`.
Chart slide đọc cùng log, làm tròn 10 chữ số thập phân cho workbook nhúng;
CSV/JSON gốc giữ nguyên độ chính xác. Không thêm dataset, seed hoặc HPO khác.

Giữ cấu hình chọn bằng CV (50 cây, depth 16, leaf 1). AUC nhỉnh hơn nhưng F1 và
confusion matrix không đổi; chưa chứng minh ưu thế thống kê. Mỗi cấu hình bỏ sót
3 malignant; ma trận tổng hợp không xác định danh tính các mẫu sai.

## Chỉnh sửa và dựng lại

Có thể sửa Word/PPTX trực tiếp, xuất lại PDF sau khi điền thông tin. Script sau
dành cho biên soạn, **không cần chạy để tái lập thí nghiệm**:

- `build_report.py`: python-docx, đọc JSON/hình để tạo Word; không fit model.
- `build_slides.mjs`: `@oai/artifact-tool`, đọc CSV/JSON, dựng bảng/chart native,
  chạy package/layout/chart validators trước xuất PPTX.

Word được dựng bằng Python của Codex bundled runtime có python-docx. LibreOffice
không có trong môi trường Windows này; PDF xuất bằng Microsoft Word, render PNG
bằng Poppler để kiểm tra từng trang. Không thêm gói biên soạn vào requirements ML.

Đã kiểm tra toàn bộ 11 trang và 8 slide. PPTX qua package/layout/chart validators;
đã mở trong PowerPoint, xuất cả 8 slide và kiểm tra chỉnh sửa/resize chart rồi
lưu/mở lại trên bản sao. Chưa kiểm tra Google Slides; không suy ra tương thích
hoàn toàn giữa các ứng dụng chỉ từ việc PowerPoint mở được.

Để chạy script slide trong Codex có kỹ năng Presentations: sao chép script vào
thư mục build riêng, liên kết `node_modules` tới package bundle; dùng Node bundle,
đặt `PROJECT_ROOT`, `PRESENTATIONS_SKILL_DIR`, `ARTIFACT_PYTHON`, `RUNTIME_NODE_MODULES`.
`SLIDES_FILENAME` phải là tên mới nếu đầu ra cũ đã tồn tại; finalizer không ghi đè
bản đã kiểm chứng. Các đường dẫn runtime phụ thuộc máy, không phải dependency ML.

## Việc người học cần hoàn tất

Điền thông tin và phần tự kiểm chứng, tự đọc/chạy/giải thích code, luyện demo, nộp
LMS và đối chiếu checkpoint 5 tuần. Codex không xác nhận thay hoạt động cá nhân này.
Checklist nằm cuối [DEMO.md](DEMO.md).
