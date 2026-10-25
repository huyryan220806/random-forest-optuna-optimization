# Dataset

Dự án dùng duy nhất **Breast Cancer Wisconsin (Diagnostic)**: 569 mẫu,
30 đặc trưng số, hai lớp malignant/benign. Các đặc trưng mô tả nhân tế bào
từ ảnh FNA. Đây là benchmark công khai cho bài tập học thuật.

## Nguồn và citation

- [UCI dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- [scikit-learn 1.7: load_breast_cancer](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)

Citation theo UCI: Wolberg, W., Mangasarian, O., Street, N., & Street, W.
(1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine
Learning Repository. DOI: 10.24432/C5DW2B.

Trang UCI công bố giấy phép **CC BY 4.0**. Khi tái sử dụng cần ghi nguồn,
dẫn giấy phép và nêu thay đổi; xem [điều kiện CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Nguồn/citation/giấy phép được đối chiếu ngày 25/10/2026.

## Cách lấy và biến đổi

```python
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X = dataset.data
y = (dataset.target == 0).astype(int)
```

Dataset đi kèm scikit-learn, chạy không cần tải CSV hoặc truy cập mạng.
Không lưu bản sao dữ liệu thô trong repo.

| Nhãn | scikit-learn gốc | Trong dự án |
|---|---:|---:|
| malignant (ác tính) | 0 | **1 — positive** |
| benign (lành tính) | 1 | 0 |

Thay đổi duy nhất với dữ liệu là đảo mã nhãn như trên và chia train/test.
Không chuẩn hóa, điền khuyết, chọn đặc trưng hay cân bằng lại lớp.
ROC-AUC dùng xác suất của lớp 1; F1 ở S3 cũng sẽ dùng malignant làm nhãn dương.

Tách stratified 80/20, seed 42. S1 chỉ chạy 3-fold stratified CV trên train;
holdout được dành lại cho S3. Chỉ dùng mô hình cho thực nghiệm môn học,
không suy diễn thành công cụ chẩn đoán lâm sàng.
