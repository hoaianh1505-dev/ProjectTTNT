# Customer Clustering Project

Dự án này sử dụng thuật toán **K-Means Clustering** để phân nhóm khách hàng dựa trên dữ liệu thẻ tín dụng. Mục tiêu là giúp doanh nghiệp hiểu rõ hơn về hành vi khách hàng để có chiến lược kinh doanh và tiếp thị phù hợp.

## Cấu trúc thư mục

```
customer-clustering/
├── data/
│   ├── raw/               # Dữ liệu gốc (credit_card_customers.csv)
│   └── processed/         # Dữ liệu sau khi phân cụm và bảng tóm tắt (cluster_summary.csv, ... )
├── results/
│   └── figures/           # Biểu đồ (Elbow, Silhouette, PCA)
├── src/
│   └── kmeans_clustering.py  # Script chính thực hiện phân cụm
├── requirements.txt       # Danh sách các thư viện cần thiết
└── README.md              # Tài liệu dự án
```

## Yêu cầu cài đặt

Đảm bảo bạn đã cài đặt Python. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

## Cách chạy dự án

Để thực hiện phân cụm, chạy script `kmeans_clustering.py` từ thư mục gốc hoặc thư mục `src`:

```bash
python src/kmeans_clustering.py
```

## Quy trình xử lý (Pipeline)

1.  **Load Data**: Đọc dữ liệu từ `data/raw/credit_card_customers.csv`.
2.  **Preprocessing**: 
    - Loại bỏ các cột không cần thiết (`CUST_ID`, `TENURE`).
    - Xử lý dữ liệu thiếu (Imputation).
    - Chuẩn hóa dữ liệu (Standard Scaling).
3.  **Clustering**:
    - Xác định K tối ưu bằng phương pháp **Elbow** và **Silhouette Score**.
    - Chạy K-Means với K tối ưu.
4.  **Analysis & Visualization**:
    - Lưu biểu đồ đánh giá vào `results/figures/`.
    - Lưu dữ liệu đã gắn nhãn cụm vào `data/processed/`.
    - In ra đặc điểm trung bình của từng nhóm để phân tích.

## Kết quả

Mỗi lần chạy script, kết quả sẽ được lưu vào một thư mục mới trong `results/` với tên theo định dạng `run_YYYYMMDD_HHMMSS`.

Cấu trúc mỗi lần chạy:
- **`results/run_<timestamp>/figures/`**: Chứa các biểu đồ (Elbow, Silhouette, PCA).
- **`results/run_<timestamp>/data/`**: Chứa dữ liệu kết quả của lần chạy đó.

Ngoài ra, bản sao mới nhất cũng được lưu tại:
- **`data/processed/`**: Dữ liệu đã gắn nhãn và bảng tóm tắt mới nhất.

## Kiểm tra phù hợp với "Đề tài 5: Áp dụng thuật toán K-means để phân cụm khách hàng"

- Yêu cầu chính:
  - Dữ liệu khách hàng: sử dụng file `data/raw/credit_card_customers.csv` — OK
  - Thuật toán: K-Means — OK (được sử dụng trong `src/kmeans_clustering.py`)
  - Tiền xử lý: loại bỏ cột ID/TENURE, impute, chuẩn hoá — OK (StandardScaler, median imputation)
  - Chọn K: Elbow + Silhouette — OK (cả hai biểu đồ được sinh)
  - Lưu kết quả: dữ liệu có cột `Cluster`, bảng `cluster_summary.csv`, biểu đồ vào `results/figures/` — OK

- Gợi ý cải tiến (không bắt buộc):
  - Đóng gói logic vào hàm `main()` và thêm CLI/logging để chạy linh hoạt.
  - Tránh gọi plt.show() khi chạy trên server/headless (chỉ lưu ảnh).
  - Lưu scaler/mô hình nếu cần tái sử dụng.
  - Thêm mô tả/nhãn cụm trong báo cáo hoặc notebook phân tích.

## Lưu ý về tái cấu trúc

- Thư mục `report/` (nếu trống) và `outputs/tables/` được loại bỏ trong cấu trúc đề xuất.
- `outputs/` được đổi tên thành `results/`; tất cả biểu đồ lưu ở `results/figures/`.
- Bảng/tóm tắt lưu ở `data/processed/` (ví dụ `cluster_summary.csv` và `credit_card_customers_with_clusters.csv`).

## Verification Plan

1. Chạy script:
```bash
python src/kmeans_clustering.py
```
2. Kiểm tra:
- `report/` và `outputs/` không tồn tại (hoặc đã được loại bỏ nếu có).
- `results/figures/` chứa: `elbow_method.png`, `silhouette_scores.png`, `pca_clusters_2d.png`.
- `data/processed/` chứa: `credit_card_customers_with_clusters.csv`, `cluster_summary.csv`.
