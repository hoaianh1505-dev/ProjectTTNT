# Customer Clustering Project

Dự án này sử dụng thuật toán **K-Means Clustering** để phân nhóm khách hàng dựa trên dữ liệu thẻ tín dụng. Mục tiêu là giúp doanh nghiệp hiểu rõ hơn về hành vi khách hàng để có chiến lược kinh doanh và tiếp thị phù hợp.

## Cấu trúc thư mục

```
customer-clustering/
├── data/
│   ├── raw/               # Dữ liệu gốc (credit_card_customers.csv)
│   └── processed/         # Dữ liệu sau khi phân cụm và bảng tóm tắt
├── results/
│   └── run_YYYYMMDD_.../  # Kết quả mỗi lần chạy (biểu đồ, bảng số liệu)
├── src/
│   └── kmeans_clustering.py  # Script chính (đã được tối ưu hóa)
├── requirements.txt       # Danh sách thư viện
└── README.md              # Tài liệu dự án
```

## Yêu cầu cài đặt

Bạn cần cài đặt Python và các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

*(Các thư viện chính: `pandas`, `numpy`, `scikit-learn`, `matplotlib`)*

## Cách chạy dự án

Chạy file script chính từ thư mục gốc:

```bash
python src/kmeans_clustering.py
```

### Quá trình thực hiện của code:
1.  **Load Data**: Tự động tìm file csv trong `data/raw/` (mặc định là `credit_card_customers.csv`).
2.  **Preprocessing**: Làm sạch dữ liệu, xử lý giá trị thiếu, và chuẩn hóa (Scaling).
3.  **Clustering**: Phân chia khách hàng thành 4 nhóm (K=4) bằng K-Means.
    *   *(Tùy chọn: Bạn có thể bỏ comment dòng `find_optimal_k` trong code để chạy lại thuật toán Elbow/Silhouette tìm K).*
4.  **Analysis**: Tính toán các chỉ số trung bình của từng nhóm.
5.  **Visualization**: Vẽ và lưu các biểu đồ vào thư mục `results/`.

## Kết quả đầu ra

Sau khi chạy xong, kết quả sẽ nằm trong thư mục `results/run_<thời-gian>/`:

-   **Biểu đồ (`figures/`)**:
    -   `pca_clusters_2d.png`: Biểu đồ phân cụm (PCA).
    -   `cluster_counts.png`: Số lượng khách hàng mỗi nhóm.
    -   `radar_chart.png`: So sánh đặc điểm các nhóm (Radar chart).
    -   `elbow_method.png` / `silhouette_scores.png` (nếu chạy tìm K).
-   **Dữ liệu (`*.csv`)**:
    -   `clustered_data.csv`: Dữ liệu gốc kèm cột `Cluster`.
    -   `cluster_summary.csv`: Bảng đặc điểm trung bình của từng nhóm.

## Cải tiến & Tái cấu trúc

Code đã được tái cấu trúc để sạch sẽ và dễ bảo trì hơn:
-   Sử dụng hàm (`load_data`, `preprocess_data`, `main`...) thay vì viết tuần tự.
-   Dễ dàng đọc hiểu luồng xử lý chính.
-   Tự động quản lý thư mục đầu ra theo thời gian chạy.
