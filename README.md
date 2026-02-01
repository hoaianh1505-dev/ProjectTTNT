# Customer Clustering Project

Dự án này sử dụng thuật toán **K-Means Clustering** để phân nhóm khách hàng dựa trên dữ liệu thẻ tín dụng. Mục tiêu là giúp doanh nghiệp hiểu rõ hơn về hành vi khách hàng để có chiến lược kinh doanh và tiếp thị phù hợp.

## Cấu trúc thư mục

```
customer-clustering/
├── data/
│   ├── raw/               # Dữ liệu gốc (credit_card_customers.csv)
│   └── processed/         # Dữ liệu sau khi phân cụm và các bảng tóm tắt
├── outputs/               # Biểu đồ (Elbow, Silhouette, PCA)
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
    - Lưu biểu đồ đánh giá vào `outputs/`.
    - Lưu dữ liệu đã gắn nhãn cụm vào `data/processed/`.
    - In ra đặc điểm trung bình của từng nhóm để phân tích.

## Kết quả

Sau khi chạy thành công, bạn sẽ nhận được:
- **`data/processed/credit_card_customers_with_clusters.csv`**: File dữ liệu gốc kèm theo cột `Cluster`.
- **`data/processed/cluster_summary.csv`**: Bảng thống kê giá trị trung bình các đặc trưng của từng cụm.
- **`outputs/`**: Các biểu đồ minh họa quá trình chọn K và kết quả phân cụm (PCA 2D).
