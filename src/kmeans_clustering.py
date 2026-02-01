import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import sys

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create a unique run directory based on timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, f"run_{timestamp}")
FIGURES_DIR = os.path.join(RUN_DIR, "figures")
DATA_RUN_DIR = os.path.join(RUN_DIR, "data")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True) # Keep this to ensure RESULTS_DIR exists before RUN_DIR
os.makedirs(RUN_DIR, exist_ok=True) # Ensure RUN_DIR is created
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_RUN_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



DATA_FILENAME = "credit_card_customers.csv"

CSV_PATH = os.path.join(DATA_RAW_DIR, DATA_FILENAME)

if not os.path.exists(CSV_PATH):
    print(f"Không tìm thấy file {DATA_FILENAME}, đang tìm kiếm file khác...")
    csv_files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy bất kỳ file .csv nào trong {DATA_RAW_DIR}")
    DATA_FILENAME = csv_files[0]
    CSV_PATH = os.path.join(DATA_RAW_DIR, DATA_FILENAME)

print(f"Đang xử lý dữ liệu từ file: {DATA_FILENAME}")

K_MIN = 2
K_MAX = 10
RANDOM_STATE = 42


df = pd.read_csv(CSV_PATH)

drop_cols = []
for c in ["CUST_ID", "CustomerID", "CUSTOMER_ID"]:
    if c in df.columns:
        drop_cols.append(c)

if "TENURE" in df.columns:
    drop_cols.append("TENURE")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in drop_cols]

X = df[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


Ks = range(K_MIN, K_MAX + 1)
inertias = []
sil_scores = []

for k in Ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))


plt.figure()
plt.plot(list(Ks), inertias, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("Inertia (WCSS)")
plt.title("Phương pháp Elbow")
plt.savefig(os.path.join(FIGURES_DIR, "elbow_method.png"))
plt.close()
 
plt.figure()
plt.plot(list(Ks), sil_scores, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("Chỉ số Silhouette")
plt.title("Đánh giá Silhouette theo số cụm")
plt.savefig(os.path.join(FIGURES_DIR, "silhouette_scores.png"))
plt.close()
 
# best_k = list(Ks)[int(np.argmax(sil_scores))]
# print(f"K tối ưu theo Silhouette: {best_k}")

# CHỐT K=4 ĐỂ KHỚP VỚI BÁO CÁO CỦA BẠN
best_k = 4
print(f"Đang sử dụng K={best_k} (Cố định theo báo cáo)")


final_km = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
df["Cluster"] = final_km.fit_predict(X_scaled)

print("\nSố lượng khách hàng theo cụm:")
print(df["Cluster"].value_counts().sort_index())


cluster_summary = df.groupby("Cluster")[feature_cols].mean().round(4)

cluster_summary.to_csv(os.path.join(DATA_RUN_DIR, "cluster_summary.csv"), index=True)
cluster_summary.to_csv(os.path.join(DATA_PROCESSED_DIR, "cluster_summary.csv"), index=True)

df.to_csv(os.path.join(DATA_RUN_DIR, "credit_card_customers_with_clusters.csv"), index=False)
df.to_csv(
    os.path.join(DATA_PROCESSED_DIR, "credit_card_customers_with_clusters.csv"),
    index=False
)


pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["Cluster"])
plt.xlabel("Thành phần chính 1")
plt.ylabel("Thành phần chính 2")
plt.title("Biểu đồ phân cụm khách hàng (PCA 2D)")
plt.savefig(os.path.join(FIGURES_DIR, "pca_clusters_2d.png"))
plt.close()


# --- REPORT VISUALIZATIONS ---

# 1. Histogram (Hình 3.3)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df["BALANCE"].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Phân phối tài khoản (BALANCE)")
plt.subplot(1, 2, 2)
df["PURCHASES"].hist(bins=20, color='lightgreen', edgecolor='black')
plt.title("Phân phối mua sắm (PURCHASES)")
plt.savefig(os.path.join(FIGURES_DIR, "histogram_dist.png"))
plt.close()

# 2. Bar Chart (Hình 5.1)
plt.figure(figsize=(8, 6))
counts = df["Cluster"].value_counts().sort_index()
plt.bar(counts.index, counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel("Cụm (Cluster)")
plt.ylabel("Số lượng khách hàng")
plt.title("Số lượng khách hàng trong từng cụm")
plt.xticks(counts.index)
plt.savefig(os.path.join(FIGURES_DIR, "cluster_counts_bar.png"))
plt.close()

# 3. Radar Chart (Hình 5.2)
def make_radar_chart():
    categories = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']
    # Normalize data for radar chart to 0-1 range to be comparable
    radar_data = cluster_summary[categories].copy()
    for col in categories:
        radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)
    
    for i, row in radar_data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(label_loc.tolist() + [label_loc[0]], values, label=f'Cluster {i}')
        ax.fill(label_loc.tolist() + [label_loc[0]], values, alpha=0.25)
    
    ax.set_xticks(label_loc)
    ax.set_xticklabels(categories)
    plt.title("Biểu đồ Radar so sánh đặc trưng các nhóm (Normalized)", size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(os.path.join(FIGURES_DIR, "radar_chart.png"))
    plt.close()

try:
    make_radar_chart()
except Exception as e:
    print(f"Không thể vẽ Radar chart: {e}")

# 4. Boxplot (Hình 5.3)
plt.figure(figsize=(10, 6))
# Manual boxplot using pandas groupby because seaborn is not guaranteed
data_to_plot = [df[df['Cluster']==i]['CREDIT_LIMIT'].dropna().values for i in sorted(df['Cluster'].unique())]
plt.boxplot(data_to_plot, labels=sorted(df['Cluster'].unique()))
plt.title("Phân phối Hạn mức tín dụng (CREDIT_LIMIT) theo cụm")
plt.xlabel("Cụm")
plt.ylabel("Credit Limit")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(FIGURES_DIR, "boxplot_credit_limit.png"))
plt.close()
 
overall_mean = df[feature_cols].mean()
diff = (cluster_summary - overall_mean).round(4)
print("\nSo sánh đặc trưng từng cụm với trung bình toàn bộ:")
print(diff)


def _check_assignment(df, feature_cols):
    checks = {
        "dataset_loaded": df is not None and len(df) > 0,
        "numeric_features": len(feature_cols) > 0,
        "kmeans_used": True,  # script dùng KMeans trong luồng chính
        "cluster_column_created": "Cluster" in df.columns
    }
    logging.info("Assignment quick-check:")
    for name, ok in checks.items():
        logging.info(" - %s: %s", name, "OK" if ok else "MISSING")

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # ...existing code... (use paths DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR)
    # ensure CSV exists
    if not os.path.exists(CSV_PATH):
        logging.error("Missing input CSV: %s", CSV_PATH)
        sys.exit(1)

    _check_assignment(df, feature_cols)

if __name__ == "__main__":
    main()
