import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


import os

CSV_PATH = "credit_card_customers.csv"  
# Xác định đường dẫn gốc dự án (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Đường dẫn input/output
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Đảm bảo thư mục tồn tại
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_RAW_DIR, "credit_card_customers.csv")
K_MIN = 2
K_MAX = 10
RANDOM_STATE = 42


df = pd.read_csv(CSV_PATH)

print("=== Preview ===")
print(df.head())
print("\n=== Columns ===")
print(df.columns.tolist())
print("\n=== Missing per column ===")
print(df.isna().sum())


drop_cols = []
for c in ["CUST_ID", "CustomerID", "CUSTOMER_ID"]:
    if c in df.columns:
        drop_cols.append(c)

if "TENURE" in df.columns:
    drop_cols.append("TENURE")


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


feature_cols = [c for c in numeric_cols if c not in drop_cols]

if len(feature_cols) < 2:
    raise ValueError(f"Không đủ cột số để phân cụm. Numeric cols: {numeric_cols}")

print("\nCột dùng để phân cụm:", feature_cols)

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
plt.xlabel("K (số cụm)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.savefig(os.path.join(OUTPUTS_DIR, "elbow_method.png"))
plt.show()
plt.close()

plt.figure()
plt.plot(list(Ks), sil_scores, marker="o")
plt.xlabel("K (số cụm)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Scores")
plt.savefig(os.path.join(OUTPUTS_DIR, "silhouette_scores.png"))
plt.show()
plt.close()

best_k = list(Ks)[int(np.argmax(sil_scores))]
print(f"\nK tốt nhất theo Silhouette: {best_k} (score={max(sil_scores):.4f})")


final_km = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
df["Cluster"] = final_km.fit_predict(X_scaled)

print("\n=== Số lượng khách mỗi cụm ===")
print(df["Cluster"].value_counts().sort_index())


cluster_summary = df.groupby("Cluster")[feature_cols].mean().round(4)
print("\n=== Trung bình các feature theo cụm ===")
print(cluster_summary)


cluster_summary_path = os.path.join(DATA_PROCESSED_DIR, "cluster_summary.csv")
cluster_summary.to_csv(cluster_summary_path, index=True)
print(f"\nĐã lưu: {cluster_summary_path}")



out_path = os.path.join(DATA_PROCESSED_DIR, "credit_card_customers_with_clusters.csv")
df.to_csv(out_path, index=False)
print(f"Đã lưu: {out_path}")


pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["Cluster"])
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Customer Clusters (PCA 2D)")
plt.savefig(os.path.join(OUTPUTS_DIR, "pca_clusters_2d.png"))
plt.show()
plt.close()


print("\n=== Gợi ý đặt tên cụm (so với trung bình toàn bộ) ===")
overall_mean = df[feature_cols].mean()
diff = (cluster_summary - overall_mean).round(4)
print(diff)
