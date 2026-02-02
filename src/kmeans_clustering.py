import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- Cấu hình ---
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# Đường dẫn (Tự động lấy theo vị trí file script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credit_card_customers.csv")

# Tạo thư mục output
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", f"run_{timestamp}")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

for d in [FIGURES_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

def load_data(path):
    print(f"Loading data from: {path}")
    if not os.path.exists(path):
         # Thử tìm file csv bất kỳ nếu file mặc định không có
        raw_dir = os.path.dirname(path)
        csvs = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if csvs:
            path = os.path.join(raw_dir, csvs[0])
            print(f"File not found, checking alternative: {path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file data tại {path}")
            
    return pd.read_csv(path)

def preprocess_data(df):
    print("Preprocessing data...")
    # Loại bỏ cột không dùng
    drop_cols = ["CUST_ID", "CustomerID", "CUSTOMER_ID", "TENURE"]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Lấy cột số và điền dữ liệu thiếu
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])
    
    return X_scaled, numeric_cols, df_clean

def find_optimal_k(X_scaled):
    print("Running Elbow & Silhouette method (K=2..10)...")
    Ks = range(2, 11)
    inertias = []
    sil_scores = []

    for k in Ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Vẽ Elbow
    plt.figure()
    plt.plot(Ks, inertias, 'bo-')
    plt.title("Elbow Method")
    plt.savefig(os.path.join(FIGURES_DIR, "elbow_method.png"))
    plt.close()

    # Vẽ Silhouette
    plt.figure()
    plt.plot(Ks, sil_scores, 'ro-')
    plt.title("Silhouette Scores")
    plt.savefig(os.path.join(FIGURES_DIR, "silhouette_scores.png"))
    plt.close()
    print(f"Saved charts to {FIGURES_DIR}")

def visualize_results(df, X_scaled, summary):
    print("Generating visualizations...")
    
    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    plt.figure()
    plt.scatter(X_2d[:,0], X_2d[:,1], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.title("PCA Clusters 2D")
    plt.savefig(os.path.join(FIGURES_DIR, "pca_clusters_2d.png"))
    plt.close()

    # 2. Bar Chart
    plt.figure()
    counts = df['Cluster'].value_counts().sort_index()
    counts.plot(kind='bar', color='teal')
    plt.title("Customer Count per Cluster")
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_counts.png"))
    plt.close()
    
    # 3. Radar Chart (Simplified)
    try:
        features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']
        features = [f for f in features if f in summary.columns]
        
        # Normalize 0-1
        data_norm = summary[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        plt.figure(figsize=(8,8))
        ax = plt.subplot(111, polar=True)
        
        for i, row in data_norm.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, label=f'Cluster {i}')
            ax.fill(angles, values, alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.title("Cluster Characteristics (Normalized)")
        plt.savefig(os.path.join(FIGURES_DIR, "radar_chart.png"))
        plt.close()
    except Exception as e:
        print(f"Skipping Radar Chart: {e}")

def main():
    # 1. Load và Xử lý
    df = load_data(DATA_PATH)
    X_scaled, feature_cols, df_features = preprocess_data(df)
    
    # 2. Tìm K tối ưu (tùy chọn, uncomment để chạy)
    # find_optimal_k(X_scaled)

    # 3. Chạy K-Means (K=4 cố định theo bài toán)
    k = 4
    print(f"Clustering with K={k}...")
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    df['Cluster'] = km.fit_predict(X_scaled)
    
    # 4. Phân tích & Lưu
    print("Saving results...")
    summary = df.groupby('Cluster')[feature_cols].mean().round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, "cluster_summary.csv"))
    summary.to_csv(os.path.join(PROCESSED_DATA_DIR, "cluster_summary.csv")) # Save latest
    
    df.to_csv(os.path.join(RESULTS_DIR, "clustered_data.csv"), index=False)
    
    print("\nCluster Counts:")
    print(df['Cluster'].value_counts().sort_index())
    
    # 5. Vẽ biểu đồ
    visualize_results(df, X_scaled, summary)
    
    print("\nDone! Results saved to:")
    print(f"- Reports: {RESULTS_DIR}")
    print(f"- Processed Data: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
