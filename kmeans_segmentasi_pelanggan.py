
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Pembuatan data dummy
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.00, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Pelatihan model K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 3. Visualisasi hasil clustering
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("Segmentasi Pelanggan dengan K-Means (3 Cluster)")
plt.xlabel("Fitur 1 (Annual Income)")
plt.ylabel("Fitur 2 (Spending Score)")
plt.legend()
plt.grid(True)
plt.show()
