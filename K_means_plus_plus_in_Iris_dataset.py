import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 加載 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target

# 2. 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用 KMeans++ 進行分群
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=100, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# 4. 使用 PCA 將數據降至 2 維進行可視化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. 視覺化聚類結果
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 6. 打印分群結果和實際類別進行比較
print("聚類結果：", labels)
print("實際類別：", y)
