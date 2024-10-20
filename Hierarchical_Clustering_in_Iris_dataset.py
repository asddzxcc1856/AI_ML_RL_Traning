import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# 1. 加載 Iris 資料集
iris = load_iris()
X = iris.data

# 2. 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用 linkage 方法進行凝聚層次聚類 (使用歐氏距離和平均連結)
Z = linkage(X_scaled, method='average')

# 4. 繪製樹狀圖
plt.figure(figsize=(32, 32))
dendrogram(Z, labels=iris.target, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (Iris Dataset)')
plt.xlabel('Sample Index')
plt.ylabel('Distance (Average Linkage)')
plt.show()

# 5. 使用 AgglomerativeClustering 進行分群 (設定群數為 3)
clustering = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = clustering.fit_predict(X_scaled)

# 6. 打印分群結果
print("分群結果：", labels)
