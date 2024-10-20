import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 1. 加載 Mall Customers 資料集
df = pd.read_csv('Mall_Customers.csv')

# 2. 選擇需要的特徵進行分群
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 使用 linkage 進行凝聚層次聚類 (使用歐氏距離和平均連結)
Z = linkage(X_scaled, method='average')

# 5. 繪製樹狀圖
plt.figure(figsize=(32, 32))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram (Mall Customers Dataset)')
plt.xlabel('Customer Index')
plt.ylabel('Distance (Average Linkage)')
plt.show()

# 6. 使用 AgglomerativeClustering 進行分群 (設定群數為 5)
clustering = AgglomerativeClustering(n_clusters=5, linkage='average')
df['cluster'] = clustering.fit_predict(X_scaled)

# 7. 打印分群結果
print(df[['CustomerID', 'cluster']].head())
