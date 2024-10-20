import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 加載 Mall Customers 資料集
df = pd.read_csv('Mall_Customers.csv')

# 2. 選擇需要的特徵進行分群
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 使用 KMeans++ 進行分群
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. 視覺化聚類結果
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='rainbow', edgecolor='k', s=100)
plt.title('K-Means Clustering on Mall Customers Dataset')
plt.xlabel('Standardized Annual Income (k$)')
plt.ylabel('Standardized Spending Score (1-100)')
plt.show()

# 6. 打印分群結果
print(df[['CustomerID', 'cluster']].head())
