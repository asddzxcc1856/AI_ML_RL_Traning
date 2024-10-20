import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 模擬客戶數據
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'purchase_amount': [2000, 500, 10000, 800, 3000],
    'purchase_frequency': [20, 5, 50, 10, 25],
    'product_views': [150, 60, 500, 120, 200]
}

df = pd.DataFrame(data)

# 2. 選擇需要的特徵並進行標準化
features = ['purchase_amount', 'purchase_frequency', 'product_views']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 使用 KMeans++ 進行分群
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 4. 輸出分群結果
print("客戶分群結果：")
print(df)

# 5. 視覺化結果
# 使用兩個主成分進行簡單可視化
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('Normalized Purchase Amount')
plt.ylabel('Normalized Purchase Frequency')
plt.title('Customer Clusters')
plt.show()
