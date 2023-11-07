import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[ : , [3, 4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(10):
    kmeans = KMeans(n_clusters=(i + 1), random_state=42, init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1, 10+1), wcss)
# plt.title("Elbow method")
# plt.xlabel("number of clusters")
# plt.ylabel("WCSS")
# plt.show()

# Best number found from above is 5
kmeans = KMeans(n_clusters= 5, random_state=42, init='k-means++')
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, color='red', label='cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, color='blue', label='cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, color='green', label='cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, color='cyan', label='cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, color='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color='yellow', label='Centroids')
plt.title("Clusters")
plt.xlabel("Salary")
plt.ylabel("Purchase score")
plt.show()