import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[ : , [3, 4]].values

import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()

# no. of clusters come from the test we did on top (through plot) to find what is the best treshold for agglomeration
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, linkage='ward', affinity='euclidean')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, color='red', label='cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, color='blue', label='cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, color='green', label='cluster3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, color='cyan', label='cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, color='magenta', label='cluster5')
plt.title("Clusters")
plt.xlabel("Salary")
plt.ylabel("Purchase score")
plt.show()