# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data
# Step 3: Create dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()
# Step 4: Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean',
linkage='ward')
labels = hc.fit_predict(X)
# Step 5: Evaluate clustering
sil_score = silhouette_score(X, labels)
print("Cluster Labels:\n", labels)
print("\nSilhouette Score:", sil_score)