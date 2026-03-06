# Suppress Windows joblib warnings
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y_true = iris.target  # For reference/evaluation

# Step 1: Determine optimal number of clusters
inertia = []
sil_scores = []
K_range = range(2, 7)  # Test 2 to 6 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, labels))

# Plot Elbow Method
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia (Within-cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")

# Plot Silhouette Scores
plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, 'go-', markersize=8)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different k")
plt.tight_layout()
plt.show()

# Step 2: Apply K-Means with optimal clusters (k=3 for Iris)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Step 3: Evaluate clustering
print(f"Optimal number of clusters: {optimal_k}")
print(f"Inertia: {kmeans.inertia_:.4f}")
sil_score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {sil_score:.4f}")

# Step 4: Visualize clusters (2D projection using first two features)
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis', s=100, alpha=0.6, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering on Iris Dataset (2D projection)")
plt.legend()
plt.show()

# Step 5: Pairplot visualization using Seaborn
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = y_kmeans
sns.pairplot(df, hue='Cluster', palette='viridis', markers=["o","s","D"])
plt.show()
