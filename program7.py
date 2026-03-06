# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Step 2: Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 4: Apply PCA (reduce to 2 principal components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Step 5: Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Retained:", sum(pca.explained_variance_ratio_))
# Step 6: View transformed data
print("\nTransformed Data Shape:", X_pca.shape)
# Step 7: PCA Graph
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Wine Dataset")
plt.show()