"""
Lab 1 - Principal Component Analysis (PCA)
- Standardize data
- Perform PCA for dimensionality reduction
- Scree plot
- 2D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load data
d = load_iris()
X = d.data
y = d.target

# Standardize
sc = StandardScaler()
X_s = sc.fit_transform(X)

# PCA - all components for scree plot
pca = PCA()
pca.fit(X_s)

# Scree plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.tight_layout()
plt.savefig('scree_plot.png')
plt.show()

# PCA - 2 components for visualization
pca2 = PCA(n_components=2)
X_r = pca2.fit_transform(X_s)

# 2D scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(X_r[:, 0], X_r[:, 1], c=y, cmap='viridis', edgecolors='k', s=60)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - 2D Visualization (Iris Dataset)')
plt.colorbar(label='Class')
plt.tight_layout()
plt.savefig('pca_2d.png')
plt.show()

print(f"Variance explained by PC1: {pca2.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2: {pca2.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total variance captured:   {sum(pca2.explained_variance_ratio_)*100:.2f}%")
