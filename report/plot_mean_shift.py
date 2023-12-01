"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""

import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

import faiss

# %%
# Generate sample data
# --------------------
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.4)

# %%
# Compute clustering with MeanShift
# ---------------------------------

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

X = X.astype(np.float32)
n, d = X.shape
X_ptr = faiss.swig_ptr(np.ravel(X))

ms = faiss.MeanShiftClustering(d, n, X_ptr)
ms.bandwidth = bandwidth
ms.train()
ms.connected_components()

Y = np.reshape(faiss.rev_swig_ptr(ms.ys, n * d), (n, d))
labels = faiss.rev_swig_ptr(ms.labels, n)
centroids = faiss.vector_to_array(ms.centroids)
cluster_centers = np.array([Y[i] for i in centroids])

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# %%
# Plot result
# -----------
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()

colors = ["#dede00", "#377eb8", "#f781bf", "#0000ff", "#ff0000", "#00ff00"]
markers = ["x", "o", "^", "+", "+", "+"]

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
