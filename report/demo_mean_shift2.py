# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import faiss

d = 32
n = 200

X = np.random.standard_normal((n, d)).astype(np.float32)
X[n // 2:,:2] += 5
X -= np.mean(X, 0)
X /= np.std(X, 0)
X_ptr = faiss.swig_ptr(np.ascontiguousarray(X))

ms = faiss.MeanShiftClustering(d, n, X_ptr)
ms.train()
ms.connected_components()

Y = np.reshape(faiss.rev_swig_ptr(ms.ys, n * d), (n, d))
labels = faiss.rev_swig_ptr(ms.labels, n)
centroids = faiss.vector_to_array(ms.centroids)
cluster_centers = np.array([Y[i] for i in centroids])

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

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
