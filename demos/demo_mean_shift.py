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

mean_shift = faiss.MeanShiftClustering(d, n, X_ptr)
mean_shift.train()
mean_shift.connected_components()

Y = np.reshape(faiss.rev_swig_ptr(mean_shift.ys, n * d), (n, d))
centroids = faiss.vector_to_array(mean_shift.centroids)
labels = faiss.rev_swig_ptr(mean_shift.labels, n)

print('centroids:')
print(np.array([Y[i] for i in centroids]))
print('labels:')
print(labels)
