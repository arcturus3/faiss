import os.path
import clustbench
import faiss
import numpy as np
import sklearn.cluster


def run(X):
    X = np.ascontiguousarray(X.astype(np.float32))
    n, d = X.shape
    # invalid pointer produced when calling swig_ptr on the result of a numpy function call for some reason
    X_ptr = faiss.swig_ptr(X)

    bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=0.2, n_samples=500)

    ms = faiss.MeanShiftClustering(d, n, X_ptr)
    # ms.bandwidth = bandwidth
    ms.bandwidth = 0.5
    ms.train()
    ms.connected_components()

    Y = np.reshape(faiss.rev_swig_ptr(ms.ys, n * d), (n, d))
    labels = faiss.rev_swig_ptr(ms.labels, n)
    centroids = np.array([Y[i] for i in faiss.vector_to_array(ms.centroids)])

    return labels, centroids


path = os.path.join('~', 'clustering-data-v1')
dataset = clustbench.load_dataset('wut', 'x3', path=path)

labels, centroids = run(dataset.data)
labels += 1 # clustbench requires labels between 1 and k
k_pred = centroids.shape[0]
print(k_pred)
assert(k_pred in dataset.n_clusters) # scoring will error if k_pred is not k_true in any of the reference clusterings

print(centroids)
print(labels)

score = clustbench.get_score(dataset.labels, labels)
print(score)
