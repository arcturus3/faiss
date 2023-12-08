import os.path
import clustbench
import faiss
import numpy as np
import sklearn.cluster
import seaborn as sns
import matplotlib.pyplot as plt
from wrapper import MeanShift


def to_raw(array):
    array = np.ascontiguousarray(array.astype(np.float32))
    # invalid pointer produced when calling swig_ptr on the result of a numpy function call for some reason
    return faiss.swig_ptr(array)


def to_array(raw, shape):
    count = int(np.prod(shape))
    array = faiss.rev_swig_ptr(raw, count)
    return np.reshape(array, shape)


def run(X):
    n, d = X.shape
    bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=0.2, n_samples=500)

    ms = faiss.MeanShiftClustering(d, n, to_raw(X))
    ms.bandwidth = bandwidth
    ms.train()
    ms.connected_components()

    Y = to_array(ms.ys, (n, d))
    labels = to_array(ms.labels, n)
    centroids = np.array([Y[i] for i in faiss.vector_to_array(ms.centroids)])

    print(ms.stats)

    return labels, centroids, Y


path = os.path.join('~', 'clustering-data-v1')
dataset = clustbench.load_dataset('wut', 'x1', path=path)

mean_shift = MeanShift(dataset.data)
labels, centroids, Ys = mean_shift.run()

labels += 1 # clustbench requires labels between 1 and k
k_pred = centroids.shape[0]
print(k_pred)
assert k_pred in dataset.n_clusters # scoring will error if k_pred is not k_true in any of the reference clusterings

print(labels)
print(centroids)
print(Ys)

score = clustbench.get_score(dataset.labels, labels)
print(score)


# sns.set_theme()
# sns.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels)
# plt.show()
