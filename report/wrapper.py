import faiss
import numpy as np


def to_raw(array):
    array = np.ascontiguousarray(array.astype(np.float32))
    # invalid pointer produced when calling swig_ptr on the result of a numpy function call for some reason
    return faiss.swig_ptr(array)


def to_array(raw, shape):
    count = int(np.prod(shape))
    array = faiss.rev_swig_ptr(raw, count)
    return np.reshape(array, shape)


class MeanShift:
    def __init__(
        self,
        X,
        *,
        bandwidth=0.1,
        kernel_radius=0.5,
        weighted_lambda=10,
        tolerance=1e-5,
        epsilon=1e-4,
        weighted=False,
        verbose=False,
        should_generate_stats=True,
        **kwargs,
    ):
        self.n, self.d = X.shape
        self.handle = faiss.MeanShiftClustering(self.d, self.n, to_raw(X))
        for param, value in kwargs.items():
            setattr(self.handle, param, value)

    def run(self):
        self.handle.train()
        self.handle.connected_components()
        Ys = np.reshape(faiss.vector_to_array(self.handle.stats_ys), (-1, self.n, self.d))
        labels = to_array(self.handle.labels, self.n)
        centroid_idxs = faiss.vector_to_array(self.handle.centroids)
        centroids = np.array([Ys[-1][i] for i in centroid_idxs])
        return labels, centroids, Ys
