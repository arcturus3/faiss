import os.path
import clustbench
from sklearn.cluster import estimate_bandwidth
import seaborn.objects as so
import numpy as np
from wrapper import MeanShift


path = os.path.join('~', 'clustering-data-v1')
dataset = clustbench.load_dataset('wut', 'x3', path=path)
rng = np.random.default_rng()
X = rng.choice(dataset.data, size=100)
# X = dataset.data

bandwidth = estimate_bandwidth(X)
mean_shift = MeanShift(X, bandwidth=bandwidth)
labels, centroids, Ys = mean_shift.run()

# labels += 1 # clustbench requires labels between 1 and k
# k_pred = centroids.shape[0]
# assert k_pred in dataset.n_clusters # scoring will error if k_pred is not k_true in any of the reference clusterings
# score = clustbench.get_score(dataset.labels, labels)

data = {
    'iter': [],
    'x': [],
    'y': [],
    'label': [],
}
for i in range(Ys.shape[0]):
    for j in range(Ys.shape[1]):
        data['iter'].append(i)
        data['x'].append(Ys[i,j,0])
        data['y'].append(Ys[i,j,1])
        data['label'].append(labels[j])

(so.Plot(data, x='x', y='y', color='label')
.facet('iter', wrap=4)
.add(so.Dot(), legend=False)
.label(x='', y='', col='Iteration')
.theme({'axes.facecolor': '#ffffff', 'axes.edgecolor': '#000000'})
.show())
