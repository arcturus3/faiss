import os.path
import clustbench
import faiss
import numpy as np
import sklearn.cluster
import seaborn as sns
import matplotlib.pyplot as plt
from wrapper import MeanShift


path = os.path.join('~', 'clustering-data-v1')
dataset = clustbench.load_dataset('wut', 'x1', path=path)

mean_shift = MeanShift(dataset.data)
labels, centroids, Ys = mean_shift.run()

labels += 1 # clustbench requires labels between 1 and k
k_pred = centroids.shape[0]
assert k_pred in dataset.n_clusters # scoring will error if k_pred is not k_true in any of the reference clusterings

score = clustbench.get_score(dataset.labels, labels)

# print(k_pred)
# print(labels)
# print(centroids)
# print(Ys)
# print(score)

# sns.set_theme()
# sns.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels)
# plt.show()
