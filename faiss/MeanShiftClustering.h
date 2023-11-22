/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Implementation of Weighted Blurring Mean Shift clustering as described in https://arxiv.org/pdf/2012.10929.pdf  */

#ifndef FAISS_CLUSTERING_H
#define FAISS_CLUSTERING_H
#include <faiss/Index.h>

#include <vector>

namespace faiss {

/** Class for the clustering parameters. Can be passed to the
 * constructor of the Clustering object.
 */
struct MeanShiftClusteringParameters {
    /// bandwidth used for KDE
    float bandwidth = 0.1;
    /// entropy regularization coefficient for weight update
    float lambda = 10.;
    /// stop when the mean shift magnitude is less than tolerance for all points
    float tolerance = 1e-6;
    /// distance threshold for cluster extraction via connected components (should be larger than threshold)
    float epsilon = 1e-5;
    /// log info during clustering
    bool verbose = false;
};

struct MeanShiftClusteringIterationStats {
    /// seconds for iteration
    double time;
    /// seconds for just search
    double time_search;
};

/** K-means clustering based on assignment - centroid update iterations
 *
 * The clustering is based on an Index object that assigns training
 * points to the centroids. Therefore, at each iteration the centroids
 * are added to the index.
 *
 * On output, the centoids table is set to the latest version
 * of the centroids and they are also added to the index. If the
 * centroids table it is not empty on input, it is also used for
 * initialization.
 *
 */
struct MeanShiftClustering : MeanShiftClusteringParameters {
    /// dimension of the vectors
    size_t d;

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as
     * initialization
     */
    float* centroids;

    /// stats at every iteration of clustering
    std::vector<MeanShiftClusteringIterationStats> iteration_stats;

    MeanShiftClustering(int d);
    MeanShiftClustering(int d, const MeanShiftClusteringParameters& cp);

    /** run mean shift
     *
     * @param n        number of training vectors
     * @param xs        training vectors, size n * d
     * @param index     index used for assignment
     */
    virtual void train(idx_t n, const float* xs, faiss::Index& index);

    void connected_components(idx_t n, const float* xs);

    virtual ~MeanShiftClustering() {}
};

/** simplified interface
 *
 * @param d dimension of the data
 * @param n nb of training vectors
 * @param xs training set (size n * d)
 * @param centroids output centroids (size k * d)
 * @return number of centroids k
 */
size_t mean_shift_clustering(
        size_t d,
        size_t n,
        const float* xs,
        float* centroids);

} // namespace faiss

#endif
