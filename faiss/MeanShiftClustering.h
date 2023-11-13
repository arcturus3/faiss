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
    /// number of clustering iterations
    int niter = 25;
    /// redo clustering this many times and keep the clusters with the best
    /// objective
    int nredo = 1;

    bool verbose = false;
    /// whether to normalize centroids after each iteration (useful for inner
    /// product clustering)
    bool spherical = false;
    /// round centroids coordinates to integer after each iteration?
    bool int_centroids = false;
    /// re-train index after each iteration?
    bool update_index = false;

    /// Use the subset of centroids provided as input and do not change them
    /// during iterations
    bool frozen_centroids = false;
    /// If fewer than this number of training vectors per centroid are provided,
    /// writes a warning. Note that fewer than 1 point per centroid raises an
    /// exception.
    int min_points_per_centroid = 39;
    /// to limit size of dataset, otherwise the training set is subsampled
    int max_points_per_centroid = 256;
    /// seed for the random number generator
    int seed = 1234;

    /// when the training set is encoded, batch size of the codec decoder
    size_t decode_block_size = 32768;
};

struct MeanShiftClusteringIterationStats {
    float obj;   ///< objective values (sum of distances reported by index)
    double time; ///< seconds for iteration
    double time_search;      ///< seconds for just search
    double imbalance_factor; ///< imbalance factor of iteration
    int nsplit;              ///< number of cluster splits
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
    size_t d; ///< dimension of the vectors

    float h = 0.1;
    float lambda = 10;

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as
     * initialization
     */
    std::vector<float> centroids;

    /// stats at every iteration of clustering
    std::vector<MeanShiftClusteringIterationStats> iteration_stats;

    MeanShiftClustering(int d, int k);
    MeanShiftClustering(int d, int k, const MeanShiftClusteringParameters& cp);

    /** run mean shift
     *
     * @param nx        number of training vectors
     * @param xs        training vectors, size nx * d
     * @param index     index used for assignment
     */
    virtual void train(idx_t nx, const float* xs, faiss::Index& index);

    virtual ~MeanShiftClustering() {}
};

/** simplified interface
 *
 * @param d dimension of the data
 * @param n nb of training vectors
 * @param k nb of output centroids
 * @param x training set (size n * d)
 * @param centroids output centroids (size k * d)
 * @return final quantization error
 */
float mean_shift_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids);

} // namespace faiss

#endif
