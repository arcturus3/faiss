/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Implementation of Weighted Blurring Mean Shift clustering as described in https://arxiv.org/pdf/2012.10929.pdf  */

#ifndef FAISS_MEAN_SHIFT_CLUSTERING_H
#define FAISS_MEAN_SHIFT_CLUSTERING_H
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <vector>

namespace faiss {

struct MeanShiftClusteringIterationStats {
    float* ys;
    float* weights;
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
struct MeanShiftClustering {
    /// bandwidth used for KDE
    float bandwidth = 0.1;
    /// set to HUGE_VAL for unbounded kernel
    float kernel_radius=0.5;
    /// entropy regularization coefficient for weight update
    float lambda = 10.;
    /// stop when the mean shift magnitude is less than tolerance for all points
    float tolerance = 1e-5;
    /// distance threshold for cluster extraction via connected components (should be larger than threshold)
    float epsilon = 1e-4;
    /// do weighted blurring mean shift
    bool weighted = false;
    /// log info during clustering
    bool verbose = false;

    bool should_generate_stats = true;

    /// dimension of vectors
    size_t d;
    /// number of vectors
    idx_t n;
    /// original vectors, size n * d
    float* xs;
    /// resulting vectors, size n * d
    float* ys;
    /// intermediate vectors during clustering, size n * d
    float* ys1;
    /// transformed vectors during clustering, size n * d
    float* yst;
    /// weight vector used during clustering, size d
    float* weights;
    /// centroids represented as indexes to ys
    std::vector<size_t> centroids;
    /// assignment to centroids represented as indexes to centroids, size n
    size_t* labels;
    /// index used during clustering
    IndexFlatL2 index;
    std::vector<MeanShiftClusteringIterationStats> stats;
    /// workaround for issues generating MeanShiftClusteringIterationStatsVector interface
    std::vector<float> stats_ys;

    MeanShiftClustering(size_t d, idx_t n, float* xs);

    /** run mean shift
     *
     * @param n        number of training vectors
     * @param xs        training vectors, size n * d
     * @param index     index used for assignment
     */
    void train();

    void connected_components();

    void generate_stats();

    virtual ~MeanShiftClustering() {}
};

} // namespace faiss

#endif
