/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/MeanShiftClustering.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

MeanShiftClustering::MeanShiftClustering(int d) : d(d) {}

MeanShiftClustering::MeanShiftClustering(int d, const MeanShiftClusteringParameters& cp)
        : MeanShiftClusteringParameters(cp), d(d) {}

// double check this?
float k(float x) {
    // return x <= 1. ? 1. : 0.;
    return exp(-x * x);
}

void MeanShiftClustering::connected_components(idx_t n, const float* xs) {
    std::vector<float*> centroids;
    // labels = ?
    for (int i = 0; i < n; i++) {
        bool assigned = false;
        for (int k = 0; k < centroids.size(); k++) {
            if (d(xs[i], centroids[k]) < epsilon) {
                labels[i] = k;
                break;
            }
        }
        if (!assigned) {
            centroids.push_back(&xs[i * d]);
        }
    }
}

void MeanShiftClustering::train(idx_t n, const float* xs, Index& index) {
    FAISS_THROW_IF_NOT_FMT(
            index.d == d,
            "Index dimension %d not the same as data dimension %d",
            int(index.d),
            int(d));

    double t0 = getmillisecs();

    // if (verbose) {
    //     printf("Clustering %" PRId64
    //            " points in %zdD to %zd clusters, "
    //            "redo %d times, %d iterations\n",
    //            n,
    //            d,
    //            k,
    //            nredo,
    //            niter);
    // }


    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n", (getmillisecs() - t0) / 1000.);
    }
    t0 = getmillisecs();

    // mean shift iterations

    std::vector<float> weights(d, 1. / d);
    std::unique_ptr<idx_t[]> assign(new idx_t[n]);
    std::unique_ptr<float[]> dis(new float[n]);

    RangeSearchResult search_result(n);

    float* ys = new float[n * d];
    float* ys1 = new float[n * d];
    memcpy(ys, xs, n * d * sizeof(float));

    bool converged = false;
    while (!converged) {
        double t0s = getmillisecs();

        // perform transform
        index.reset();
        index.train(n, ys);
        index.add(n, ys);
        index.range_search(n, ys, HUGE_VAL, &search_result);

        InterruptCallback::check();
        t_search_tot += getmillisecs() - t0s;

        converged = true;
        // reset weights to 0
        for (int i = 0; i < n; i++) {
            float y1_norm = 0;
            for (int j = search_result.lims[i]; j < search_result.lims[i + 1]; j++) {
                float coeff = k(search_result.distances[j] / bandwidth);
                y1_norm += coeff;
                for (int l = 0; l < d; l++) {
                    ys1[i * d + l] += coeff * ys[search_result.labels[j] * d + l];
                }
            }
            float shift = 0;
            for (int l = 0; l < d; l++) {
                ys1[i * d + l] /= y1_norm;
                float disp = xs[i * d + l] - ys[i * d + l];
                weights[l] += disp * disp;
                float shift_d = ys1[i * d + l] - ys[i * d + l];
                shift += shift_d * shift_d;
            }
            shift = sqrt(shift);
            converged &= shift < tolerance;
        }
        float weight_norm = 0;
        for (int l = 0; l < d; l++) {
            weights[l] = exp(-weights[l] / n / lambda);
            weight_norm += weights[l];
        }
        for (int l = 0; l < d; l++) {
            weights[l] /= weight_norm;
        }

        // collect statistics
        // MeanShiftClusteringIterationStats stats = {
        //         obj,
        //         (getmillisecs() - t0) / 1000.0,
        //         t_search_tot / 1000,
        //         imbalance_factor(nx, k, assign.get()),
        //         nsplit};
        // iteration_stats.push_back(stats);

        // if (verbose) {
        //     printf("  Iteration %d (%.2f s, search %.2f s): "
        //             "objective=%g imbalance=%.3f nsplit=%d       \r",
        //             i,
        //             stats.time,
        //             stats.time_search,
        //             stats.obj,
        //             stats.imbalance_factor,
        //             nsplit);
        //     fflush(stdout);
        // }

        InterruptCallback::check();
    }

    if (verbose)
        printf("\n");
}

size_t mean_shift_clustering(
    size_t d,
    size_t n,
    const float* xs,
    float* centroids
) {
    MeanShiftClustering clus(d);
    IndexFlatL2 index(d);
    clus.train(n, xs, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

} // namespace faiss
