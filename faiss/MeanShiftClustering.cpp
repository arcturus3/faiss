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

MeanShiftClustering::MeanShiftClustering(int d, int k) : d(d), k(k) {}

MeanShiftClustering::MeanShiftClustering(int d, int k, const MeanShiftClusteringParameters& cp)
        : MeanShiftClusteringParameters(cp), d(d), k(k) {}

namespace {

idx_t subsample_training_set(
        const MeanShiftClustering& clus,
        idx_t nx,
        const uint8_t* x,
        size_t line_size,
        const float* weights,
        uint8_t** x_out,
        float** weights_out) {
    if (clus.verbose) {
        printf("Sampling a subset of %zd / %" PRId64 " for training\n",
               clus.k * clus.max_points_per_centroid,
               nx);
    }
    std::vector<int> perm(nx);
    rand_perm(perm.data(), nx, clus.seed);
    nx = clus.k * clus.max_points_per_centroid;
    uint8_t* x_new = new uint8_t[nx * line_size];
    *x_out = x_new;
    for (idx_t i = 0; i < nx; i++) {
        memcpy(x_new + i * line_size, x + perm[i] * line_size, line_size);
    }
    if (weights) {
        float* weights_new = new float[nx];
        for (idx_t i = 0; i < nx; i++) {
            weights_new[i] = weights[perm[i]];
        }
        *weights_out = weights_new;
    } else {
        *weights_out = nullptr;
    }
    return nx;
}

}; // namespace

float k(float x) {
    return x <= 1. ? 1. : 0.;
}

void MeanShiftClustering::train(idx_t n, const float* xs, Index& index) {
    FAISS_THROW_IF_NOT_FMT(
            index.d == d,
            "Index dimension %d not the same as data dimension %d",
            int(index.d),
            int(d));

    double t0 = getmillisecs();

    if (!codec) {
        // Check for NaNs in input data. Normally it is the user's
        // responsibility, but it may spare us some hard-to-debug
        // reports.
        const float* x = reinterpret_cast<const float*>(x_in);
        for (size_t i = 0; i < n * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]), "input contains NaN's or Inf's");
        }
    }

    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;

    if (verbose) {
        printf("Clustering %" PRId64
               " points in %zdD to %zd clusters, "
               "redo %d times, %d iterations\n",
               n,
               d,
               k,
               nredo,
               niter);
        if (codec) {
            printf("Input data encoded in %zd bytes per vector\n",
                   codec->sa_code_size());
        }
    }


    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n", (getmillisecs() - t0) / 1000.);
    }
    t0 = getmillisecs();

    // mean shift iterations

    std::vector<float> weights(d, 1. / d);
    std::unique_ptr<idx_t[]> assign(new idx_t[n]);
    std::unique_ptr<float[]> dis(new float[n]);

    RangeSearchResult search_result;

    float* ys = new float[n * d];
    float* ys1 = new float[n * d];
    memcpy(ys, xs, n * d * sizeof(float));

    while (!converged) {
        double t0s = getmillisecs();

        index.reset();
        index.train(n, ys);
        index.add(n, ys);
        index.range_search(n, ys, HUGE_VAL, &search_result);

        InterruptCallback::check();
        t_search_tot += getmillisecs() - t0s;

        // reset weights to 0
        for (int i = 0; i < n; i++) {
            float y1_norm = 0;
            for (int j = search_result.lims[i]; j < search_result.lims[i + 1]; j++) {
                float coeff = k(search_result.distances[j] / h);
                y1_norm += coeff;
                for (int l = 0; l < d; l++) {
                    ys1[i * d + l] += coeff * ys[search_result.labels[j] * d + l];
                }
            }
            for (int l = 0; l < d; l++) {
                ys1[i * d + l] /= y1_norm;
            }
            for (int l = 0; l < d; l++) {
                float disp = xs[i * d + l] - ys[i * d + l];
                weights[l] += disp * disp;
            }
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
        MeanShiftClusteringIterationStats stats = {
                obj,
                (getmillisecs() - t0) / 1000.0,
                t_search_tot / 1000,
                imbalance_factor(nx, k, assign.get()),
                nsplit};
        iteration_stats.push_back(stats);

        if (verbose) {
            printf("  Iteration %d (%.2f s, search %.2f s): "
                    "objective=%g imbalance=%.3f nsplit=%d       \r",
                    i,
                    stats.time,
                    stats.time_search,
                    stats.obj,
                    stats.imbalance_factor,
                    nsplit);
            fflush(stdout);
        }

        InterruptCallback::check();
    }

    if (verbose)
        printf("\n");
}

float mean_shift_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids) {
    MeanShiftClustering clus(d, k);
    clus.verbose = d * n * k > (size_t(1) << 30);
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index(d);
    clus.train(n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

namespace {

void copy_columns(idx_t n, idx_t d1, const float* src, idx_t d2, float* dest) {
    idx_t d = std::min(d1, d2);
    for (idx_t i = 0; i < n; i++) {
        memcpy(dest, src, sizeof(float) * d);
        src += d1;
        dest += d2;
    }
}

}; // namespace

} // namespace faiss
