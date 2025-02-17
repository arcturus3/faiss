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

MeanShiftClustering::MeanShiftClustering(size_t d, idx_t n, float* xs) : d(d), n(n), xs(xs) {
    ys = new float[n * d];
    ys1 = new float[n * d];
    if (weighted) {
        yst = new float[n * d];
        weights = new float[d];
    }
    labels = new size_t[n];
    index = IndexFlatL2(d);
}

float k(float x) {
    return exp(-x);
}

void MeanShiftClustering::connected_components() {
    centroids.clear();
    for (int i = 0; i < n; i++) {
        bool assigned = false;
        for (int k = 0; k < centroids.size(); k++) {
            float dist = 0;
            for (int l = 0; l < d; l++) {
                float diff = ys[i * d + l] - ys[centroids[k] * d + l];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist < epsilon) {
                labels[i] = k;
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            labels[i] = centroids.size();
            centroids.push_back(i);
        }
    }
}

void MeanShiftClustering::generate_stats() {
    if (!should_generate_stats) return;
    for (size_t i = 0; i < n * d; i++) {
        stats_ys.push_back(ys[i]);
    }
}

// void MeanShiftClustering::generate_stats() {
//     if (!should_generate_stats) return;
//     MeanShiftClusteringIterationStats stat;
//     stat.ys = new float[n * d];
//     memcpy(stat.ys, ys, n * d * sizeof(float));
//     if (weighted) {
//         stat.weights = new float[d];
//         memcpy(stat.weights, weights, d * sizeof(float));
//     }
//     stats.push_back(stat);
// }

void MeanShiftClustering::train() {
    memcpy(ys, xs, n * d * sizeof(float));

    if (weighted) {
        for (size_t l = 0; l < d; l++)  weights[l] = 1. / d;
    }

    bool converged = false;
    generate_stats();
    while (!converged) {
        if (weighted) {
            for (idx_t i = 0; i < n; i++) {
                for (size_t l = 0; l < d; l++) {
                    yst[i * d + l] = sqrt(weights[l]) * ys[i * d + l];
                }
            }
        }

        index.reset();
        index.add(n, weighted ? yst : ys);
        RangeSearchResult search_result(n);
        index.range_search(n, weighted ? yst : ys, kernel_radius, &search_result);

        if (weighted) {
            memset(weights, 0, d * sizeof(float));
        }

        memset(ys1, 0, n * d * sizeof(float));
        converged = true;
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

                if (weighted) {
                    float disp = xs[i * d + l] - ys[i * d + l];
                    weights[l] += disp * disp;
                }

                float shift_d = ys1[i * d + l] - ys[i * d + l];
                shift += shift_d * shift_d;
            }
            shift = sqrt(shift);
            converged &= shift < tolerance;
        }

        if (weighted) {
            float weight_norm = 0;
            for (int l = 0; l < d; l++) {
                weights[l] = exp(-weights[l] / n / lambda);
                weight_norm += weights[l];
            }
            for (int l = 0; l < d; l++) {
                weights[l] /= weight_norm;
            }
        }

        float* temp = ys;
        ys = ys1;
        ys1 = temp;
        generate_stats();
    }
}

} // namespace faiss
