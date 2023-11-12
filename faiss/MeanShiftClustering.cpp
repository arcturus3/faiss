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
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

MeanShiftClustering::MeanShiftClustering(int d, int k) : d(d), k(k) {}

MeanShiftClustering::MeanShiftClustering(int d, int k, const MeanShiftClusteringParameters& cp)
        : MeanShiftClusteringParameters(cp), d(d), k(k) {}

static double imbalance_factor(int n, int k, int64_t* assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++)
        hist[assign[i]]++;

    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}

void MeanShiftClustering::post_process_centroids() {
    if (spherical) {
        fvec_renorm_L2(d, k, centroids.data());
    }

    if (int_centroids) {
        for (size_t i = 0; i < centroids.size(); i++)
            centroids[i] = roundf(centroids[i]);
    }
}

void MeanShiftClustering::train(
        idx_t nx,
        const float* x_in,
        Index& index,
        const float* weights) {
    train_encoded(
            nx,
            reinterpret_cast<const uint8_t*>(x_in),
            nullptr,
            index,
            weights);
}

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

/** compute centroids as (weighted) sum of training points
 *
 * @param x            training vectors, size n * code_size (from codec)
 * @param codec        how to decode the vectors (if NULL then cast to float*)
 * @param weights      per-training vector weight, size n (or NULL)
 * @param assign       nearest centroid for each training vector, size n
 * @param k_frozen     do not update the k_frozen first centroids
 * @param centroids    centroid vectors (output only), size k * d
 * @param hassign      histogram of assignments per centroid (size k),
 *                     should be 0 on input
 *
 */

void compute_centroids(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        const uint8_t* x,
        const Index* codec,
        const int64_t* assign,
        const float* weights,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    memset(centroids, 0, sizeof(*centroids) * d * k);

    size_t line_size = codec ? codec->sa_code_size() : d * sizeof(float);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;
        std::vector<float> decode_buffer(d);

        for (size_t i = 0; i < n; i++) {
            int64_t ci = assign[i];
            assert(ci >= 0 && ci < k + k_frozen);
            ci -= k_frozen;
            if (ci >= c0 && ci < c1) {
                float* c = centroids + ci * d;
                const float* xi;
                if (!codec) {
                    xi = reinterpret_cast<const float*>(x + i * line_size);
                } else {
                    float* xif = decode_buffer.data();
                    codec->sa_decode(1, x + i * line_size, xif);
                    xi = xif;
                }
                if (weights) {
                    float w = weights[i];
                    hassign[ci] += w;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j] * w;
                    }
                } else {
                    hassign[ci] += 1.0;
                    for (size_t j = 0; j < d; j++) {
                        c[j] += xi[j];
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (idx_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }
        float norm = 1 / hassign[ci];
        float* c = centroids + ci * d;
        for (size_t j = 0; j < d; j++) {
            c[j] *= norm;
        }
    }
}

// a bit above machine epsilon for float16
#define EPS (1 / 1024.)

/** Handle empty clusters by splitting larger ones.
 *
 * It works by slightly changing the centroids to make 2 clusters from
 * a single one. Takes the same arguments as compute_centroids.
 *
 * @return           nb of spliting operations (larger is worse)
 */
int split_clusters(
        size_t d,
        size_t k,
        size_t n,
        size_t k_frozen,
        float* hassign,
        float* centroids) {
    k -= k_frozen;
    centroids += k_frozen * d;

    /* Take care of void clusters */
    size_t nsplit = 0;
    RandomGenerator rng(1234);
    for (size_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0; true; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n - k);
                float r = rng.rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy(centroids + ci * d,
                   centroids + cj * d,
                   sizeof(*centroids) * d);

            /* small symmetric pertubation */
            for (size_t j = 0; j < d; j++) {
                if (j % 2 == 0) {
                    centroids[ci * d + j] *= 1 + EPS;
                    centroids[cj * d + j] *= 1 - EPS;
                } else {
                    centroids[ci * d + j] *= 1 - EPS;
                    centroids[cj * d + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}

}; // namespace

void MeanShiftClustering::train_encoded(
        idx_t nx,
        const uint8_t* x_in,
        const Index* codec,
        Index& index,
        const float* weights) {
    FAISS_THROW_IF_NOT_FMT(
            nx >= k,
            "Number of training points (%" PRId64
            ") should be at least "
            "as large as number of clusters (%zd)",
            nx,
            k);

    FAISS_THROW_IF_NOT_FMT(
            (!codec || codec->d == d),
            "Codec dimension %d not the same as data dimension %d",
            int(codec->d),
            int(d));

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
        for (size_t i = 0; i < nx * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]), "input contains NaN's or Inf's");
        }
    }

    const uint8_t* x = x_in;
    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;

    if (verbose) {
        printf("Clustering %" PRId64
               " points in %zdD to %zd clusters, "
               "redo %d times, %d iterations\n",
               nx,
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

    // temporary buffer to decode vectors during the optimization
    std::vector<float> decode_buffer(codec ? d * decode_block_size : 0);

    // prepare the index

    if (index.ntotal != 0) {
        index.reset();
    }

    if (!index.is_trained) {
        index.train(nx, x);
    }

    index.add(nx, x);

    // mean shift iterations

    std::vector<float> weights(d, 1. / d);
    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);
    std::unique_ptr<float[]> dis(new float[nx]);

    RangeSearchResult search_result;

    float[] next_x = new float[nx];

    // for (int i = 0; i < niter; i++) {
    while (!converged) {
        double t0s = getmillisecs();

        index.range_search(nx, reinterpret_cast<const float*>(x), HUGE_VAL, &search_result);

        InterruptCallback::check();
        t_search_tot += getmillisecs() - t0s;



        float norm = 0;
        float weight_norm = 0;
        // nx == search_result.nq
        for (int i = 0; i < nx; i++) {
            float[] next = new float[d];
            for (int j = search_result.lims[i]; j < search_result.lims[i + 1]; j++) {
                search_result.labels[j];
                search_result.distances[j];
                float coeff = k(search_result.distances[j] / h);
                next += coeff * x[i];
                norm += coeff;
            }
            next_x[i] = next / norm;


            for (int l = 0; l < d; l++) {
                weights[l] += (x_il - y_il) ** 2;
                weights[l] = exp(-1/n/lambda*weights[l]);
                weight_norm += weights[l];
            }
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

        post_process_centroids();

        // add centroids to index for the next iteration (or for output)

        index.reset();
        if (update_index) {
            index.train(k, centroids.data());
        }

        index.add(k, centroids.data());
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
