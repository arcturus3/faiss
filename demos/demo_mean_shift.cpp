/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <memory>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

#include <faiss/MeanShiftClustering.h>

namespace {

enum WeightedKMeansType {
    WKMT_FlatL2,
    WKMT_FlatIP,
    WKMT_FlatIP_spherical,
    WKMT_HNSW,
};

float weighted_kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* input,
        const float* weights,
        float* centroids,
        WeightedKMeansType index_num) {
    using namespace faiss;
    Clustering clus(d, k);
    clus.verbose = true;

    std::unique_ptr<Index> index;

    switch (index_num) {
        case WKMT_FlatL2:
            index = std::make_unique<IndexFlatL2>(d);
            break;
        case WKMT_FlatIP:
            index = std::make_unique<IndexFlatIP>(d);
            break;
        case WKMT_FlatIP_spherical:
            index = std::make_unique<IndexFlatIP>(d);
            clus.spherical = true;
            break;
        case WKMT_HNSW:
            IndexHNSWFlat* ihnsw = new IndexHNSWFlat(d, 32);
            ihnsw->hnsw.efSearch = 128;
            index.reset(ihnsw);
            break;
    }

    clus.train(n, input, *index.get(), weights);
    // on output the index contains the centroids.
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

int d = 32;
float sigma = 0.1;

#define BIGTEST

#ifdef BIGTEST
// the production setup = setting of https://fb.quip.com/CWgnAAYbwtgs
int nc = 200000;
int n_big = 4;
int n_small = 2;
#else
int nc = 5;
int n_big = 100;
int n_small = 10;
#endif

int n; // number of training points

void generate_dataset(size_t d, size_t n, std::vector<float>& xs) {
    xs.resize(n * d);
    for (size_t i = 0; i < n; i++) {
        for (size_t l = 0; l < d; l++) {
            xs[i * d + l] = i & 1;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t d = 2;
    faiss::idx_t n = 10;
    std::vector<float> xs;

    printf("generate training set\n");
    generate_dataset(d, n, xs);

    using namespace faiss;
    MeanShiftClustering mean_shift(d, n, xs.data());
    mean_shift.train();
    mean_shift.connected_components();

    for (size_t i = 0; i < mean_shift.centroids.size(); i++) {
        printf("cluster %d: [", i);
        for (size_t l = 0; l < d - 1; l++) {
            printf("%d, ", mean_shift.ys[mean_shift.centroids[i] * d + l]);
        }
        printf("%d]\n", mean_shift.ys[mean_shift.centroids[i] * d + d - 1]);
    }

    for (size_t i = 0; i < n; i++) {
        printf("label %d: %d\n", i, mean_shift.labels[i]);
    }

    // int the_index_num = -1;
    // int the_with_weights = -1;

    // if (argc == 3) {
    //     the_index_num = atoi(argv[1]);
    //     the_with_weights = atoi(argv[2]);
    // }

    // for (int index_num = WKMT_FlatL2; index_num <= WKMT_HNSW; index_num++) {
    //     if (the_index_num >= 0 && index_num != the_index_num) {
    //         continue;
    //     }

    //     for (int with_weights = 0; with_weights <= 1; with_weights++) {
    //         if (the_with_weights >= 0 && with_weights != the_with_weights) {
    //             continue;
    //         }

    //         printf("=================== index_num=%d Run %s weights\n",
    //                index_num,
    //                with_weights ? "with" : "without");

    //         weighted_kmeans_clustering(
    //                 d,
    //                 n,
    //                 nc,
    //                 x.data(),
    //                 with_weights ? weights.data() : nullptr,
    //                 centroids.data(),
    //                 (WeightedKMeansType)index_num);

    //         { // compute distance of points to centroids
    //             faiss::IndexFlatL2 cent_index(d);
    //             cent_index.add(nc, centroids.data());
    //             std::vector<float> dis(n);
    //             std::vector<faiss::idx_t> idx(n);

    //             cent_index.search(
    //                     nc * 2, ccent.data(), 1, dis.data(), idx.data());

    //             float dis1 = 0, dis2 = 0;
    //             for (int i = 0; i < nc; i++) {
    //                 dis1 += dis[i];
    //             }
    //             printf("average distance of points from big clusters: %g\n",
    //                    dis1 / nc);

    //             for (int i = 0; i < nc; i++) {
    //                 dis2 += dis[i + nc];
    //             }

    //             printf("average distance of points from small clusters: %g\n",
    //                    dis2 / nc);
    //         }
    //     }
    // }
    return 0;
}
