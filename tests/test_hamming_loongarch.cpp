/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/utils/hamming.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
using namespace faiss;
void check_hamming(SIMDLevel level) {
    if (!SIMDConfig::is_simd_level_available(level))
        GTEST_SKIP();
    struct Guard {
        SIMDLevel saved = SIMDConfig::get_level();
        ~Guard() {
            SIMDConfig::set_level(saved);
        }
    } guard;
    SIMDConfig::set_level(level);
    EXPECT_EQ(with_hamming_simd_level([]<SIMDLevel L>() { return L; }), level);
    std::mt19937 rng(99);
    for (size_t size : {1,  3,  4,  7,  8,  9,  15, 16, 17,  20,
                        24, 31, 32, 33, 40, 63, 64, 65, 128, 257}) {
        SCOPED_TRACE(size);
        constexpr size_t na = 3, nb = 73, k = 7;
        std::vector<uint8_t> a(na * size + 1), b(nb * size + 1);
        for (auto& v : a)
            v = rng();
        for (auto& v : b)
            v = rng();
        auto x = a.data() + 1;
        auto y = b.data() + 1;
        std::vector<hamdis_t> expected(na * nb), actual(na * nb);
        for (size_t i = 0; i < na; ++i)
            for (size_t j = 0; j < nb; ++j) {
                int count = 0;
                for (size_t t = 0; t < size; ++t)
                    count += __builtin_popcount(
                            unsigned(x[i * size + t] ^ y[j * size + t]));
                expected[i * nb + j] = count;
            }
        hammings(x, y, na, nb, size, actual.data());
        EXPECT_EQ(actual, expected);
        std::vector<hamdis_t> distances(na * k), reference(na * k);
        std::vector<idx_t> ids(na * k), refids(na * k);
        IndexBinaryFlat flat(size * 8);
        flat.add(nb, y);
        SIMDConfig::set_level(SIMDLevel::NONE);
        flat.search(na, x, k, reference.data(), refids.data());
        SIMDConfig::set_level(level);
        flat.search(na, x, k, distances.data(), ids.data());
        EXPECT_EQ(distances, reference);
        EXPECT_EQ(ids, refids);
        for (auto mode :
             {ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D3,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B8_D2,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B16_D2,
              ApproxTopK_mode_t::APPROX_TOPK_BUCKETS_B32_D2}) {
            int_maxheap_array_t heap{na, k, ids.data(), distances.data()},
                    refheap{na, k, refids.data(), reference.data()};
            SIMDConfig::set_level(SIMDLevel::NONE);
            hammings_knn_hc(&refheap, x, y, nb, size, 1, mode);
            SIMDConfig::set_level(level);
            hammings_knn_hc(&heap, x, y, nb, size, 1, mode);
            EXPECT_EQ(distances, reference);
            EXPECT_EQ(ids, refids);
        }
        if (size % 8 == 0) {
            int_maxheap_array_t heap{na, k, ids.data(), distances.data()};
            int_maxheap_array_t refheap{na, k, refids.data(), reference.data()};
            SIMDConfig::set_level(SIMDLevel::NONE);
            generalized_hammings_knn_hc(&refheap, x, y, nb, size, 1);
            SIMDConfig::set_level(level);
            generalized_hammings_knn_hc(&heap, x, y, nb, size, 1);
            EXPECT_EQ(distances, reference);
            EXPECT_EQ(ids, refids);
        }
        if (size == 8 || size == 16 || size == 32 || size == 64) {
            size_t count = 0, refcount = 0;
            int threshold = size * 4;
            for (auto v : expected)
                refcount += v <= threshold;
            hamming_count_thres(x, y, na, nb, threshold, size, &count);
            EXPECT_EQ(count, refcount);
            std::vector<idx_t> matches(na * nb * 2);
            count = match_hamming_thres(
                    x,
                    y,
                    na,
                    nb,
                    threshold,
                    size,
                    matches.data(),
                    actual.data());
            EXPECT_EQ(count, refcount);
            for (size_t i = 0; i < count; ++i)
                EXPECT_EQ(
                        actual[i],
                        expected[matches[2 * i] * nb + matches[2 * i + 1]]);
            SIMDConfig::set_level(SIMDLevel::NONE);
            crosshamming_count_thres(y, nb, threshold, size, &refcount);
            SIMDConfig::set_level(level);
            crosshamming_count_thres(y, nb, threshold, size, &count);
            EXPECT_EQ(count, refcount);
            IndexBinaryFlat quantizer(size * 8);
            IndexBinaryIVF ivf(&quantizer, size * 8, 1);
            ivf.train(nb, y);
            ivf.add(nb, y);
            flat.search(na, x, k, reference.data(), refids.data());
            ivf.search(na, x, k, distances.data(), ids.data());
            EXPECT_EQ(distances, reference);
            EXPECT_EQ(ids, refids);
        }
    }
}
} // namespace
TEST(TestLoongArchHamming, LSX) {
    check_hamming(faiss::SIMDLevel::LOONGARCH_LSX);
}
TEST(TestLoongArchHamming, LASX) {
    check_hamming(faiss::SIMDLevel::LOONGARCH_LASX);
}
