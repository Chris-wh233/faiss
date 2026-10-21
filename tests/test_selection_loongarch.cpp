/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/partitioning.h>
#include <faiss/utils/distances_fused/distances_fused.h>
#include <gtest/gtest.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>
#include <algorithm>
#include <numeric>
#include <random>
namespace {
using namespace faiss;
void check_selection(SIMDLevel level) {
    if (!SIMDConfig::is_simd_level_available(level))
        GTEST_SKIP();
    struct Guard {
        SIMDLevel saved = SIMDConfig::get_level();
        ~Guard() {
            SIMDConfig::set_level(saved);
        }
    } guard;
    SIMDConfig::set_level(level);
    std::mt19937 rng(411);
    for (size_t n : {17, 31, 32, 33, 65, 255, 1025}) {
        AlignedTable<uint16_t> data(n);
        std::vector<uint16_t> original(n);
        std::vector<int64_t> ids(n);
        for (auto& v : original)
            v = rng();
        for (bool reverse : {false, true}) {
            std::copy(original.begin(), original.end(), data.get());
            std::iota(ids.begin(), ids.end(), 0);
            size_t q = n / 3;
            if (reverse)
                partition<CMin<uint16_t, int64_t>>(
                        data.get(), ids.data(), n, q);
            else
                partition<CMax<uint16_t, int64_t>>(
                        data.get(), ids.data(), n, q);
            auto sorted = original;
            if (reverse)
                std::sort(
                        sorted.begin(), sorted.end(), std::greater<uint16_t>());
            else
                std::sort(sorted.begin(), sorted.end());
            std::vector<uint16_t> kept(data.get(), data.get() + q);
            if (reverse)
                std::sort(kept.begin(), kept.end(), std::greater<uint16_t>());
            else
                std::sort(kept.begin(), kept.end());
            EXPECT_TRUE(std::equal(kept.begin(), kept.end(), sorted.begin()));
            for (size_t i = 0; i < q; ++i) {
                ASSERT_GE(ids[i], 0);
                ASSERT_LT(size_t(ids[i]), n);
                EXPECT_EQ(data[i], original[ids[i]]);
            }
        }
        for (int shift : {0, 3, 12}) {
            std::copy(original.begin(), original.end(), data.get());
            int ref[16] = {}, actual[16] = {};
            for (auto v : original) {
                int bin = (int(v) - 123) >> shift;
                if (bin >= 0 && bin < 16)
                    ++ref[bin];
            }
            simd_histogram_16(data.get(), n, 123, shift, actual);
            for (int i = 0; i < 16; ++i)
                EXPECT_EQ(actual[i], ref[i]);
        }
    }
    for (int d = 1; d <= 8; ++d)
        for (size_t ny : {1, 7, 8, 9, 17, 33}) {
            constexpr size_t nx = 5;
            std::vector<float> x(nx * d), y(ny * d), dist(nx);
            std::vector<int64_t> ids(nx);
            for (auto& v : x)
                v = float(int(rng() % 1000) - 500) / 500;
            for (auto& v : y)
                v = float(int(rng() % 1000) - 500) / 500;
            Top1BlockResultHandler<CMax<float, int64_t>> handler(
                    nx, dist.data(), ids.data());
            EXPECT_TRUE(exhaustive_L2sqr_fused_cmax(
                    x.data(), y.data(), d, nx, ny, handler, nullptr));
            for (size_t i = 0; i < nx; ++i) {
                float best = INFINITY;
                int64_t id = -1;
                for (size_t j = 0; j < ny; ++j) {
                    float v = 0;
                    for (int t = 0; t < d; ++t) {
                        float delta = x[i * d + t] - y[j * d + t];
                        v += delta * delta;
                    }
                    if (v < best) {
                        best = v;
                        id = j;
                    }
                }
                EXPECT_NEAR(dist[i], best, 3e-6f);
                EXPECT_EQ(ids[i], id);
            }
        }
    for (int mode = 1; mode <= 4; ++mode)
        for (uint32_t n : {31, 32, 33, 129}) {
            std::vector<float> values(n * 3);
            for (auto& v : values)
                v = float(rng() % 1000000) / 1000000;
            float val[1];
            int32_t id[1];
            heap_heapify<CMax<float, int32_t>>(1, val, id);
            approx_topk_by_mode(
                    ApproxTopK_mode_t(mode), 3, n, values.data(), 1, val, id);
            EXPECT_EQ(val[0], *std::min_element(values.begin(), values.end()));
            ASSERT_GE(id[0], 0);
            ASSERT_LT(size_t(id[0]), values.size());
            EXPECT_EQ(val[0], values[id[0]]);
        }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchSelection, LSX) {
    check_selection(SIMDLevel::LOONGARCH_LSX);
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchSelection, LASX) {
    check_selection(SIMDLevel::LOONGARCH_LASX);
}
#endif

namespace {
template <SIMDLevel SL, class HC>
void check_graph_heap() {
    if (!SIMDConfig::is_simd_level_available(SL))
        return;
    for (int n : {1, 3, 4, 7, 8, 9, 15, 16, 17, 65, 128})
        for (int trial = 0; trial < 10; ++trial) {
            faiss::MinimaxHeapT<HC> a(n);
            for (int j = 0; j < n; ++j)
                a.push(j,
                       trial % 3 ? float((j * 17) % 13)
                                 : std::numeric_limits<float>::infinity());
            auto b = a;
            for (int j = 0; j < n; ++j) {
                float x, y;
                int expected = faiss::pop_min_tpl<HC, SIMDLevel::NONE>(&a, &x);
                int actual = faiss::pop_min_tpl<HC, SL>(&b, &y);
                EXPECT_EQ(actual, expected);
                EXPECT_EQ(x, y);
                EXPECT_EQ(a.nvalid, b.nvalid);
            }
            EXPECT_EQ((faiss::pop_min_tpl<HC, SL>(&b, nullptr)), -1);
        }
}
template <SIMDLevel SL>
void check_graph() {
    check_graph_heap<SL, faiss::CMax<float, int32_t>>();
    check_graph_heap<SL, faiss::CMin<float, int32_t>>();
    if (!SIMDConfig::is_simd_level_available(SL))
        return;
    std::vector<float> x(257), y(257);
    for (int j = 0; j < 257; ++j) {
        x[j] = float(j % 11) * 0.3f;
        y[j] = float(j % 7) * 0.7f;
    }
    for (int n = 1; n <= 256; ++n) {
        float expected = faiss::detail::block_l2<SIMDLevel::NONE>(
                x.data() + 1, y.data() + 1, n);
        EXPECT_NEAR(
                expected,
                faiss::detail::block_l2<SL>(x.data() + 1, y.data() + 1, n),
                expected * 1e-5f);
    }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchSelection, GraphLSX) {
    check_graph<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchSelection, GraphLASX) {
    check_graph<SIMDLevel::LOONGARCH_LASX>();
}
#endif
