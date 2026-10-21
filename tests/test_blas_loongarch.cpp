/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/simd_impl/L2sqr_blas.h>

namespace {
using namespace faiss;
struct BlasGuard {
    SIMDLevel level = SIMDConfig::get_level();
    int threshold = distance_compute_blas_threshold;
    int query_bs = distance_compute_blas_query_bs;
    int database_bs = distance_compute_blas_database_bs;
    ~BlasGuard() {
        SIMDConfig::set_level(level);
        distance_compute_blas_threshold = threshold;
        distance_compute_blas_query_bs = query_bs;
        distance_compute_blas_database_bs = database_bs;
    }
};

void expect_distance(float actual, float expected) {
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else {
        EXPECT_EQ(actual, expected);
        if (expected == 0) {
            EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        }
    }
}

template <SIMDLevel SL>
void check_correction() {
    if (!SIMDConfig::is_simd_level_available(SL)) {
        GTEST_SKIP();
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float max = std::numeric_limits<float>::max();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float values[] = {0, -0.0f, 1, -1, 2, 3, inf, -inf, nan, max, tiny};
    for (size_t n : {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 65}) {
        for (float x : values) {
            // Offset inputs and sentinels detect alignment and tail mistakes.
            std::vector<float> ip(n + 2, -777), norms(n + 2), ref(n);
            for (size_t j = 0; j < n; ++j) {
                ip[j + 1] = values[j % 11];
                norms[j + 1] = values[(j * 3 + 1) % 11];
                float dis = std::fma(-2.0f, ip[j + 1], x + norms[j + 1]);
                ref[j] = dis < 0 ? 0 : dis;
            }
            fvec_L2sqr_blas<SL>(n, ip.data() + 1, norms.data() + 1, x);
            for (size_t j = 0; j < n; ++j) {
                SCOPED_TRACE(j);
                expect_distance(ip[j + 1], ref[j]);
            }
            EXPECT_EQ(ip.front(), -777);
            EXPECT_EQ(ip.back(), -777);
        }
    }
}

template <SIMDLevel SL>
void check_search() {
    if (!SIMDConfig::is_simd_level_available(SL)) {
        GTEST_SKIP();
    }
    BlasGuard guard;
    distance_compute_blas_threshold = 0;
    distance_compute_blas_query_bs = 3;
    constexpr size_t nx = 7, ny = 137;
    for (size_t d : {9, 16, 33}) {
        // Binary fractions keep norm reductions exact, isolating BLAS
        // postprocessing from ISA-dependent norm summation order.
        std::vector<float> x(nx * d), y(ny * d), norms(ny);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = float(int(i * 17 % 41) - 20) / 8;
        for (size_t i = 0; i < y.size(); ++i)
            y[i] = float(int(i * 13 % 97) - 48) / 8;
        SIMDConfig::set_level(SIMDLevel::NONE);
        fvec_norms_L2sqr(norms.data(), y.data(), d, ny);
        for (int bs : {7, 8, 17, 64}) {
            distance_compute_blas_database_bs = bs;
            for (size_t k : {1, 3, 101, 140}) {
                for (bool precomputed : {false, true}) {
                    const float* yn = precomputed ? norms.data() : nullptr;
                    std::vector<float> ref(nx * k), actual(nx * k);
                    std::vector<idx_t> ri(nx * k), ai(nx * k);
                    SIMDConfig::set_level(SIMDLevel::NONE);
                    knn_L2sqr(
                            x.data(),
                            y.data(),
                            d,
                            nx,
                            ny,
                            k,
                            ref.data(),
                            ri.data(),
                            yn);
                    SIMDConfig::set_level(SL);
                    knn_L2sqr(
                            x.data(),
                            y.data(),
                            d,
                            nx,
                            ny,
                            k,
                            actual.data(),
                            ai.data(),
                            yn);
                    EXPECT_EQ(actual, ref);
                    EXPECT_EQ(ai, ri);
                }
            }
            // Range search shares the same BLAS correction path.
            RangeSearchResult ref(nx), actual(nx);
            SIMDConfig::set_level(SIMDLevel::NONE);
            range_search_L2sqr(x.data(), y.data(), d, nx, ny, 150, &ref);
            SIMDConfig::set_level(SL);
            range_search_L2sqr(x.data(), y.data(), d, nx, ny, 150, &actual);
            ASSERT_TRUE(std::equal(ref.lims, ref.lims + nx + 1, actual.lims));
            EXPECT_TRUE(
                    std::equal(
                            ref.labels,
                            ref.labels + ref.lims[nx],
                            actual.labels));
            EXPECT_TRUE(
                    std::equal(
                            ref.distances,
                            ref.distances + ref.lims[nx],
                            actual.distances));
        }
        // A contiguous selector restricts the database and restores global ids.
        IDSelectorRange sel(7, 29);
        std::vector<float> ref(nx * 3), actual(nx * 3);
        std::vector<idx_t> ri(nx * 3), ai(nx * 3);
        SIMDConfig::set_level(SIMDLevel::NONE);
        knn_L2sqr(
                x.data(),
                y.data(),
                d,
                nx,
                ny,
                3,
                ref.data(),
                ri.data(),
                nullptr,
                &sel);
        SIMDConfig::set_level(SL);
        knn_L2sqr(
                x.data(),
                y.data(),
                d,
                nx,
                ny,
                3,
                actual.data(),
                ai.data(),
                nullptr,
                &sel);
        EXPECT_EQ(actual, ref);
        EXPECT_EQ(ai, ri);
    }
}

template <SIMDLevel SL>
void check_top1_kernel() {
    if (!SIMDConfig::is_simd_level_available(SL)) {
        GTEST_SKIP();
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float max = std::numeric_limits<float>::max();
    for (size_t n : {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 65}) {
        for (float fill : {nan, inf, max, 5.0f}) {
            for (float limit : {max, 5.0f, 0.0f}) {
                for (size_t pos = 0; pos <= n; ++pos) {
                    std::vector<float> ip(n + 2, 0), norms(n + 2, fill), ref(n);
                    if (pos < n) {
                        norms[pos + 1] = -1;
                        norms[n] = -2; // after clamping, first zero must win
                    }
                    float best = limit;
                    size_t expected = n;
                    for (size_t j = 0; j < n; ++j) {
                        const float dis =
                                std::fma(-2.0f, ip[j + 1], norms[j + 1]);
                        ref[j] = dis < 0 ? 0 : dis;
                        if (ref[j] < best) {
                            best = ref[j];
                            expected = j;
                        }
                    }
                    EXPECT_EQ(
                            fvec_L2sqr_blas_cmax<SL>(
                                    n,
                                    ip.data() + 1,
                                    norms.data() + 1,
                                    0,
                                    limit),
                            expected);
                    for (size_t j = 0; j < n; ++j)
                        expect_distance(ip[j + 1], ref[j]);
                    EXPECT_EQ(ip.front(), 0);
                    EXPECT_EQ(ip.back(), 0);
                }
            }
        }
    }
}

template <SIMDLevel SL>
void check_top1_edges() {
    if (!SIMDConfig::is_simd_level_available(SL)) {
        GTEST_SKIP();
    }
    BlasGuard guard;
    distance_compute_blas_threshold = 0;
    distance_compute_blas_query_bs = 3;
    distance_compute_blas_database_bs = 7;
    constexpr size_t d = 9, nx = 7, ny = 33;
    const float max = std::numeric_limits<float>::max();
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x(nx * d, 0), y(ny * d, 0), norms(ny);
    std::vector<float> actual(nx), ref(nx);
    std::vector<idx_t> ai(nx), ri(nx);
    // All invalid, ordinary ties, and clipped ties spanning database blocks.
    for (float fill : {nan, inf, max, 0.0f, 5.0f}) {
        for (bool clipped : {false, true}) {
            std::fill(norms.begin(), norms.end(), fill);
            if (clipped) {
                norms[5] = -1;
                norms[8] = -2;
                norms[31] = -3;
            }
            SIMDConfig::set_level(SIMDLevel::NONE);
            knn_L2sqr(
                    x.data(),
                    y.data(),
                    d,
                    nx,
                    ny,
                    1,
                    ref.data(),
                    ri.data(),
                    norms.data());
            SIMDConfig::set_level(SL);
            knn_L2sqr(
                    x.data(),
                    y.data(),
                    d,
                    nx,
                    ny,
                    1,
                    actual.data(),
                    ai.data(),
                    norms.data());
            EXPECT_EQ(actual, ref);
            EXPECT_EQ(ai, ri);
            if (clipped)
                EXPECT_EQ(ai[0], fill == 0 ? 0 : 5);
        }
    }
    // Nonfinite queries cannot replace the handler's finite neutral value.
    for (float fill : {nan, inf}) {
        std::fill(x.begin(), x.end(), fill);
        knn_L2sqr(x.data(), y.data(), d, nx, ny, 1, actual.data(), ai.data());
        for (size_t i = 0; i < nx; ++i) {
            EXPECT_EQ(actual[i], max);
            EXPECT_EQ(ai[i], -1);
        }
    }
    knn_L2sqr(x.data(), y.data(), d, nx, 0, 1, actual.data(), ai.data());
    for (size_t i = 0; i < nx; ++i) {
        EXPECT_EQ(actual[i], max);
        EXPECT_EQ(ai[i], -1);
    }
    actual[0] = -777;
    ai[0] = -777;
    knn_L2sqr(x.data(), y.data(), d, 0, ny, 1, actual.data(), ai.data());
    EXPECT_EQ(actual[0], -777);
    EXPECT_EQ(ai[0], -777);
}
} // namespace

#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(LoongArchBlas, LSXCorrection) {
    check_correction<SIMDLevel::LOONGARCH_LSX>();
}
TEST(LoongArchBlas, LSXSearch) {
    check_search<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(LoongArchBlas, LASXCorrection) {
    check_correction<SIMDLevel::LOONGARCH_LASX>();
}
TEST(LoongArchBlas, LASXSearch) {
    check_search<SIMDLevel::LOONGARCH_LASX>();
}
#endif

#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(LoongArchBlas, LSXTop1Kernel) {
    check_top1_kernel<SIMDLevel::LOONGARCH_LSX>();
}
TEST(LoongArchBlas, LSXTop1Edges) {
    check_top1_edges<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(LoongArchBlas, LASXTop1Kernel) {
    check_top1_kernel<SIMDLevel::LOONGARCH_LASX>();
}
TEST(LoongArchBlas, LASXTop1Edges) {
    check_top1_edges<SIMDLevel::LOONGARCH_LASX>();
}
#endif
