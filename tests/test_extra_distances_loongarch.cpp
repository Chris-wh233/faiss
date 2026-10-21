/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <faiss/utils/extra_distances.h>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {
using namespace faiss;
void expect_distance(float expected, float actual) {
    if (std::isnan(expected))
        EXPECT_TRUE(std::isnan(actual));
    else if (std::isinf(expected))
        EXPECT_EQ(expected, actual);
    else
        EXPECT_NEAR(expected, actual, 2e-5f * (1 + std::abs(expected)));
}
template <SIMDLevel SL, MetricType metric>
void check_extra_metric() {
    std::mt19937 rng(619);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (size_t d :
         {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 513}) {
        SCOPED_TRACE(d);
        VectorDistance<metric, SIMDLevel::NONE> ref(d, 1.5f);
        VectorDistance<metric, SL> simd(d, 1.5f);
        for (int scenario = 0; scenario < 10; ++scenario) {
            SCOPED_TRACE(scenario);
            std::vector<float> a(d + 1), b(d + 1);
            float* x = a.data() + 1;
            float* y = b.data() + 1;
            for (size_t j = 0; j < d; ++j) {
                x[j] = 0.01f + float(rng() % 1000) / 1000;
                y[j] = 0.01f + float(rng() % 1000) / 1000;
                if (scenario == 1) {
                    x[j] = -x[j];
                    y[j] = -y[j];
                }
                if (scenario == 2) {
                    x[j] = 0;
                    y[j] = -0.0f;
                }
                if (scenario == 3) {
                    x[j] = nan;
                }
                if (scenario == 4) {
                    y[j] = nan;
                }
                if (scenario == 5) {
                    x[j] = nan;
                    y[j] = nan;
                }
                if (scenario == 6) {
                    x[j] = inf;
                }
                if (scenario == 7) {
                    x[j] = 1;
                    y[j] = -1;
                }
                if (scenario == 8) {
                    x[j] = std::numeric_limits<float>::max();
                    y[j] = x[j] / 2;
                }
                if (scenario == 9) {
                    x[j] = std::numeric_limits<float>::denorm_min() * 4;
                    y[j] = std::numeric_limits<float>::denorm_min();
                }
            }
            expect_distance(ref(x, y), simd(x, y));
        }
        // Missing/non-finite values in every lane, including the scalar tail.
        std::vector<float> x(d + 1, 0.25f), y(d + 1, 0.75f);
        for (size_t j = 1; j <= d; ++j) {
            for (float value : {nan, inf, -inf}) {
                x[j] = value;
                expect_distance(
                        ref(x.data() + 1, y.data() + 1),
                        simd(x.data() + 1, y.data() + 1));
            }
            x[j] = 0.25f;
        }
    }
}
template <SIMDLevel SL>
void check_arithmetic_metrics() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    check_extra_metric<SL, METRIC_Canberra>();
    check_extra_metric<SL, METRIC_BrayCurtis>();
    check_extra_metric<SL, METRIC_Jaccard>();
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchExtraDistances, ArithmeticLSX) {
    check_arithmetic_metrics<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchExtraDistances, ArithmeticLASX) {
    check_arithmetic_metrics<SIMDLevel::LOONGARCH_LASX>();
}
#endif

namespace {
template <SIMDLevel SL>
void check_masked_and_transcendental_metrics() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    check_extra_metric<SL, METRIC_NaNEuclidean>();
    check_extra_metric<SL, METRIC_GOWER>();
    check_extra_metric<SL, METRIC_Lp>();
    check_extra_metric<SL, METRIC_JensenShannon>();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (size_t d : {1, 4, 7, 8, 9, 17, 65, 65539}) {
        std::vector<float> x(d), y(d);
        for (size_t j = 0; j < d; ++j) {
            x[j] = j % 3 ? 0.25f : -2.0f;
            y[j] = j % 3 ? 0.75f : -3.0f;
            if (j % 7 == 0) {
                x[j] = nan;
                y[j] = 2.0f;
            }
        }
        VectorDistance<METRIC_GOWER, SIMDLevel::NONE> gr(d, 0);
        VectorDistance<METRIC_GOWER, SL> gv(d, 0);
        VectorDistance<METRIC_NaNEuclidean, SIMDLevel::NONE> nr(d, 0);
        VectorDistance<METRIC_NaNEuclidean, SL> nv(d, 0);
        expect_distance(gr(x.data(), y.data()), gv(x.data(), y.data()));
        expect_distance(nr(x.data(), y.data()), nv(x.data(), y.data()));
        // A type mismatch or out-of-range numeric value must be rejected.
        // Missing dimensions, even with invalid partners, must be skipped.
        for (size_t j : {size_t(0), d / 2, d - 1}) {
            x[j] = -1;
            y[j] = 0.5f;
            EXPECT_TRUE(std::isnan(gv(x.data(), y.data())));
            x[j] = 2;
            y[j] = 0.5f;
            EXPECT_TRUE(std::isnan(gv(x.data(), y.data())));
            x[j] = nan;
            expect_distance(gr(x.data(), y.data()), gv(x.data(), y.data()));
        }
    }
    for (float power : {-2.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, nan, inf}) {
        for (size_t d : {0, 1, 4, 7, 8, 9, 17, 65}) {
            VectorDistance<METRIC_Lp, SIMDLevel::NONE> ref(d, power);
            VectorDistance<METRIC_Lp, SL> simd(d, power);
            std::vector<float> x(d + 1), y(d + 1, 0);
            for (size_t j = 1; j <= d; ++j)
                x[j] = float(j % 7) * 0.5f;
            expect_distance(
                    ref(x.data() + 1, y.data() + 1),
                    simd(x.data() + 1, y.data() + 1));
            if (d) {
                x[d] = nan;
                expect_distance(
                        ref(x.data() + 1, y.data() + 1),
                        simd(x.data() + 1, y.data() + 1));
            }
        }
    }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchExtraDistances, MaskedAndTranscendentalLSX) {
    check_masked_and_transcendental_metrics<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchExtraDistances, MaskedAndTranscendentalLASX) {
    check_masked_and_transcendental_metrics<SIMDLevel::LOONGARCH_LASX>();
}
#endif
