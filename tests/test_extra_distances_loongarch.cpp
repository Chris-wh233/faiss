/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <algorithm>
#include <memory>
#include <type_traits>
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

namespace {
struct ExtraLevelGuard {
    SIMDLevel saved = SIMDConfig::get_level();
    ~ExtraLevelGuard() {
        SIMDConfig::set_level(saved);
    }
};
template <SIMDLevel SL, MetricType metric>
void check_extra_dispatch() {
    constexpr int d = 17, nb = 43, nq = 11, k = 7;
    constexpr float arg = 1.5f;
    std::mt19937 rng(156);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (auto* values : {&xb, &xq})
        for (size_t j = 0; j < values->size(); ++j) {
            (*values)[j] = 0.01f + float(rng() % 9001) / 10000;
            if constexpr (metric == METRIC_GOWER) {
                if (j % d % 3 == 0)
                    (*values)[j] = -1.0f - float(rng() % 3);
            }
        }
    SIMDConfig::set_level(SL);
    with_VectorDistance(d, metric, arg, [&](auto vd) {
        EXPECT_TRUE((std::is_same_v<decltype(vd), VectorDistance<metric, SL>>));
        VectorDistance<metric, SIMDLevel::NONE> ref(d, arg);
        expect_distance(ref(xb.data(), xq.data()), vd(xb.data(), xq.data()));
    });
    // Strided pairwise input and output exercise the public wrapper, including
    // its OpenMP loop. Padding must remain untouched.
    std::vector<float> padded_x(nq * (d + 3), 123), padded_y(nb * (d + 2), 123);
    for (int i = 0; i < nq; ++i)
        std::copy_n(xq.data() + i * d, d, padded_x.data() + i * (d + 3));
    for (int i = 0; i < nb; ++i)
        std::copy_n(xb.data() + i * d, d, padded_y.data() + i * (d + 2));
    std::vector<float> pairwise(nq * (nb + 5), 123);
    pairwise_extra_distances(
            d,
            nq,
            padded_x.data(),
            nb,
            padded_y.data(),
            metric,
            arg,
            pairwise.data(),
            d + 3,
            d + 2,
            nb + 5);
    VectorDistance<metric, SIMDLevel::NONE> ref(d, arg);
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < nb; ++j)
            expect_distance(
                    ref(xq.data() + i * d, xb.data() + j * d),
                    pairwise[i * (nb + 5) + j]);
        for (int j = nb; j < nb + 5; ++j)
            EXPECT_EQ(pairwise[i * (nb + 5) + j], 123);
    }
    IndexFlat flat(d, metric);
    flat.metric_arg = arg;
    flat.add(nb, xb.data());
    // Pre-populate one coarse centroid to isolate scanner dispatch from
    // clustering and make every vector participate in the IVF comparison.
    IndexFlatL2 coarse(d);
    coarse.add(1, xb.data());
    IndexIVFFlat ivf(&coarse, d, 1, metric);
    ivf.metric_arg = arg;
    ASSERT_TRUE(ivf.is_trained);
    ivf.add(nb, xb.data());
    std::unique_ptr<FlatCodesDistanceComputer> dc(
            flat.get_FlatCodesDistanceComputer());
    dc->set_query(xq.data());
    for (int j = 0; j < nb; ++j) {
        expect_distance(ref(xq.data(), xb.data() + j * d), (*dc)(j));
        expect_distance(
                ref(xb.data(), xb.data() + j * d), dc->symmetric_dis(0, j));
    }
    IDSelectorRange selector(2, nb - 3);
    SearchParametersIVF params;
    params.nprobe = 1;
    for (Index* index :
         {static_cast<Index*>(&flat), static_cast<Index*>(&ivf)}) {
        for (bool filtered : {false, true}) {
            params.sel = filtered ? &selector : nullptr;
            std::vector<float> expected(nq * k), actual(nq * k);
            std::vector<idx_t> expected_ids(nq * k), actual_ids(nq * k);
            SIMDConfig::set_level(SIMDLevel::NONE);
            index->search(
                    nq,
                    xq.data(),
                    k,
                    expected.data(),
                    expected_ids.data(),
                    &params);
            SIMDConfig::set_level(SL);
            index->search(
                    nq,
                    xq.data(),
                    k,
                    actual.data(),
                    actual_ids.data(),
                    &params);
            EXPECT_EQ(expected_ids, actual_ids);
            for (size_t i = 0; i < actual.size(); ++i)
                expect_distance(expected[i], actual[i]);
            // A threshold inside the distance range exercises both accepted
            // and rejected candidates, for distances and Jaccard similarity.
            const float radius = (pairwise[0] + pairwise[nb - 1]) * 0.5f;
            RangeSearchResult r(nq), a(nq);
            SIMDConfig::set_level(SIMDLevel::NONE);
            index->range_search(nq, xq.data(), radius, &r, &params);
            SIMDConfig::set_level(SL);
            index->range_search(nq, xq.data(), radius, &a, &params);
            for (int i = 0; i < nq; ++i) {
                ASSERT_EQ(r.lims[i + 1] - r.lims[i], a.lims[i + 1] - a.lims[i]);
                std::vector<std::pair<idx_t, float>> rr, aa;
                for (size_t j = r.lims[i]; j < r.lims[i + 1]; ++j)
                    rr.emplace_back(r.labels[j], r.distances[j]);
                for (size_t j = a.lims[i]; j < a.lims[i + 1]; ++j)
                    aa.emplace_back(a.labels[j], a.distances[j]);
                std::sort(rr.begin(), rr.end());
                std::sort(aa.begin(), aa.end());
                for (size_t j = 0; j < rr.size(); ++j) {
                    EXPECT_EQ(rr[j].first, aa[j].first);
                    expect_distance(rr[j].second, aa[j].second);
                }
            }
        }
    }
}
template <SIMDLevel SL>
void check_extra_integration() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    ExtraLevelGuard guard;
    check_extra_dispatch<SL, METRIC_Canberra>();
    check_extra_dispatch<SL, METRIC_BrayCurtis>();
    check_extra_dispatch<SL, METRIC_Jaccard>();
    check_extra_dispatch<SL, METRIC_NaNEuclidean>();
    check_extra_dispatch<SL, METRIC_GOWER>();
    check_extra_dispatch<SL, METRIC_Lp>();
    check_extra_dispatch<SL, METRIC_JensenShannon>();
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchExtraDistances, IntegrationLSX) {
    check_extra_integration<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchExtraDistances, IntegrationLASX) {
    check_extra_integration<SIMDLevel::LOONGARCH_LASX>();
}
#endif
