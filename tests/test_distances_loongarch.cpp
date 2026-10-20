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
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/distances_dispatch.h>

namespace {
using namespace faiss;
struct LevelGuard {
    SIMDLevel saved = SIMDConfig::get_level();
    ~LevelGuard() {
        SIMDConfig::set_level(saved);
    }
};

template <SIMDLevel SL>
void check_distances() {
    if (!SIMDConfig::is_simd_level_available(SL)) {
        GTEST_SKIP() << "SIMD level is unavailable";
    }
    LevelGuard guard;
    SIMDConfig::set_level(SL);
    EXPECT_EQ(SIMDConfig::get_dispatched_level(), SL);
    EXPECT_EQ(with_simd_level_distances([]<SIMDLevel L>() { return L; }), SL);
    // Families not ported yet must retain a valid fallback.
    EXPECT_EQ(
            with_simd_level([]<SIMDLevel L>() { return L; }), SIMDLevel::NONE);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    for (size_t d : {0,  1,  2,  3,  4,  5,  7,   8,   9,   15,
                     16, 17, 31, 32, 33, 64, 127, 128, 129, 513}) {
        SCOPED_TRACE(d);
        // Exact-size vectors, offset by one float, also exercise unaligned
        // loads.
        std::vector<float> a(d + 1), b(d + 1), out(d + 2, -777),
                ref(d + 2, -777);
        for (auto& v : a)
            v = random(rng);
        for (auto& v : b)
            v = random(rng);
        const float* x = a.data() + 1;
        const float* y = b.data() + 1;
        float tol = std::max(1e-6f, float(d) * 1e-6f);
        EXPECT_NEAR(
                fvec_L2sqr<SL>(x, y, d),
                fvec_L2sqr<SIMDLevel::NONE>(x, y, d),
                tol);
        EXPECT_NEAR(
                fvec_inner_product<SL>(x, y, d),
                fvec_inner_product<SIMDLevel::NONE>(x, y, d),
                tol);
        EXPECT_NEAR(
                fvec_norm_L2sqr<SL>(x, d),
                fvec_norm_L2sqr<SIMDLevel::NONE>(x, d),
                tol);
        EXPECT_NEAR(
                fvec_L1<SL>(x, y, d), fvec_L1<SIMDLevel::NONE>(x, y, d), tol);
        EXPECT_FLOAT_EQ(
                fvec_Linf<SL>(x, y, d), fvec_Linf<SIMDLevel::NONE>(x, y, d));
        EXPECT_EQ(fvec_L2sqr(x, y, d), fvec_L2sqr<SL>(x, y, d));
        EXPECT_EQ(fvec_inner_product(x, y, d), fvec_inner_product<SL>(x, y, d));
        fvec_add(d, x, y, out.data() + 1);
        for (size_t i = 0; i < d; ++i)
            EXPECT_FLOAT_EQ(out[i + 1], x[i] + y[i]);
        fvec_sub(d, x, y, out.data() + 1);
        for (size_t i = 0; i < d; ++i)
            EXPECT_FLOAT_EQ(out[i + 1], x[i] - y[i]);
        fvec_add(d, x, 2.0f, out.data() + 1);
        for (size_t i = 0; i < d; ++i)
            EXPECT_FLOAT_EQ(out[i + 1], x[i] + 2.0f);
        fvec_madd(d, x, 0.25f, y, out.data() + 1);
        for (size_t i = 0; i < d; ++i)
            EXPECT_NEAR(out[i + 1], x[i] + 0.25f * y[i], 1e-6f);
        int actual = fvec_madd_and_argmin(d, x, 0.25f, y, out.data() + 1);
        int expected = fvec_madd_and_argmin<SIMDLevel::NONE>(
                d, x, 0.25f, y, ref.data() + 1);
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(out.front(), -777);
        EXPECT_EQ(out.back(), -777);
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t n : {0, 1, 3, 4, 7, 8, 9, 16, 17, 33}) {
        std::vector<float> a(n, 1e20f), b(n, 0), out(n);
        EXPECT_EQ(
                fvec_madd_and_argmin(n, a.data(), 1, b.data(), out.data()), -1);
        std::fill(a.begin(), a.end(), nan);
        EXPECT_EQ(
                fvec_madd_and_argmin(n, a.data(), 1, b.data(), out.data()), -1);
        std::fill(a.begin(), a.end(), inf);
        EXPECT_EQ(
                fvec_madd_and_argmin(n, a.data(), 1, b.data(), out.data()), -1);
        if (n > 2) {
            a[1] = a[n - 1] = -10;
            EXPECT_EQ(
                    fvec_madd_and_argmin(n, a.data(), 1, b.data(), out.data()),
                    1);
            a[1] = a[n - 1] = -inf;
            EXPECT_EQ(
                    fvec_madd_and_argmin(n, a.data(), 1, b.data(), out.data()),
                    1);
        }
    }
    // Nearest ties and empty sets; both row-major and padded transposed input.
    for (size_t ny : {0, 1, 3, 4, 7, 8, 9, 16, 17}) {
        constexpr size_t d = 5;
        const size_t stride = ny + 3;
        std::vector<float> x(d, 1), y(d * ny, 2), yt(d * stride, 2),
                norms(ny, 20), out(ny);
        EXPECT_EQ(
                fvec_L2sqr_ny_nearest(out.data(), x.data(), y.data(), d, ny),
                0u);
        EXPECT_EQ(
                fvec_L2sqr_ny_nearest_y_transposed(
                        out.data(),
                        x.data(),
                        yt.data(),
                        norms.data(),
                        d,
                        stride,
                        ny),
                0u);
    }
    // Exercise VectorDistance and IVFFlatScanner through real indices.
    constexpr size_t d = 17, nb = 64, nq = 3, k = 5;
    std::vector<float> data(d * nb), query(d * nq), reference(nq * k),
            actual(nq * k);
    std::vector<idx_t> ref_ids(nq * k), ids(nq * k);
    for (auto& v : data)
        v = random(rng);
    for (auto& v : query)
        v = random(rng);
    for (auto metric :
         {METRIC_L2, METRIC_INNER_PRODUCT, METRIC_L1, METRIC_Linf}) {
        IndexFlat flat(d, metric);
        flat.add(nb, data.data());
        SIMDConfig::set_level(SIMDLevel::NONE);
        flat.search(nq, query.data(), k, reference.data(), ref_ids.data());
        SIMDConfig::set_level(SL);
        flat.search(nq, query.data(), k, actual.data(), ids.data());
        EXPECT_EQ(ids, ref_ids);
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], reference[i], 1e-4f);
        IndexFlat quantizer(d, metric);
        IndexIVFFlat ivf(&quantizer, d, 1, metric);
        ivf.train(nb, data.data());
        ivf.add(nb, data.data());
        ivf.search(nq, query.data(), k, actual.data(), ids.data());
        EXPECT_EQ(ids, ref_ids);
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], reference[i], 1e-4f);
    }
}
} // namespace

#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchDistances, LSX) {
    check_distances<faiss::SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchDistances, LASX) {
    check_distances<faiss::SIMDLevel::LOONGARCH_LASX>();
}
#endif
