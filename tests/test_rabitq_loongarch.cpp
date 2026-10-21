/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/utils/rabitq_simd.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
namespace {
template <faiss::SIMDLevel SL>
void check_bits() {
    using namespace faiss;
    using namespace faiss::rabitq;
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    std::mt19937 rng(12);
    for (size_t n : {0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 65, 1025, 131073}) {
        std::vector<float> x(n + 1);
        for (size_t i = 0; i < n; ++i)
            x[i + 1] =
                    n > 1025 ? 255.0f : float(int(rng() % 2049) - 1024) * 0.25f;
        float rmin = 123, rmax = 456, amin = rmin, amax = rmax;
        minmax_values<SIMDLevel::NONE>(x.data() + 1, n, rmin, rmax);
        minmax_values<SL>(x.data() + 1, n, amin, amax);
        EXPECT_EQ(rmin, amin);
        EXPECT_EQ(rmax, amax);
        for (size_t qb = 1; qb <= 8; ++qb)
            for (bool centered : {false, true}) {
                std::vector<uint8_t> ref(n + 2, 123), actual(n + 2, 123);
                size_t rs = 17, as = 17;
                int64_t r2 = 19, a2 = 19;
                quantize_query_values<SIMDLevel::NONE>(
                        x.data() + 1,
                        n,
                        0,
                        1,
                        (1u << qb) - 1,
                        centered,
                        ref.data() + 1,
                        rs,
                        r2);
                quantize_query_values<SL>(
                        x.data() + 1,
                        n,
                        0,
                        1,
                        (1u << qb) - 1,
                        centered,
                        actual.data() + 1,
                        as,
                        a2);
                EXPECT_EQ(ref, actual);
                EXPECT_EQ(rs, as);
                EXPECT_EQ(r2, a2);
            }
    }
    for (int trial = 0; trial < 300; ++trial) {
        float tab[16];
        for (int j = 0; j < 16; ++j) {
            float v = float(rng() % 257) - 0.5f;
            tab[j] = trial % 2 ? std::nextafter(v, 0.0f) : v;
        }
        uint8_t ref[16], actual[16];
        lut_quantize_16_to_uint8<SIMDLevel::NONE>(tab, 0, 1, ref);
        lut_quantize_16_to_uint8<SL>(tab, 0, 1, actual);
        for (int j = 0; j < 16; ++j)
            EXPECT_EQ(ref[j], actual[j]);
        float mn, mx, a, b;
        lut_minmax_16<SIMDLevel::NONE>(tab, mn, mx);
        lut_minmax_16<SL>(tab, a, b);
        EXPECT_EQ(a, mn);
        EXPECT_EQ(b, mx);
    }
    for (size_t d : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129, 1025}) {
        for (size_t bits = 1; bits <= 7; ++bits) {
            std::vector<uint8_t> signs((d + 7) / 8), extra((d * bits + 7) / 8);
            std::vector<float> q(d);
            for (auto& v : signs)
                v = rng();
            for (auto& v : extra)
                v = rng();
            const float cb = -(float(1u << bits) - 0.5f);
            double ref = 0;
            for (size_t i = 0; i < d; ++i) {
                q[i] = float(int(rng() % 2001) - 1000) / 1000;
                unsigned value = 0;
                for (size_t b = 0; b < bits; ++b) {
                    size_t bit = i * bits + b;
                    value |= ((extra[bit / 8] >> (bit % 8)) & 1u) << b;
                }
                value += ((signs[i / 8] >> (i % 8)) & 1u) << bits;
                ref += q[i] * (float(value) + cb);
            }
            EXPECT_NEAR(
                    multibit::compute_inner_product<SL>(
                            signs.data(), extra.data(), q.data(), d, bits, cb),
                    ref,
                    0.003 + std::abs(ref) * 2e-5);
        }
    }
    for (size_t n :
         {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 1025, 131073}) {
        SCOPED_TRACE(n);
        for (size_t qb = 1; qb <= 8; ++qb) {
            std::vector<uint8_t> data(n + 1), query(n * qb + 1);
            for (auto& v : data)
                v = n > 1025 ? 255 : rng();
            for (auto& v : query)
                v = n > 1025 ? 255 : rng();
            auto d = data.data() + 1;
            auto q = query.data() + 1;
            uint64_t pop = 0, dot = 0, xor_dot = 0;
            for (size_t i = 0; i < n; ++i) {
                pop += __builtin_popcount(unsigned(d[i]));
                for (size_t b = 0; b < qb; ++b) {
                    dot += uint64_t(__builtin_popcount(
                                   unsigned(d[i] & q[b * n + i])))
                            << b;
                    xor_dot += uint64_t(__builtin_popcount(
                                       unsigned(d[i] ^ q[b * n + i])))
                            << b;
                }
            }
            EXPECT_EQ(popcount<SL>(d, n), pop);
            EXPECT_EQ(bitwise_and_dot_product<SL>(q, d, n, qb), dot);
            EXPECT_EQ(bitwise_xor_dot_product<SL>(q, d, n, qb), xor_dot);
            auto both = bitwise_and_dot_product_with_popcount<SL>(q, d, n, qb);
            EXPECT_EQ(both.dot_product, dot);
            EXPECT_EQ(both.popcount, pop);
            size_t bytes = qb * ((n + 7) / 8);
            std::vector<uint8_t> expected(bytes + 2, 123),
                    actual(bytes + 2, 123);
            rearrange_bit_planes<SIMDLevel::NONE>(
                    d, n, qb, expected.data() + 1);
            rearrange_bit_planes<SL>(d, n, qb, actual.data() + 1);
            EXPECT_EQ(actual, expected);
        }
    }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchRaBitQ, LSX) {
    check_bits<faiss::SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchRaBitQ, LASX) {
    check_bits<faiss::SIMDLevel::LOONGARCH_LASX>();
}
#endif
