/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/utils/simd_impl/quantizer_preprocessing.h>
#include <faiss/impl/EDENQuantizer.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/fp16.h>
#include <gtest/gtest.h>
#include <bit>
#include <memory>
#include <random>
namespace {
using namespace faiss;
using QT = ScalarQuantizer::QuantizerType;
const QT types[] = {
        ScalarQuantizer::QT_8bit,
        ScalarQuantizer::QT_4bit,
        ScalarQuantizer::QT_6bit,
        ScalarQuantizer::QT_8bit_uniform,
        ScalarQuantizer::QT_4bit_uniform,
        ScalarQuantizer::QT_fp16,
        ScalarQuantizer::QT_bf16,
        ScalarQuantizer::QT_8bit_direct,
        ScalarQuantizer::QT_8bit_direct_signed};
template <SIMDLevel SL>
void check_codec() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    std::mt19937 rng(718);
    for (auto qt : types)
        for (size_t d :
             {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129}) {
            SCOPED_TRACE(int(qt));
            SCOPED_TRACE(d);
            ScalarQuantizer sq(d, qt);
            sq.trained.resize(2 * d);
            bool uniform = qt == ScalarQuantizer::QT_4bit_uniform ||
                    qt == ScalarQuantizer::QT_8bit_uniform;
            for (size_t i = 0; i < d; ++i) {
                sq.trained[i] = float(i % 3) * 0.1f;
                sq.trained[d + i] = i % 7 ? 1.3f : 0.0f;
            }
            if (uniform) {
                sq.trained[0] = 0;
                sq.trained[1] = 1;
            }
            std::unique_ptr<ScalarQuantizer::SQuantizer> ref(
                    scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                            qt, d, sq.trained));
            std::unique_ptr<ScalarQuantizer::SQuantizer> simd(
                    scalar_quantizer::sq_select_quantizer<SL>(
                            qt, d, sq.trained));
            ASSERT_NE(simd, nullptr);
            for (int trial = 0; trial < 20; ++trial) {
                std::vector<float> x(d + 1), expected(d + 2, 123),
                        actual(d + 2, 123);
                for (size_t i = 0; i < d; ++i) {
                    float v = float(int(rng() % 3001) - 1000) / 1000;
                    if (qt == ScalarQuantizer::QT_8bit_direct)
                        v = float(rng() % 256) + 0.25f;
                    if (qt == ScalarQuantizer::QT_8bit_direct_signed)
                        v = float(int(rng() % 256) - 128) + 0.25f;
                    if (qt == ScalarQuantizer::QT_fp16 ||
                        qt == ScalarQuantizer::QT_bf16)
                        v = std::bit_cast<float>(uint32_t(rng()));
                    if (uniform && trial < 6) {
                        int levels = qt == ScalarQuantizer::QT_4bit_uniform
                                ? 15
                                : 255;
                        v = float(i % levels) / levels;
                        if (trial % 2)
                            v = std::nextafter(v, 0.0f);
                    }
                    x[i + 1] = v;
                }
                std::vector<uint8_t> ec(sq.code_size + 2), ac(sq.code_size + 2);
                ec.front() = ac.front() = 123;
                ec.back() = ac.back() = 231;
                ref->encode_vector(x.data() + 1, ec.data() + 1);
                simd->encode_vector(x.data() + 1, ac.data() + 1);
                ASSERT_EQ(ec, ac);
                ref->decode_vector(ec.data() + 1, expected.data() + 1);
                simd->decode_vector(ec.data() + 1, actual.data() + 1);
                EXPECT_EQ(actual.front(), 123);
                EXPECT_EQ(actual.back(), 123);
                for (size_t i = 1; i <= d; ++i) {
                    if (std::isnan(expected[i]))
                        EXPECT_TRUE(std::isnan(actual[i]));
                    else if (std::isinf(expected[i]))
                        EXPECT_EQ(actual[i], expected[i]);
                    else
                        EXPECT_NEAR(
                                actual[i],
                                expected[i],
                                std::abs(expected[i]) * 2e-6f + 1e-7f);
                }
            }
        }
    // Exhaust every half-precision code, including subnormal and NaN payloads.
    for (auto qt : {ScalarQuantizer::QT_fp16, ScalarQuantizer::QT_bf16}) {
        std::vector<float> trained, decoded(65536);
        std::vector<uint16_t> codes(65536);
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = i;
        std::unique_ptr<ScalarQuantizer::SQuantizer> simd(
                scalar_quantizer::sq_select_quantizer<SL>(
                        qt, codes.size(), trained));
        simd->decode_vector(
                reinterpret_cast<const uint8_t*>(codes.data()), decoded.data());
        for (size_t i = 0; i < codes.size(); ++i) {
            float ref = qt == ScalarQuantizer::QT_fp16 ? decode_fp16(i)
                                                       : decode_bf16(i);
            if (std::isnan(ref))
                EXPECT_TRUE(std::isnan(decoded[i]));
            else
                EXPECT_EQ(
                        std::bit_cast<uint32_t>(ref),
                        std::bit_cast<uint32_t>(decoded[i]));
        }
    }
}
template <SIMDLevel SL>
void check_distances() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    struct Guard {
        SIMDLevel saved = SIMDConfig::get_level();
        ~Guard() {
            SIMDConfig::set_level(saved);
        }
    } guard;
    std::mt19937 rng(919);
    for (auto qt : types)
        for (size_t d : {1, 7, 8, 9, 16, 17, 32, 65}) {
            SCOPED_TRACE(int(qt));
            SCOPED_TRACE(d);
            ScalarQuantizer sq(d, qt);
            sq.trained.resize(2 * d, 1);
            for (size_t i = 0; i < d; ++i)
                sq.trained[i] = -1;
            if (qt == ScalarQuantizer::QT_4bit_uniform ||
                qt == ScalarQuantizer::QT_8bit_uniform) {
                sq.trained[0] = -1;
                sq.trained[1] = 2;
            }
            std::vector<float> x(2 * d), query(d);
            for (auto& v : x)
                v = float(rng() % 1000) / 1000;
            for (auto& v : query)
                v = float(rng() % 1000) / 1000;
            if (qt == ScalarQuantizer::QT_8bit_direct ||
                qt == ScalarQuantizer::QT_8bit_direct_signed)
                for (auto& v : x)
                    v = std::floor(v * 100);
            std::vector<uint8_t> codes(2 * sq.code_size);
            SIMDConfig::set_level(SL);
            sq.compute_codes(x.data(), codes.data(), 2);
            for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
                std::unique_ptr<ScalarQuantizer::SQDistanceComputer> ref(
                        scalar_quantizer::sq_select_distance_computer<
                                SIMDLevel::NONE>(metric, qt, d, sq.trained));
                std::unique_ptr<ScalarQuantizer::SQDistanceComputer> simd(
                        scalar_quantizer::sq_select_distance_computer<SL>(
                                metric, qt, d, sq.trained));
                ASSERT_NE(simd, nullptr);
                for (auto dc : {ref.get(), simd.get()}) {
                    dc->set_query(query.data());
                    dc->codes = codes.data();
                    dc->code_size = sq.code_size;
                }
                for (int j = 0; j < 2; ++j) {
                    float expected =
                            ref->query_to_code(codes.data() + j * sq.code_size);
                    EXPECT_NEAR(
                            simd->query_to_code(
                                    codes.data() + j * sq.code_size),
                            expected,
                            2e-5f * std::abs(expected) + 1e-6f);
                }
                float expected = ref->symmetric_dis(0, 1);
                EXPECT_NEAR(
                        simd->symmetric_dis(0, 1),
                        expected,
                        2e-5f * std::abs(expected) + 1e-6f);
            }
        }
    for (auto qt : types)
        for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT})
            for (bool residual : {false, true}) {
                constexpr int d = 17, nb = 83, nq = 5, k = 7;
                IndexFlat coarse(d, metric);
                std::vector<float> centroid(d, 0.125f);
                coarse.add(1, centroid.data());
                IndexIVFScalarQuantizer index(
                        &coarse, d, 1, qt, metric, residual);
                index.sq.trained.resize(2 * d, 2);
                for (int i = 0; i < d; ++i)
                    index.sq.trained[i] = -1;
                if (qt == ScalarQuantizer::QT_4bit_uniform ||
                    qt == ScalarQuantizer::QT_8bit_uniform) {
                    index.sq.trained[0] = -1;
                    index.sq.trained[1] = 2;
                }
                index.is_trained = true;
                std::vector<float> xb(nb * d), xq(nq * d);
                for (auto& v : xb)
                    v = float(rng() % 1000) / 1000;
                for (auto& v : xq)
                    v = float(rng() % 1000) / 1000;
                if (qt == ScalarQuantizer::QT_8bit_direct ||
                    qt == ScalarQuantizer::QT_8bit_direct_signed)
                    for (auto& v : xb)
                        v = v * 100 + 1;
                SIMDConfig::set_level(SL);
                index.add(nb, xb.data());
                IDSelectorRange sel(3, nb - 2);
                SearchParametersIVF params;
                params.sel = &sel;
                std::vector<float> expected(nq * k), actual(nq * k);
                std::vector<idx_t> ei(nq * k), ai(nq * k);
                SIMDConfig::set_level(SIMDLevel::NONE);
                index.search(
                        nq, xq.data(), k, expected.data(), ei.data(), &params);
                SIMDConfig::set_level(SL);
                index.search(
                        nq, xq.data(), k, actual.data(), ai.data(), &params);
                for (size_t i = 0; i < actual.size(); ++i) {
                    EXPECT_NEAR(
                            actual[i],
                            expected[i],
                            std::abs(expected[i]) * 3e-5f + 1e-5f);
                    EXPECT_EQ(ai[i], ei[i]);
                }
            }
}

} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchSQ, LSXDistances) {
    check_distances<faiss::SIMDLevel::LOONGARCH_LSX>();
}
TEST(TestLoongArchSQ, LSXCodec) {
    check_codec<faiss::SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchSQ, LASXDistances) {
    check_distances<faiss::SIMDLevel::LOONGARCH_LASX>();
}
TEST(TestLoongArchSQ, LASXCodec) {
    check_codec<faiss::SIMDLevel::LOONGARCH_LASX>();
}
#endif

namespace {
template <SIMDLevel SL>
void check_advanced_codecs() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    std::mt19937 rng(1234);
    auto original = SIMDConfig::get_level();
    struct Restore {
        SIMDLevel level;
        ~Restore() {
            SIMDConfig::set_level(level);
        }
    } restore{original};
    SIMDConfig::set_level(SL);
    for (int bits = 1; bits <= 8; ++bits)
        for (size_t d : {1, 3, 7, 8, 9, 17, 32, 65}) {
            auto qt = QT(int(ScalarQuantizer::QT_1bit_eden) + bits - 1);
            ScalarQuantizer sq(d, qt);
            int k = 1 << bits;
            sq.trained.resize(2 * k - 1);
            for (int j = 0; j < k; ++j)
                sq.trained[j] = float(j) - k / 2;
            for (int j = 0; j < k - 1; ++j)
                sq.trained[k + j] = sq.trained[j] + 0.5f;
            std::unique_ptr<ScalarQuantizer::SQuantizer> ref(
                    scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                            qt, d, sq.trained));
            std::unique_ptr<ScalarQuantizer::SQuantizer> simd(
                    scalar_quantizer::sq_select_quantizer<SL>(
                            qt, d, sq.trained));
            ASSERT_NE(simd, nullptr);
            for (int trial = 0; trial < 8; ++trial) {
                std::vector<float> x(d), a(d + 1, 77), b(d + 1, 77);
                for (size_t j = 0; j < d; ++j) {
                    x[j] = sq.trained[k + rng() % (k - 1)];
                    if (trial % 3 == 1)
                        x[j] = std::nextafter(x[j], -INFINITY);
                    if (trial % 3 == 2)
                        x[j] = std::nextafter(x[j], INFINITY);
                }
                std::vector<uint8_t> ca(sq.code_size + 1, 0), cb(ca);
                ca.back() = cb.back() = 0xa5;
                ref->encode_vector(x.data(), ca.data());
                simd->encode_vector(x.data(), cb.data());
                ASSERT_EQ(ca, cb) << bits << " " << d;
                ref->decode_vector(ca.data(), a.data());
                simd->decode_vector(ca.data(), b.data());
                ASSERT_EQ(a, b);
            }
        }
    for (int bits = 1; bits <= 8; ++bits)
        for (size_t d : {8, 17, 32}) {
            ScalarQuantizer sq(d, eden_utils::quantizer_type_for_bits(bits));
            int k = 1 << bits;
            sq.trained.resize(2 * k - 1);
            for (int j = 0; j < k; ++j)
                sq.trained[j] = (float(j) - float(k - 1) / 2) * 0.1f;
            for (int j = 0; j < k - 1; ++j)
                sq.trained[k + j] = (sq.trained[j] + sq.trained[j + 1]) * 0.5f;
            std::vector<float> x(16 * d), q(d);
            for (float& v : x)
                v = float(int(rng() % 2001) - 1000) / 100;
            for (float& v : q)
                v = float(int(rng() % 2001) - 1000) / 100;
            size_t cs = eden_utils::code_size(d, bits);
            std::vector<uint8_t> codes(16 * cs);
            for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT})
                for (auto scale :
                     {EDENScaleType_UNBIASED, EDENScaleType_BIASED}) {
                    eden_utils::compute_codes(
                            sq, metric, scale, x.data(), codes.data(), 16);
                    SIMDConfig::set_level(SIMDLevel::NONE);
                    std::unique_ptr<EDENFlatCodesDistanceComputer> ref(
                            eden_utils::get_distance_computer(sq, metric));
                    SIMDConfig::set_level(SL);
                    std::unique_ptr<EDENFlatCodesDistanceComputer> simd(
                            eden_utils::get_distance_computer(sq, metric));
                    ref->codes = simd->codes = codes.data();
                    ref->code_size = simd->code_size = cs;
                    ref->set_query(q.data());
                    simd->set_query(q.data());
                    float actual[16];
                    simd->consecutive_distances_batch_16(0, actual);
                    for (int j = 0; j < 16; ++j) {
                        float expected = (*ref)(j);
                        EXPECT_NEAR(
                                expected,
                                actual[j],
                                1e-4f * (1 + std::abs(expected)))
                                << bits << " " << d;
                    }
                }
        }
    for (int bits = 2; bits <= 5; ++bits)
        for (size_t d : {1, 3, 7, 8, 9, 16, 17, 32, 65})
            for (int projection : {0, 2}) {
                SCOPED_TRACE(bits);
                SCOPED_TRACE(d);
                SCOPED_TRACE(projection);
                auto qt = QT(int(ScalarQuantizer::QT_2bit_tq) + bits - 2);
                ScalarQuantizer sq(d, qt);
                int k = 1 << (bits - 1);
                sq.trained.resize(2 * k + 2);
                for (int j = 0; j < k; ++j)
                    sq.trained[j] = (float(j) - float(k - 1) / 2) * 0.5f;
                for (int j = 0; j < k - 1; ++j)
                    sq.trained[k + j] =
                            (sq.trained[j] + sq.trained[j + 1]) * 0.5f;
                ScalarQuantizer::TurboQuantRefine::pack_seed(
                        123, sq.trained.data() + 2 * k - 1);
                sq.trained[2 * k + 1] = projection;
                std::unique_ptr<ScalarQuantizer::SQuantizer> ref(
                        scalar_quantizer::sq_select_quantizer<SIMDLevel::NONE>(
                                qt, d, sq.trained));
                std::unique_ptr<ScalarQuantizer::SQuantizer> simd(
                        scalar_quantizer::sq_select_quantizer<SL>(
                                qt, d, sq.trained));
                ASSERT_NE(simd, nullptr);
                for (int trial = 0; trial < 5; ++trial) {
                    std::vector<float> x(d), q(d), a(d + 1, 77), b(d + 1, 77);
                    for (size_t j = 0; j < d; ++j) {
                        x[j] = trial ? float(int(rng() % 2001) - 1000) / 100
                                     : 0;
                        q[j] = float(int(rng() % 2001) - 1000) / 100;
                    }
                    std::vector<uint8_t> ca(sq.code_size + 1, 0), cb(ca);
                    ca.back() = cb.back() = 0xa5;
                    ref->encode_vector(x.data(), ca.data());
                    simd->encode_vector(x.data(), cb.data());
                    // Packed bit planes are exact; residual norms may differ by
                    // the
                    // final rounding of a fused multiply-subtract.
                    EXPECT_TRUE(
                            std::equal(
                                    ca.begin(),
                                    ca.begin() + sq.code_size - 8,
                                    cb.begin()));
                    EXPECT_EQ(ca.back(), cb.back());
                    float ref_factors[2], simd_factors[2];
                    memcpy(ref_factors, ca.data() + sq.code_size - 8, 8);
                    memcpy(simd_factors, cb.data() + sq.code_size - 8, 8);
                    for (int f = 0; f < 2; ++f)
                        EXPECT_NEAR(
                                ref_factors[f],
                                simd_factors[f],
                                2e-6f * (1 + std::abs(ref_factors[f])));
                    ref->decode_vector(ca.data(), a.data());
                    simd->decode_vector(ca.data(), b.data());
                    for (size_t j = 0; j <= d; ++j)
                        EXPECT_NEAR(a[j], b[j], 1e-5 * (1 + std::abs(a[j])));
                    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
                        std::unique_ptr<ScalarQuantizer::SQDistanceComputer>
                                rdc(scalar_quantizer::
                                            sq_select_distance_computer<
                                                    SIMDLevel::NONE>(
                                                    metric, qt, d, sq.trained));
                        std::unique_ptr<ScalarQuantizer::SQDistanceComputer>
                                vdc(scalar_quantizer::
                                            sq_select_distance_computer<SL>(
                                                    metric, qt, d, sq.trained));
                        ASSERT_NE(vdc, nullptr);
                        auto* r =
                                dynamic_cast<ScalarQuantizer::TurboQuantRefine::
                                                     DistanceComputer*>(
                                        rdc.get());
                        auto* v =
                                dynamic_cast<ScalarQuantizer::TurboQuantRefine::
                                                     DistanceComputer*>(
                                        vdc.get());
                        ASSERT_NE(v, nullptr);
                        for (uint8_t qb : {0, 1, 4, 8})
                            for (bool iq : {false, true}) {
                                r->configure(qb, iq);
                                v->configure(qb, iq);
                                r->set_query(q.data());
                                v->set_query(q.data());
                                float expected = r->query_to_code(ca.data());
                                EXPECT_NEAR(
                                        expected,
                                        v->query_to_code(ca.data()),
                                        2e-4 * (1 + std::abs(expected)))
                                        << int(qb) << " " << iq;
                                float threshold =
                                        metric == METRIC_L2 ? -1e20f : 1e20f;
                                r->set_prescreen_threshold(
                                        &threshold, metric == METRIC_L2);
                                v->set_prescreen_threshold(
                                        &threshold, metric == METRIC_L2);
                                EXPECT_EQ(
                                        r->query_to_code(ca.data()),
                                        v->query_to_code(ca.data()));
                                r->clear_prescreen_threshold();
                                v->clear_prescreen_threshold();
                            }
                    }
                }
            }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchScalarQuantizer, AdvancedLSX) {
    check_advanced_codecs<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchScalarQuantizer, AdvancedLASX) {
    check_advanced_codecs<SIMDLevel::LOONGARCH_LASX>();
}
#endif

namespace {
template <SIMDLevel SL>
void check_preprocessing() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    using namespace quantizer_preprocessing;
    std::mt19937 rng(529);
    for (size_t d = 0; d <= 65; ++d) {
        SCOPED_TRACE(d);
        std::vector<float> x(d + 2, 123), c(d + 2), q(d + 2);
        for (size_t j = 1; j <= d; ++j) {
            x[j] = float(int(rng() % 1001) - 500) / 31;
            c[j] = float(int(rng() % 1001) - 500) / 83;
            q[j] = j % 2 ? 1024.125f : 0.03125f;
        }
        for (const float* center :
             {static_cast<const float*>(nullptr),
              static_cast<const float*>(c.data() + 1)}) {
            double rn, ri, an, ai;
            eden_code_stats<SIMDLevel::NONE>(
                    d, x.data() + 1, center, q.data() + 1, rn, ri);
            eden_code_stats<SL>(d, x.data() + 1, center, q.data() + 1, an, ai);
            EXPECT_NEAR(rn, an, 1e-13 * (1 + rn));
            EXPECT_NEAR(ri, ai, 1e-12 * (1 + std::abs(ri)));
            std::vector<float> a(d + 2, 123), b(a);
            eden_normalize<SIMDLevel::NONE>(
                    d, x.data() + 1, center, 3.15f, 0.125f, a.data() + 1);
            eden_normalize<SL>(
                    d, x.data() + 1, center, 3.15f, 0.125f, b.data() + 1);
            EXPECT_EQ(a, b);
            eden_reconstruct<SIMDLevel::NONE>(
                    d, x.data() + 1, center, 1.7f, a.data() + 1);
            eden_reconstruct<SL>(d, x.data() + 1, center, 1.7f, b.data() + 1);
            for (size_t j = 0; j < a.size(); ++j)
                EXPECT_NEAR(a[j], b[j], 2e-6 * (1 + std::abs(a[j])));
        }
        for (int trial = 0; trial < 5; ++trial) {
            if (trial == 1 && d)
                x[1] = std::numeric_limits<float>::quiet_NaN();
            if (trial == 2) {
                std::fill(x.begin() + 1, x.end() - 1, 0.0f);
                if (d > 1) {
                    x[1] = -0.0f;
                    x[2] = 0.0f;
                }
            }
            if (trial == 3) {
                std::fill(
                        x.begin() + 1,
                        x.end() - 1,
                        std::numeric_limits<float>::quiet_NaN());
                if (d > 9) {
                    x[3] = -0.0f;
                    x[9] = 0.0f;
                }
            }
            if (trial == 4) {
                std::fill(
                        x.begin() + 1,
                        x.end() - 1,
                        std::numeric_limits<float>::infinity());
                if (d > 8)
                    x[8] = -std::numeric_limits<float>::infinity();
            }
            float rl, rh, al, ah;
            rowwise_minmax<SIMDLevel::NONE>(d, x.data() + 1, rl, rh);
            rowwise_minmax<SL>(d, x.data() + 1, al, ah);
            EXPECT_EQ(std::bit_cast<uint32_t>(rl), std::bit_cast<uint32_t>(al));
            EXPECT_EQ(std::bit_cast<uint32_t>(rh), std::bit_cast<uint32_t>(ah));
            for (float scale : {0.0f, 1.3f}) {
                std::vector<float> a(d + 2, 123), b(a);
                rowwise_normalize<SIMDLevel::NONE>(
                        d, x.data() + 1, rl, scale, a.data() + 1);
                rowwise_normalize<SL>(d, x.data() + 1, al, scale, b.data() + 1);
                for (size_t j = 0; j < a.size(); ++j) {
                    if (std::isnan(a[j]))
                        EXPECT_TRUE(std::isnan(b[j]));
                    else
                        EXPECT_EQ(
                                std::bit_cast<uint32_t>(a[j]),
                                std::bit_cast<uint32_t>(b[j]));
                }
                // Training normalizes in place.
                auto inplace = x;
                rowwise_normalize<SL>(
                        d, inplace.data() + 1, al, scale, inplace.data() + 1);
                for (size_t j = 1; j <= d; ++j) {
                    if (std::isnan(a[j]))
                        EXPECT_TRUE(std::isnan(inplace[j]));
                    else
                        EXPECT_EQ(a[j], inplace[j]);
                }
            }
        }
    }
    struct Restore {
        SIMDLevel level = SIMDConfig::get_level();
        ~Restore() {
            SIMDConfig::set_level(level);
        }
    } restore;
    for (int d : {1, 7, 8, 9, 31, 65}) {
        std::vector<float> x(4 * d);
        for (size_t j = 0; j < x.size(); ++j)
            x[j] = j < size_t(d) ? 1.25f : float(int(rng() % 1000) - 500) / 73;
        IndexFlatL2 flat(d);
        IndexRowwiseMinMax fp32(&flat);
        IndexRowwiseMinMaxFP16 fp16(&flat);
        for (IndexRowwiseMinMaxBase* index :
             {static_cast<IndexRowwiseMinMaxBase*>(&fp32),
              static_cast<IndexRowwiseMinMaxBase*>(&fp16)}) {
            std::vector<uint8_t> a(4 * index->sa_code_size()), b(a);
            SIMDConfig::set_level(SIMDLevel::NONE);
            index->sa_encode(4, x.data(), a.data());
            auto ref_train = x;
            std::vector<float> ref_decode(x.size());
            index->train_inplace(4, ref_train.data());
            index->sa_decode(4, a.data(), ref_decode.data());
            SIMDConfig::set_level(SL);
            index->sa_encode(4, x.data(), b.data());
            EXPECT_EQ(a, b);
            auto tx = x;
            index->train_inplace(4, tx.data());
            std::vector<float> decoded(x.size());
            index->sa_decode(4, b.data(), decoded.data());
            for (size_t j = 0; j < x.size(); ++j) {
                EXPECT_NEAR(
                        ref_train[j],
                        tx[j],
                        2e-6 * (1 + std::abs(ref_train[j])));
                EXPECT_NEAR(
                        ref_decode[j],
                        decoded[j],
                        2e-6 * (1 + std::abs(ref_decode[j])));
            }
        }
    }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchScalarQuantizer, PreprocessingLSX) {
    check_preprocessing<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchScalarQuantizer, PreprocessingLASX) {
    check_preprocessing<SIMDLevel::LOONGARCH_LASX>();
}
#endif

namespace {
template <SIMDLevel SL>
void check_fp16_rounding() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    ScalarQuantizer sq(8, ScalarQuantizer::QT_fp16);
    std::unique_ptr<ScalarQuantizer::SQuantizer> codec(
            scalar_quantizer::sq_select_quantizer<SL>(
                    sq.qtype, sq.d, sq.trained));
    // Every finite positive half interval, mirrored for negative values.
    // Include the neighbors of each midpoint, subnormals, and overflow.
    for (uint32_t h = 0; h < 0x7c00; ++h) {
        const float lo = decode_fp16(h);
        const float hi = h == 0x7bff ? 65536.0f : decode_fp16(h + 1);
        const float mid = (lo + hi) * 0.5f;
        const float before = std::nextafter(mid, 0.0f);
        const float after =
                std::nextafter(mid, std::numeric_limits<float>::infinity());
        const float values[8] = {
                lo, before, mid, after, -lo, -before, -mid, -after};
        uint16_t actual[8];
        codec->encode_vector(values, reinterpret_cast<uint8_t*>(actual));
        for (int j = 0; j < 8; ++j)
            ASSERT_EQ(actual[j], encode_fp16(values[j])) << h << " " << j;
    }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchScalarQuantizer, Fp16RoundingLSX) {
    check_fp16_rounding<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchScalarQuantizer, Fp16RoundingLASX) {
    check_fp16_rounding<SIMDLevel::LOONGARCH_LASX>();
}
#endif
