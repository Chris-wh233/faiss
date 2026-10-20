/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/utils/simd_levels.h>
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
