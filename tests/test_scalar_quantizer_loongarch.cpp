/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ScalarQuantizer.h>
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
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchSQ, LSXCodec) {
    check_codec<faiss::SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchSQ, LASXCodec) {
    check_codec<faiss::SIMDLevel::LOONGARCH_LASX>();
}
#endif
