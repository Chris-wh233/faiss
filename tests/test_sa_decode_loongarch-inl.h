/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Included only by the native LSX/LASX test translation units.
#include <faiss/cppcontrib/SaDecodeKernels.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
namespace {
using namespace faiss;
using namespace faiss::cppcontrib;
void put_code(uint8_t* code, int index, int bits, unsigned value) {
    for (int b = 0; b < bits; ++b)
        if (value & (1u << b))
            code[(index * bits + b) / 8] |= 1u << ((index * bits + b) % 8);
}
unsigned get_code(const uint8_t* code, int index, int bits) {
    unsigned value = 0;
    for (int b = 0; b < bits; ++b)
        value |= unsigned(
                         (code[(index * bits + b) / 8] >>
                          ((index * bits + b) % 8)) &
                         1)
                << b;
    return value;
}
void expect_output(
        const std::vector<float>& actual,
        const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual.front(), 77);
    EXPECT_EQ(actual.back(), 77);
    for (size_t j = 1; j + 1 < actual.size(); ++j)
        EXPECT_NEAR(
                actual[j], expected[j], 2e-5f * (1 + std::abs(expected[j])));
}
template <int D, int FS, int FB>
void check_pq() {
    using Decoder = IndexPQDecoder<D, FS, FB>;
    constexpr int fine_bytes = (D / FS * FB + 7) / 8;
    constexpr int coarse_bytes = 0, code_size = fine_bytes + coarse_bytes;
    std::vector<float> fine[3];
    std::vector<uint8_t> codes[3];
    std::vector<float> decoded[3];
    for (int c = 0; c < 3; ++c) {
        fine[c].resize(D * (1 << FB) + 1);
        for (size_t i = 1; i < fine[c].size(); ++i)
            fine[c][i] = float(int((i * 17 + c * 11) % 101) - 50) / 127;
        codes[c] = std::vector<uint8_t>(code_size + 1, 0);
        auto* code = codes[c].data() + 1;
        for (int m = 0; m < D / FS; ++m)
            put_code(
                    code + coarse_bytes,
                    m,
                    FB,
                    (m * 311 + c * 997) & ((1 << FB) - 1));
        decoded[c].resize(D);
        for (int j = 0; j < D; ++j) {
            unsigned idx = get_code(code + coarse_bytes, j / FS, FB);
            decoded[c][j] =
                    fine[c][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
        }
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        std::copy(decoded[c].begin(), decoded[c].end(), expected.begin() + 1);
        Decoder::store(fine[c].data() + 1, code, actual.data() + 1);
        expect_output(actual, expected);
    }
    const float weights[3] = {0.25f, -0.75f, 1.5f};
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 1; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                fine[1].data() + 1,
                codes[1].data() + 1,
                weights[1],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                unsigned idx = get_code(
                        codes[c].data() + 1 + coarse_bytes, j / FS, FB);
                value = fine[0][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                codes[1].data() + 1,
                weights[1],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                fine[1].data() + 1,
                codes[1].data() + 1,
                weights[1],
                fine[2].data() + 1,
                codes[2].data() + 1,
                weights[2],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                unsigned idx = get_code(
                        codes[c].data() + 1 + coarse_bytes, j / FS, FB);
                value = fine[0][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                codes[1].data() + 1,
                weights[1],
                codes[2].data() + 1,
                weights[2],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        using Wrapped = IndexMinMaxDecoder<Decoder>;
        std::vector<uint8_t> code(code_size + 8 + 1);
        uint8_t* bytes = code.data() + 1;
        float factors[2] = {1.5f, -0.25f};
        memcpy(bytes, factors, 8);
        memcpy(bytes + 8, codes[0].data() + 1, code_size);
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            expected[j + 1] = std::fma(decoded[0][j], 1.5f, -0.25f);
        Wrapped::store(fine[0].data() + 1, bytes, actual.data() + 1);
        expect_output(actual, expected);
        for (int j = 0; j < D; ++j) {
            actual[j + 1] = 0.125f;
            expected[j + 1] = std::fma(0.75f, decoded[0][j], 0.125f);
        }
        float bias = 0.25f;
        Wrapped::accum(
                fine[0].data() + 1, bytes, 0.5f, actual.data() + 1, bias);
        expect_output(actual, expected);
        EXPECT_FLOAT_EQ(bias, 0.125f);
    }
    {
        using Wrapped = IndexMinMaxFP16Decoder<Decoder>;
        std::vector<uint8_t> code(code_size + 4 + 1);
        uint8_t* bytes = code.data() + 1;
        uint16_t factors[2] = {encode_fp16(1.5f), encode_fp16(-0.25f)};
        memcpy(bytes, factors, 4);
        memcpy(bytes + 4, codes[0].data() + 1, code_size);
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            expected[j + 1] = std::fma(decoded[0][j], 1.5f, -0.25f);
        Wrapped::store(fine[0].data() + 1, bytes, actual.data() + 1);
        expect_output(actual, expected);
        for (int j = 0; j < D; ++j) {
            actual[j + 1] = 0.125f;
            expected[j + 1] = std::fma(0.75f, decoded[0][j], 0.125f);
        }
        float bias = 0.25f;
        Wrapped::accum(
                fine[0].data() + 1, bytes, 0.5f, actual.data() + 1, bias);
        expect_output(actual, expected);
        EXPECT_FLOAT_EQ(bias, 0.125f);
    }
}
template <int D, int CS, int FS, int CB, int FB>
void check_level2() {
    using Decoder = Index2LevelDecoder<D, CS, FS, CB, FB>;
    constexpr int fine_bytes = (D / FS * FB + 7) / 8;
    constexpr int coarse_bytes = (D / CS * CB + 7) / 8,
                  code_size = fine_bytes + coarse_bytes;
    std::vector<float> fine[3];
    std::vector<uint8_t> codes[3];
    std::vector<float> decoded[3];
    std::vector<float> coarse[3];
    for (int c = 0; c < 3; ++c) {
        fine[c].resize(D * (1 << FB) + 1);
        for (size_t i = 1; i < fine[c].size(); ++i)
            fine[c][i] = float(int((i * 17 + c * 11) % 101) - 50) / 127;
        codes[c] = std::vector<uint8_t>(code_size + 1, 0);
        auto* code = codes[c].data() + 1;
        for (int m = 0; m < D / FS; ++m)
            put_code(
                    code + coarse_bytes,
                    m,
                    FB,
                    (m * 311 + c * 997) & ((1 << FB) - 1));
        coarse[c].resize(D * (1 << CB) + 1);
        for (size_t i = 1; i < coarse[c].size(); ++i)
            coarse[c][i] = float(int((i * 13 + c * 7) % 109) - 54) / 121;
        for (int m = 0; m < D / CS; ++m)
            put_code(code, m, CB, (m * 641 + c * 239) & ((1 << CB) - 1));
        decoded[c].resize(D);
        for (int j = 0; j < D; ++j) {
            unsigned idx = get_code(code + coarse_bytes, j / FS, FB);
            decoded[c][j] =
                    fine[c][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
            idx = get_code(code, j / CS, CB);
            decoded[c][j] +=
                    coarse[c][1 + ((j / CS) * (1 << CB) + idx) * CS + j % CS];
        }
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        std::copy(decoded[c].begin(), decoded[c].end(), expected.begin() + 1);
        Decoder::store(
                coarse[c].data() + 1,
                fine[c].data() + 1,
                code,
                actual.data() + 1);
        expect_output(actual, expected);
    }
    const float weights[3] = {0.25f, -0.75f, 1.5f};
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 1; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                coarse[1].data() + 1,
                fine[1].data() + 1,
                codes[1].data() + 1,
                weights[1],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                unsigned idx = get_code(
                        codes[c].data() + 1 + coarse_bytes, j / FS, FB);
                value = fine[0][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
                idx = get_code(codes[c].data() + 1, j / CS, CB);
                value += coarse[0]
                               [1 + ((j / CS) * (1 << CB) + idx) * CS + j % CS];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                codes[1].data() + 1,
                weights[1],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                value = decoded[c][j];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                coarse[1].data() + 1,
                fine[1].data() + 1,
                codes[1].data() + 1,
                weights[1],
                coarse[2].data() + 1,
                fine[2].data() + 1,
                codes[2].data() + 1,
                weights[2],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            actual[j + 1] = expected[j + 1] = 0.125f;
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < D; ++j) {
                float value;
                unsigned idx = get_code(
                        codes[c].data() + 1 + coarse_bytes, j / FS, FB);
                value = fine[0][1 + ((j / FS) * (1 << FB) + idx) * FS + j % FS];
                idx = get_code(codes[c].data() + 1, j / CS, CB);
                value += coarse[0]
                               [1 + ((j / CS) * (1 << CB) + idx) * CS + j % CS];
                expected[j + 1] = std::fma(weights[c], value, expected[j + 1]);
            }
        Decoder::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                codes[0].data() + 1,
                weights[0],
                codes[1].data() + 1,
                weights[1],
                codes[2].data() + 1,
                weights[2],
                actual.data() + 1);
        expect_output(actual, expected);
    }
    {
        using Wrapped = IndexMinMaxDecoder<Decoder>;
        std::vector<uint8_t> code(code_size + 8 + 1);
        uint8_t* bytes = code.data() + 1;
        float factors[2] = {1.5f, -0.25f};
        memcpy(bytes, factors, 8);
        memcpy(bytes + 8, codes[0].data() + 1, code_size);
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            expected[j + 1] = std::fma(decoded[0][j], 1.5f, -0.25f);
        Wrapped::store(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                bytes,
                actual.data() + 1);
        expect_output(actual, expected);
        for (int j = 0; j < D; ++j) {
            actual[j + 1] = 0.125f;
            expected[j + 1] = std::fma(0.75f, decoded[0][j], 0.125f);
        }
        float bias = 0.25f;
        Wrapped::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                bytes,
                0.5f,
                actual.data() + 1,
                bias);
        expect_output(actual, expected);
        EXPECT_FLOAT_EQ(bias, 0.125f);
    }
    {
        using Wrapped = IndexMinMaxFP16Decoder<Decoder>;
        std::vector<uint8_t> code(code_size + 4 + 1);
        uint8_t* bytes = code.data() + 1;
        uint16_t factors[2] = {encode_fp16(1.5f), encode_fp16(-0.25f)};
        memcpy(bytes, factors, 4);
        memcpy(bytes + 4, codes[0].data() + 1, code_size);
        std::vector<float> actual(D + 2, 77), expected(D + 2, 77);
        for (int j = 0; j < D; ++j)
            expected[j + 1] = std::fma(decoded[0][j], 1.5f, -0.25f);
        Wrapped::store(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                bytes,
                actual.data() + 1);
        expect_output(actual, expected);
        for (int j = 0; j < D; ++j) {
            actual[j + 1] = 0.125f;
            expected[j + 1] = std::fma(0.75f, decoded[0][j], 0.125f);
        }
        float bias = 0.25f;
        Wrapped::accum(
                coarse[0].data() + 1,
                fine[0].data() + 1,
                bytes,
                0.5f,
                actual.data() + 1,
                bias);
        expect_output(actual, expected);
        EXPECT_FLOAT_EQ(bias, 0.125f);
    }
}
} // namespace
void TEST_LOONGARCH_SA_DECODE() {
    check_pq<1, 1, 8>();
    check_pq<24, 3, 8>();
    check_pq<40, 2, 10>();
    check_pq<30, 5, 12>();
    check_pq<24, 8, 16>();
    check_level2<1, 1, 1, 8, 8>();
    check_level2<24, 12, 3, 8, 8>();
    check_level2<40, 10, 2, 10, 10>();
    check_level2<30, 15, 5, 12, 12>();
    check_level2<24, 24, 8, 16, 16>();
}
