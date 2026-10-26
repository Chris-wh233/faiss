/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_impl/loongarch_bits.h>
#include <cstring>
namespace faiss::rabitq {
namespace {
using Bits = detail::LoongArchBits<THE_SIMD_LEVEL>;
template <size_t N>
__m128i loongarch_load_short_code(const uint8_t* p) {
    static_assert(N == 4 || N == 8);
    uint64_t word = 0;
    std::memcpy(&word, p, N);
    return __lsx_vinsgr2vr_d(__lsx_vldi(0), word, 0);
}

// Reduce each register into uint64_t before weighting; narrow lane
// accumulators can overflow for long codes or high query bit counts.
template <bool use_xor, bool count_data>
BitwiseAndDotProductResult loongarch_bitwise_dot(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    uint64_t sum = 0, count = 0;
    size_t i = 0;
    auto block = [&]<typename B, size_t N>() {
        auto load = [](const uint8_t* p) {
            if constexpr (N == B::width)
                return B::load(p);
            else
                return loongarch_load_short_code<N>(p);
        };
        const auto value = load(data + i);
        if constexpr (count_data)
            count += B::popcount(value);
        for (size_t j = 0; j < qb; ++j) {
            const auto q = load(query + j * size + i);
            const auto combined =
                    use_xor ? B::bit_xor(value, q) : B::bit_and(value, q);
            sum += B::popcount(combined) << j;
        }
        i += N;
    };
    for (; i + Bits::width <= size;)
        block.template operator()<Bits, Bits::width>();
    // LASX also uses narrow LSX blocks. Constant-size partial loads are safe
    // for short codes and let the compiler avoid a runtime-sized memcpy.
    using Narrow = detail::LoongArchBits<SIMDLevel::LOONGARCH_LSX>;
    if constexpr (Bits::width > Narrow::width) {
        if (i + 16 <= size)
            block.template operator()<Narrow, 16>();
    }
    if (i + 8 <= size)
        block.template operator()<Narrow, 8>();
    if (i + 4 <= size)
        block.template operator()<Narrow, 4>();
    for (; i < size; ++i) {
        if constexpr (count_data)
            count += popcount32(data[i]);
        for (size_t j = 0; j < qb; ++j) {
            unsigned value = use_xor ? data[i] ^ query[j * size + i]
                                     : data[i] & query[j * size + i];
            sum += uint64_t(popcount32(value)) << j;
        }
    }
    return {sum, count};
}
} // namespace
template <>
uint64_t bitwise_and_dot_product<THE_SIMD_LEVEL>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return loongarch_bitwise_dot<false, false>(query, data, size, qb)
            .dot_product;
}
template <>
uint64_t bitwise_xor_dot_product<THE_SIMD_LEVEL>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return loongarch_bitwise_dot<true, false>(query, data, size, qb)
            .dot_product;
}
template <>
BitwiseAndDotProductResult bitwise_and_dot_product_with_popcount<
        THE_SIMD_LEVEL>(
        const uint8_t* query,
        const uint8_t* data,
        size_t size,
        size_t qb) {
    return loongarch_bitwise_dot<false, true>(query, data, size, qb);
}
template <>
uint64_t popcount<THE_SIMD_LEVEL>(const uint8_t* data, size_t size) {
    uint64_t result = 0;
    size_t i = 0;
    for (; i + Bits::width <= size; i += Bits::width)
        result += Bits::popcount(Bits::load(data + i));
    using Narrow = detail::LoongArchBits<SIMDLevel::LOONGARCH_LSX>;
    if constexpr (Bits::width > Narrow::width) {
        if (i + 16 <= size) {
            result += Narrow::popcount(Narrow::load(data + i));
            i += 16;
        }
    }
    if (i + 8 <= size) {
        result += Narrow::popcount(loongarch_load_short_code<8>(data + i));
        i += 8;
    }
    if (i + 4 <= size) {
        result += Narrow::popcount(loongarch_load_short_code<4>(data + i));
        i += 4;
    }
    for (; i < size; ++i)
        result += popcount32(data[i]);
    return result;
}
template <>
void rearrange_bit_planes<THE_SIMD_LEVEL>(
        const uint8_t* values,
        size_t d,
        size_t qb,
        uint8_t* out) {
    const size_t stride = (d + 7) / 8;
    if (stride == 0)
        return;
    std::memset(out, 0, stride * qb);
    size_t i = 0;
    for (; i + Bits::width <= d; i += Bits::width) {
        const auto value = Bits::load(values + i);
        for (size_t j = 0; j < qb; ++j) {
            const uint32_t mask = Bits::bit_plane(value, j);
            std::memcpy(out + j * stride + i / 8, &mask, Bits::width / 8);
        }
    }
    for (; i < d; ++i)
        for (size_t j = 0; j < qb; ++j)
            out[j * stride + i / 8] |= ((values[i] >> j) & 1) << (i % 8);
}
} // namespace faiss::rabitq

namespace faiss::rabitq::multibit {
#ifdef COMPILE_SIMD_LOONGARCH_LSX
#if defined(FAISS_RABITQ_LSX)
template <>
float compute_inner_product<SIMDLevel::LOONGARCH_LSX>(
        const uint8_t* sign_bits,
        const uint8_t* ex_code,
        const float* rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>;
    Vec acc(0.0f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        uint64_t packed = 0;
        memcpy(&packed, ex_code + i * ex_bits / 8, ex_bits);
        const uint32_t words[2] = {
                uint32_t(packed), uint32_t(packed >> (4 * ex_bits))};
        Vec decoded;
        for (int j = 0; j < 2; ++j) {
            const uint32_t offsets[4] = {
                    0,
                    uint32_t(ex_bits),
                    uint32_t(2 * ex_bits),
                    uint32_t(3 * ex_bits)};
            auto extra = __lsx_vsrl_w(
                    __lsx_vreplgr2vr_w(words[j]), __lsx_vld(offsets, 0));
            extra = __lsx_vand_v(
                    extra, __lsx_vreplgr2vr_w((1u << ex_bits) - 1));
            const uint32_t sign_offsets[4] = {
                    uint32_t(j * 4),
                    uint32_t(j * 4 + 1),
                    uint32_t(j * 4 + 2),
                    uint32_t(j * 4 + 3)};
            auto sign = __lsx_vsrl_w(
                    __lsx_vreplgr2vr_w(sign_bits[i / 8]),
                    __lsx_vld(sign_offsets, 0));
            sign = __lsx_vand_v(sign, __lsx_vreplgr2vr_w(1));
            sign = __lsx_vsll_w(sign, __lsx_vreplgr2vr_w(ex_bits));
            decoded.data[j] =
                    (__m128i)__lsx_vffint_s_w(__lsx_vadd_w(sign, extra));
        }
        acc = acc + Vec(rotated_q + i) * (decoded + Vec(cb));
    }
    float result = horizontal_add(acc);
    for (; i < d; ++i) {
        const size_t bit = i * ex_bits;
        uint32_t value = ex_code[bit / 8] >> (bit % 8);
        if (bit % 8 + ex_bits > 8)
            value |= uint32_t(ex_code[bit / 8 + 1]) << (8 - bit % 8);
        value &= (1u << ex_bits) - 1;
        value += ((sign_bits[i / 8] >> (i % 8)) & 1u) << ex_bits;
        result += rotated_q[i] * (float(value) + cb);
    }
    return result;
}
#endif
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
#if defined(FAISS_RABITQ_LASX)
template <>
float compute_inner_product<SIMDLevel::LOONGARCH_LASX>(
        const uint8_t* sign_bits,
        const uint8_t* ex_code,
        const float* rotated_q,
        size_t d,
        size_t ex_bits,
        float cb) {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>;
    Vec acc(0.0f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        uint64_t packed = 0;
        memcpy(&packed, ex_code + i * ex_bits / 8, ex_bits);
        const uint32_t words[2] = {
                uint32_t(packed), uint32_t(packed >> (4 * ex_bits))};
        Vec decoded;
        const uint32_t offsets[8] = {
                0,
                uint32_t(ex_bits),
                uint32_t(2 * ex_bits),
                uint32_t(3 * ex_bits),
                0,
                uint32_t(ex_bits),
                uint32_t(2 * ex_bits),
                uint32_t(3 * ex_bits)};
        auto packed_words = __lasx_xvpermi_q(
                __lasx_xvreplgr2vr_w(words[1]),
                __lasx_xvreplgr2vr_w(words[0]),
                0x20);
        auto extra = __lasx_xvsrl_w(packed_words, __lasx_xvld(offsets, 0));
        extra = __lasx_xvand_v(
                extra, __lasx_xvreplgr2vr_w((1u << ex_bits) - 1));
        const uint32_t sign_offsets[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        auto sign = __lasx_xvsrl_w(
                __lasx_xvreplgr2vr_w(sign_bits[i / 8]),
                __lasx_xvld(sign_offsets, 0));
        sign = __lasx_xvand_v(sign, __lasx_xvreplgr2vr_w(1));
        sign = __lasx_xvsll_w(sign, __lasx_xvreplgr2vr_w(ex_bits));
        decoded.data = (__m256i)__lasx_xvffint_s_w(__lasx_xvadd_w(sign, extra));
        acc = acc + Vec(rotated_q + i) * (decoded + Vec(cb));
    }
    float result = horizontal_add(acc);
    for (; i < d; ++i) {
        const size_t bit = i * ex_bits;
        uint32_t value = ex_code[bit / 8] >> (bit % 8);
        if (bit % 8 + ex_bits > 8)
            value |= uint32_t(ex_code[bit / 8 + 1]) << (8 - bit % 8);
        value &= (1u << ex_bits) - 1;
        value += ((sign_bits[i / 8] >> (i % 8)) & 1u) << ex_bits;
        result += rotated_q[i] * (float(value) + cb);
    }
    return result;
}
#endif
#endif
} // namespace faiss::rabitq::multibit
