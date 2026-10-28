/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/cppcontrib/sa_decode/MinMax-inl.h>
#include <faiss/utils/simd_levels.h>
#if defined(__loongarch_asx)
#include <faiss/impl/simdlib/simdlib_lasx.h>
#elif defined(__loongarch_sx)
#include <faiss/impl/simdlib/simdlib_lsx.h>
#endif
#include <algorithm>
#include <cstring>
namespace faiss::cppcontrib::detail {
#if defined(__loongarch_asx)
static constexpr SIMDLevel loongarch_decode_level = SIMDLevel::LOONGARCH_LASX;
#else
static constexpr SIMDLevel loongarch_decode_level = SIMDLevel::LOONGARCH_LSX;
#endif
// Read exactly the bytes of a packed field; code pointers may be unaligned.
template <int Bits>
inline uint32_t loongarch_read_code(const uint8_t* code, intptr_t index) {
    const intptr_t bit = index * Bits, shift = bit % 8;
    uint32_t word = 0;
    std::memcpy(&word, code + bit / 8, (shift + Bits + 7) / 8);
    return (word >> shift) & ((1u << Bits) - 1);
}
template <SIMDLevel SL, intptr_t Size, int Bits>
simd8float32_tpl<SL> loongarch_decode_block(
        const float* table,
        const uint8_t* code,
        intptr_t pos,
        intptr_t count) {
    using Vec = simd8float32_tpl<SL>;
    if (count == 8 && pos % Size + 8 <= Size) {
        intptr_t sub = pos / Size;
        return Vec(
                table +
                (sub * (intptr_t(1) << Bits) +
                 loongarch_read_code<Bits>(code, sub)) *
                        Size +
                pos % Size);
    }
    // Neither ISA has a gather instruction. Copy each contiguous centroid slice
    // before the vector arithmetic; no slice reads beyond a centroid or code.
    float values[8] = {};
    intptr_t lane = 0;
    while (lane < count) {
        intptr_t i = pos + lane, sub = i / Size,
                 n = std::min(count - lane, Size - i % Size);
        const float* src = table +
                (sub * (intptr_t(1) << Bits) +
                 loongarch_read_code<Bits>(code, sub)) *
                        Size +
                i % Size;
        std::memcpy(values + lane, src, n * sizeof(float));
        lane += n;
    }
    return Vec(values);
}
template <SIMDLevel SL>
simd8float32_tpl<SL> loongarch_load_output(const float* x, intptr_t count) {
    if (count == 8)
        return simd8float32_tpl<SL>(x);
    float values[8] = {};
    std::memcpy(values, x, count * sizeof(float));
    return simd8float32_tpl<SL>(values);
}
template <SIMDLevel SL>
void loongarch_store_output(simd8float32_tpl<SL> v, float* x, intptr_t count) {
    if (count == 8) {
        v.storeu(x);
        return;
    }
    float values[8];
    v.storeu(values);
    std::memcpy(x, values, count * sizeof(float));
}
template <SIMDLevel SL>
struct LoongArchMinMaxOps {
    static void affine(float* x, intptr_t d, float scale, float bias) {
        using Vec = simd8float32_tpl<SL>;
        for (intptr_t i = 0; i < d; i += 8) {
            intptr_t n = std::min(intptr_t(8), d - i);
            auto value = loongarch_load_output<SL>(x + i, n);
            loongarch_store_output<SL>(
                    fmadd(value, Vec(scale), Vec(bias)), x + i, n);
        }
    }
};
} // namespace faiss::cppcontrib::detail
