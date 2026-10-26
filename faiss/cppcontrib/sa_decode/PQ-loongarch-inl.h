/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/cppcontrib/sa_decode/loongarch-inl.h>
namespace faiss::cppcontrib {
template <
        intptr_t DIM,
        intptr_t FINE_SIZE,
        intptr_t FINE_BITS = 8,
        SIMDLevel SL = detail::loongarch_decode_level>
struct IndexPQDecoder {
    static_assert(DIM > 0 && FINE_SIZE > 0 && DIM % FINE_SIZE == 0);
    static_assert(
            FINE_BITS == 8 || FINE_BITS == 10 || FINE_BITS == 12 ||
            FINE_BITS == 16);
    static constexpr intptr_t dim = DIM, fineSize = FINE_SIZE,
                              fineBits = FINE_BITS;
    static constexpr SIMDLevel simd_level = SL;
    using Vec = simd8float32_tpl<SL>;

   private:
    static Vec block(
            const float* fine,
            const uint8_t* code,
            intptr_t pos,
            intptr_t count) {
        return detail::loongarch_decode_block<SL, FINE_SIZE, FINE_BITS>(
                fine, code, pos, count);
    }
    template <int N>
    static void accumulate(
            const float* const (&fine)[N],
            const uint8_t* const (&codes)[N],
            const float (&weights)[N],
            float* out) {
        for (intptr_t pos = 0; pos < DIM; pos += 8) {
            intptr_t count = std::min(intptr_t(8), DIM - pos);
            Vec acc = detail::loongarch_load_output<SL>(out + pos, count);
            for (int j = 0; j < N; ++j)
                acc =
                        fmadd(Vec(weights[j]),
                              block(fine[j], codes[j], pos, count),
                              acc);
            detail::loongarch_store_output<SL>(acc, out + pos, count);
        }
    }

   public:
    static void store(const float* fine, const uint8_t* code, float* out) {
        for (intptr_t pos = 0; pos < DIM; pos += 8) {
            intptr_t count = std::min(intptr_t(8), DIM - pos);
            detail::loongarch_store_output<SL>(
                    block(fine, code, pos, count), out + pos, count);
        }
    }
    static void accum(
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            float* out) {
        const float* fine_tables[] = {fine0};
        const uint8_t* codes[] = {code0};
        const float weights[] = {weight0};
        accumulate(fine_tables, codes, weights, out);
    }
    static void accum(
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            const float* fine1,
            const uint8_t* code1,
            float weight1,
            float* out) {
        const float* fine_tables[] = {fine0, fine1};
        const uint8_t* codes[] = {code0, code1};
        const float weights[] = {weight0, weight1};
        accumulate(fine_tables, codes, weights, out);
    }
    static void accum(
            const float* fine,
            const uint8_t* code0,
            float weight0,
            const uint8_t* code1,
            float weight1,
            float* out) {
        const float* fine_tables[] = {fine, fine};
        const uint8_t* codes[] = {code0, code1};
        const float weights[] = {weight0, weight1};
        accumulate(fine_tables, codes, weights, out);
    }
    static void accum(
            const float* fine0,
            const uint8_t* code0,
            float weight0,
            const float* fine1,
            const uint8_t* code1,
            float weight1,
            const float* fine2,
            const uint8_t* code2,
            float weight2,
            float* out) {
        const float* fine_tables[] = {fine0, fine1, fine2};
        const uint8_t* codes[] = {code0, code1, code2};
        const float weights[] = {weight0, weight1, weight2};
        accumulate(fine_tables, codes, weights, out);
    }
    static void accum(
            const float* fine,
            const uint8_t* code0,
            float weight0,
            const uint8_t* code1,
            float weight1,
            const uint8_t* code2,
            float weight2,
            float* out) {
        const float* fine_tables[] = {fine, fine, fine};
        const uint8_t* codes[] = {code0, code1, code2};
        const float weights[] = {weight0, weight1, weight2};
        accumulate(fine_tables, codes, weights, out);
    }
};
} // namespace faiss::cppcontrib

namespace faiss::cppcontrib::detail {
template <intptr_t DIM, intptr_t FINE_SIZE, intptr_t FINE_BITS, SIMDLevel SL>
struct MinMaxOps<IndexPQDecoder<DIM, FINE_SIZE, FINE_BITS, SL>>
        : LoongArchMinMaxOps<SL> {};
} // namespace faiss::cppcontrib::detail
