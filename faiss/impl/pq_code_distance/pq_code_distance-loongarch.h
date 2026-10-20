/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
namespace faiss::pq_code_distance {
namespace detail {
template <class Decoder, SIMDLevel SL>
struct LoongArchPQCodeDistance {
    using PQDecoder = Decoder;
    static constexpr SIMDLevel simd_level = SL;
    using Vec = simd8float32_tpl<SL>;
    // LSX/LASX have no indexed float load. Gather the table entries, then
    // accumulate eight independent subquantizers in native vector registers.
    static Vec gather(Decoder& decoder, const float*& table, size_t ksub) {
        float values[8];
        for (int i = 0; i < 8; ++i) {
            values[i] = table[decoder.decode()];
            table += ksub;
        }
        return Vec(values);
    }
    static float distance_single_code(
            size_t M,
            size_t nbits,
            const float* table,
            const uint8_t* code) {
        Decoder decoder(code, nbits);
        size_t ksub = size_t(1) << nbits, m = 0;
        Vec sum(0.0f);
        for (; m + 8 <= M; m += 8)
            sum = sum + gather(decoder, table, ksub);
        float result = horizontal_add(sum);
        for (; m < M; ++m) {
            result += table[decoder.decode()];
            table += ksub;
        }
        return result;
    }
    static void distance_four_codes(
            size_t M,
            size_t nbits,
            const float* table,
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& r0,
            float& r1,
            float& r2,
            float& r3) {
        Decoder d0(c0, nbits), d1(c1, nbits), d2(c2, nbits), d3(c3, nbits);
        const size_t ksub = size_t(1) << nbits;
        Vec s0(0.0f), s1(0.0f), s2(0.0f), s3(0.0f);
        size_t m = 0;
        for (; m + 8 <= M; m += 8) {
            const float* t0 = table;
            const float* t1 = table;
            const float* t2 = table;
            s0 = s0 + gather(d0, t0, ksub);
            s1 = s1 + gather(d1, t1, ksub);
            s2 = s2 + gather(d2, t2, ksub);
            s3 = s3 + gather(d3, table, ksub);
        }
        r0 = horizontal_add(s0);
        r1 = horizontal_add(s1);
        r2 = horizontal_add(s2);
        r3 = horizontal_add(s3);
        for (; m < M; ++m) {
            r0 += table[d0.decode()];
            r1 += table[d1.decode()];
            r2 += table[d2.decode()];
            r3 += table[d3.decode()];
            table += ksub;
        }
    }
};
} // namespace detail
template <class Decoder>
struct PQCodeDistance<Decoder, THE_SIMD_LEVEL>
        : detail::LoongArchPQCodeDistance<Decoder, THE_SIMD_LEVEL> {};
template <>
float pq_code_distance_8bit_single_impl<THE_SIMD_LEVEL>(
        size_t M,
        const float* table,
        const uint8_t* code) {
    return PQCodeDistance<PQDecoder8, THE_SIMD_LEVEL>::distance_single_code(
            M, 8, table, code);
}
template <>
void pq_code_distance_8bit_four_impl<THE_SIMD_LEVEL>(
        size_t M,
        const float* table,
        const uint8_t* c0,
        const uint8_t* c1,
        const uint8_t* c2,
        const uint8_t* c3,
        float& r0,
        float& r1,
        float& r2,
        float& r3) {
    PQCodeDistance<PQDecoder8, THE_SIMD_LEVEL>::distance_four_codes(
            M, 8, table, c0, c1, c2, c3, r0, r1, r2, r3);
}
} // namespace faiss::pq_code_distance
