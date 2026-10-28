/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This TU provides non-templated PQ code distance dispatch wrappers
// (pq_code_distance_8bit_single, pq_code_distance_8bit_four) declared
// in pq_code_distance-inl.h. These use with_selected_simd_levels to route to the
// best available SIMD implementation via pq_code_distance_8bit_*_impl
// function template specializations.
//
// The NONE and ARM_NEON _impl specializations are defined inline in
// pq_code_distance-generic.h (included transitively). The AVX2, AVX512,
// and ARM_SVE specializations are in their respective per-SIMD files.

#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>

#define THE_SIMD_LEVEL SIMDLevel::NONE
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
#undef THE_SIMD_LEVEL

namespace faiss {
namespace pq_code_distance {

bool pq_scan(
        size_t M,
        size_t nbits,
        const float* table,
        const float* const* rows,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* distances,
        int64_t* labels,
        bool max_heap) {
    return with_selected_simd_levels<
            (1 << int(SIMDLevel::NONE)) | AVAILABLE_SIMD_LEVELS_LOONGARCH>(
            [&]<SIMDLevel SL>() {
                return pq_scan_impl<SL>(
                        M,
                        nbits,
                        table,
                        rows,
                        codes,
                        ncodes,
                        k,
                        distances,
                        labels,
                        max_heap);
            });
}

void pq_scan_8bit(
        size_t M,
        const float* dis_table,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        bool max_heap) {
    with_selected_simd_levels<
            AVAILABLE_SIMD_LEVELS_BASE_WITH_SVE |
            AVAILABLE_SIMD_LEVELS_LOONGARCH>([&]<SIMDLevel SL>() {
        pq_scan_8bit_impl<SL>(
                M, dis_table, codes, ncodes, k, heap_dis, heap_ids, max_heap);
    });
}

float pq_code_distance_8bit_single(
        size_t M,
        const float* sim_table,
        const uint8_t* code) {
    return with_selected_simd_levels<
            AVAILABLE_SIMD_LEVELS_BASE_WITH_SVE |
            AVAILABLE_SIMD_LEVELS_LOONGARCH>([&]<SIMDLevel SL>() {
        if (use_scalar_8bit_kernel<SL>(M)) {
            return PQCodeDistanceScalar8<SL>::distance_single_code(
                    M, sim_table, code);
        }
        return pq_code_distance_8bit_single_impl<SL>(M, sim_table, code);
    });
}

void pq_code_distance_8bit_four(
        size_t M,
        const float* sim_table,
        const uint8_t* __restrict code0,
        const uint8_t* __restrict code1,
        const uint8_t* __restrict code2,
        const uint8_t* __restrict code3,
        float& result0,
        float& result1,
        float& result2,
        float& result3) {
    with_selected_simd_levels<
            AVAILABLE_SIMD_LEVELS_BASE_WITH_SVE |
            AVAILABLE_SIMD_LEVELS_LOONGARCH>([&]<SIMDLevel SL>() {
        if (use_scalar_8bit_kernel<SL>(M)) {
            PQCodeDistanceScalar8<SL>::distance_four_codes(
                    M,
                    sim_table,
                    code0,
                    code1,
                    code2,
                    code3,
                    result0,
                    result1,
                    result2,
                    result3);
            return;
        }
        pq_code_distance_8bit_four_impl<SL>(
                M,
                sim_table,
                code0,
                code1,
                code2,
                code3,
                result0,
                result1,
                result2,
                result3);
    });
}

} // namespace pq_code_distance
} // namespace faiss
