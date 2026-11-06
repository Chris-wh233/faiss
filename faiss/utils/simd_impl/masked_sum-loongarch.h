/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <lsxintrin.h>
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif
#include <bit>

#include <faiss/utils/simd_impl/masked_sum.h>

namespace faiss {
namespace {

template <>
inline float masked_sum<SIMDLevel::LOONGARCH_LSX>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    const uint32_t lanes[4] = {1, 2, 4, 8};
    const __m128i bit_masks = __lsx_vld(lanes, 0);
    __m128 acc = (__m128)__lsx_vldi(0);
    size_t j = 0;
    for (; j + 4 <= d; j += 4) {
        const __m128i byte = __lsx_vreplgr2vr_w(bits[j / 8] >> (j % 8));
        const __m128i mask =
                __lsx_vseq_w(__lsx_vand_v(byte, bit_masks), bit_masks);
        const __m128 values = (__m128)__lsx_vand_v(mask, __lsx_vld(x + j, 0));
        acc = __lsx_vfadd_s(acc, values);
    }
    acc = __lsx_vfadd_s(acc, (__m128)__lsx_vshuf4i_w((__m128i)acc, 0x4e));
    acc = __lsx_vfadd_s(acc, (__m128)__lsx_vshuf4i_w((__m128i)acc, 0xb1));
    float sum = std::bit_cast<float>(__lsx_vpickve2gr_w((__m128i)acc, 0));
    for (; j < d; j++) {
        if ((bits[j / 8] >> (j % 8)) & 1) {
            sum += x[j];
        }
    }
    return sum;
}

#if defined(__loongarch_asx)
template <>
inline float masked_sum<SIMDLevel::LOONGARCH_LASX>(
        const float* x,
        const uint8_t* bits,
        size_t d) {
    const uint32_t lanes[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const __m256i bit_masks = __lasx_xvld(lanes, 0);
    __m256 acc = (__m256)__lasx_xvldi(0);
    const size_t full_bytes = d / 8;
    for (size_t j = 0; j < full_bytes; j++) {
        const __m256i byte = __lasx_xvreplgr2vr_w(bits[j]);
        const __m256i mask =
                __lasx_xvseq_w(__lasx_xvand_v(byte, bit_masks), bit_masks);
        const __m256 values =
                (__m256)__lasx_xvand_v(mask, __lasx_xvld(x + j * 8, 0));
        acc = __lasx_xvfadd_s(acc, values);
    }
    acc = __lasx_xvfadd_s(
            acc, (__m256)__lasx_xvpermi_q((__m256i)acc, (__m256i)acc, 0x01));
    acc = __lasx_xvfadd_s(acc, (__m256)__lasx_xvshuf4i_w((__m256i)acc, 0x4e));
    acc = __lasx_xvfadd_s(acc, (__m256)__lasx_xvshuf4i_w((__m256i)acc, 0xb1));
    return std::bit_cast<float>(__lasx_xvpickve2gr_w((__m256i)acc, 0)) +
            masked_sum<SIMDLevel::LOONGARCH_LSX>(
                    x + full_bytes * 8, bits + full_bytes, d % 8);
}
#endif

} // namespace
} // namespace faiss
