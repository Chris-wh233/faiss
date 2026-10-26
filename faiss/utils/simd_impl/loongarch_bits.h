/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/utils/simd_levels.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif
namespace faiss::detail {
template <SIMDLevel SL>
struct LoongArchBits;
#if defined(__loongarch_sx)
template <>
struct LoongArchBits<SIMDLevel::LOONGARCH_LSX> {
    using Register = __m128i;
    static constexpr size_t width = 16;
    static Register load(const uint8_t* p) {
        return __lsx_vld(p, 0);
    }
    // Zero-pad without reading past the code.
    static Register load_partial(const uint8_t* p, size_t n) {
        alignas(16) uint8_t bytes[16] = {};
        if (n)
            std::memcpy(bytes, p, n);
        return load(bytes);
    }
    static Register bit_and(Register a, Register b) {
        return __lsx_vand_v(a, b);
    }
    static Register bit_xor(Register a, Register b) {
        return __lsx_vxor_v(a, b);
    }
    static uint64_t popcount(Register v) {
        v = __lsx_vpcnt_d(v);
        v = __lsx_vadd_d(v, __lsx_vbsrl_v(v, 8));
        return static_cast<uint64_t>(__lsx_vpickve2gr_d(v, 0));
    }
    static uint32_t bit_plane(Register value, int plane) {
        const auto bits = __lsx_vmskltz_b(
                __lsx_vsll_b(value, __lsx_vreplgr2vr_b(7 - plane)));
        uint32_t mask = static_cast<uint32_t>(__lsx_vpickve2gr_w(bits, 0));
        return mask;
    }
    static uint64_t different_bytes(Register a, Register b) {
        auto equal = __lsx_vseq_b(a, b);
        // Exactly one bit per unequal byte before the native population count.
        auto different = __lsx_vandi_b(__lsx_vnor_v(equal, equal), 1);
        return popcount(different);
    }
};
#endif
#if defined(__loongarch_asx)
template <>
struct LoongArchBits<SIMDLevel::LOONGARCH_LASX> {
    using Register = __m256i;
    static constexpr size_t width = 32;
    static Register load(const uint8_t* p) {
        return __lasx_xvld(p, 0);
    }
    // Zero-pad without reading past the code.
    static Register load_partial(const uint8_t* p, size_t n) {
        alignas(32) uint8_t bytes[32] = {};
        if (n)
            std::memcpy(bytes, p, n);
        return load(bytes);
    }
    static Register bit_and(Register a, Register b) {
        return __lasx_xvand_v(a, b);
    }
    static Register bit_xor(Register a, Register b) {
        return __lasx_xvxor_v(a, b);
    }
    static uint64_t popcount(Register v) {
        v = __lasx_xvpcnt_d(v);
        v = __lasx_xvadd_d(v, __lasx_xvpermi_q(v, v, 0x01));
        v = __lasx_xvadd_d(v, __lasx_xvbsrl_v(v, 8));
        return static_cast<uint64_t>(__lasx_xvpickve2gr_d(v, 0));
    }
    static uint32_t bit_plane(Register value, int plane) {
        const auto bits = __lasx_xvmskltz_b(
                __lasx_xvsll_b(value, __lasx_xvreplgr2vr_b(7 - plane)));
        uint32_t mask = static_cast<uint32_t>(__lasx_xvpickve2gr_w(bits, 0));
        mask |= static_cast<uint32_t>(__lasx_xvpickve2gr_w(bits, 4)) << 16;
        return mask;
    }
    static uint64_t different_bytes(Register a, Register b) {
        auto equal = __lasx_xvseq_b(a, b);
        // Exactly one bit per unequal byte before the native population count.
        auto different = __lasx_xvandi_b(__lasx_xvnor_v(equal, equal), 1);
        return popcount(different);
    }
};
#endif
} // namespace faiss::detail
