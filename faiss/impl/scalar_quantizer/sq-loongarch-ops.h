/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <bit>
#include <cstring>
namespace faiss::scalar_quantizer {
namespace detail {
inline uint32_t load_sq_word(const uint8_t* code, size_t n) {
    uint32_t word = 0;
    std::memcpy(&word, code, n);
    return word;
}
inline void store_sq_word(uint8_t* code, uint32_t word, size_t n, bool merge) {
    if (merge)
        word |= load_sq_word(code, n);
    std::memcpy(code, &word, n);
}
template <SIMDLevel SL>
struct LoongArchSQOps;
#if defined(__loongarch_sx) && !defined(__loongarch_asx)
template <>
struct LoongArchSQOps<SIMDLevel::LOONGARCH_LSX> {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>;
    static Vec normalize(Vec x, Vec lo, Vec range) {
        Vec out;
        for (int i = 0; i < 2; ++i) {
            const auto zero = __lsx_vldi(0);
            const auto one = __lsx_vreplgr2vr_w(0x3f800000);
            const auto missing =
                    __lsx_vfcmp_ceq_s((__m128)range.data[i], (__m128)zero);
            const auto divisor = __lsx_vbitsel_v(range.data[i], one, missing);
            auto value = __lsx_vfdiv_s(
                    __lsx_vfsub_s((__m128)x.data[i], (__m128)lo.data[i]),
                    (__m128)divisor);
            value = __lsx_vfmin_s(
                    __lsx_vfmax_s(value, (__m128)zero), (__m128)one);
            out.data[i] = __lsx_vbitsel_v((__m128i)value, zero, missing);
        }
        return out;
    }
    template <int Bits, bool normalized = true>
    static Vec decode_codes(const uint8_t* code) {
        Vec out;
        constexpr int bytes = Bits / 2;
        const uint32_t lo = load_sq_word(code, bytes),
                       hi = load_sq_word(code + bytes, bytes);
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};

        __m128i words[2] = {__lsx_vreplgr2vr_w(lo), __lsx_vreplgr2vr_w(hi)};
        for (int i = 0; i < 2; ++i) {
            const auto indices = __lsx_vand_v(
                    __lsx_vsrl_w(words[i], __lsx_vld(shifts + i * 4, 0)),
                    __lsx_vreplgr2vr_w((1 << Bits) - 1));
            auto value = __lsx_vffint_s_w(indices);
            if constexpr (normalized)
                value = __lsx_vfdiv_s(
                        __lsx_vfadd_s(
                                value, (__m128)__lsx_vreplgr2vr_w(0x3f000000)),
                        (__m128)__lsx_vreplgr2vr_w(
                                std::bit_cast<int32_t>(
                                        float((1 << Bits) - 1))));
            out.data[i] = (__m128i)value;
        }
        return out;
    }
    template <int Bits, bool normalized = true>
    static void encode_codes(Vec value, uint8_t* code) {
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        for (int i = 0; i < 2; ++i) {
            __m128i indices;
            if constexpr (normalized && Bits != 8) {
                // Codec4bit/Codec6bit multiply in double precision before
                // truncation; retain their behavior at quantization boundaries.
                const auto scale = (__m128d)__lsx_vreplgr2vr_d(
                        std::bit_cast<int64_t>(double((1 << Bits) - 1)));
                auto lo = __lsx_vftintrz_l_d(__lsx_vfmul_d(
                        __lsx_vfcvtl_d_s((__m128)value.data[i]), scale));
                auto hi = __lsx_vftintrz_l_d(__lsx_vfmul_d(
                        __lsx_vfcvth_d_s((__m128)value.data[i]), scale));
                indices = __lsx_vpickev_w(hi, lo);
            } else {
                auto v = (__m128)value.data[i];
                if constexpr (normalized)
                    v = __lsx_vfmul_s(
                            v,
                            (__m128)__lsx_vreplgr2vr_w(
                                    std::bit_cast<int32_t>(255.0f)));
                indices = __lsx_vftintrz_w_s(v);
            }
            auto packed = __lsx_vsll_w(
                    __lsx_vand_v(indices, __lsx_vreplgr2vr_w((1 << Bits) - 1)),
                    __lsx_vld(shifts + i * 4, 0));
            packed = __lsx_vor_v(packed, __lsx_vshuf4i_w(packed, 0x4e));
            packed = __lsx_vor_v(packed, __lsx_vshuf4i_w(packed, 0xb1));

            store_sq_word(
                    code + i * (Bits / 2),
                    __lsx_vpickve2gr_w(packed, 0),
                    Bits / 2,
                    normalized && Bits != 8);
        }
    }
    template <bool bf16>
    static Vec decode_half(const uint8_t* code) {
        Vec out;

        const auto raw = __lsx_vld(code, 0);
        if constexpr (bf16) {
            out.data[0] = __lsx_vslli_w(__lsx_vilvl_h(__lsx_vldi(0), raw), 16);
            out.data[1] = __lsx_vslli_w(__lsx_vilvh_h(__lsx_vldi(0), raw), 16);
        } else {
            out.data[0] = (__m128i)__lsx_vfcvtl_s_h(raw);
            out.data[1] = (__m128i)__lsx_vfcvth_s_h(raw);
        }
        return out;
    }
    template <bool bf16>
    static void encode_half(Vec value, uint8_t* code) {
        for (int i = 0; i < 2; ++i) {
            if constexpr (bf16) {
                // Faiss BF16 uses half-up rounding, not ties-to-even.
                value.data[i] = __lsx_vsrli_w(
                        __lsx_vadd_w(value.data[i], __lsx_vreplgr2vr_w(0x8000)),
                        16);
            } else {
                // Match fp16-inl.h, including its rounding at half-way
                // values and subnormals. Hardware FCVT uses ties-to-even.
                const auto abs = __lsx_vand_v(
                        value.data[i], __lsx_vreplgr2vr_w(0x7fffffff));
                const auto inf = __lsx_vreplgr2vr_w(0x7f800000);
                const auto nan = __lsx_vslt_wu(inf, abs);
                const auto special = __lsx_vbitsel_v(
                        __lsx_vreplgr2vr_w(0x7c00),
                        __lsx_vreplgr2vr_w(0x7e00),
                        nan);
                auto scaled = __lsx_vfmul_s(
                        (__m128)__lsx_vand_v(
                                abs, __lsx_vreplgr2vr_w(0xfffff000u)),
                        (__m128)__lsx_vreplgr2vr_w(15u << 23));
                scaled = __lsx_vfmin_s(
                        scaled,
                        (__m128)__lsx_vreplgr2vr_w((31u << 23) - 0x1000u));
                const auto rounded = __lsx_vsrli_w(
                        __lsx_vadd_w(
                                (__m128i)scaled, __lsx_vreplgr2vr_w(0x1000)),
                        13);
                const auto sign = __lsx_vsrli_w(
                        __lsx_vand_v(
                                value.data[i], __lsx_vreplgr2vr_w(0x80000000u)),
                        16);
                value.data[i] = __lsx_vor_v(
                        sign,
                        __lsx_vbitsel_v(
                                special, rounded, __lsx_vslt_wu(abs, inf)));
            }
        }

        const auto packed = __lsx_vpickev_h(value.data[1], value.data[0]);
        __lsx_vst(packed, code, 0);
    }
};
#endif
#if defined(__loongarch_asx)
template <>
struct LoongArchSQOps<SIMDLevel::LOONGARCH_LASX> {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>;
    static Vec normalize(Vec x, Vec lo, Vec range) {
        Vec out;
        for (int i = 0; i < 1; ++i) {
            const auto zero = __lasx_xvldi(0);
            const auto one = __lasx_xvreplgr2vr_w(0x3f800000);
            const auto missing =
                    __lasx_xvfcmp_ceq_s((__m256)range.data[i], (__m256)zero);
            const auto divisor = __lasx_xvbitsel_v(range.data[i], one, missing);
            auto value = __lasx_xvfdiv_s(
                    __lasx_xvfsub_s((__m256)x.data[i], (__m256)lo.data[i]),
                    (__m256)divisor);
            value = __lasx_xvfmin_s(
                    __lasx_xvfmax_s(value, (__m256)zero), (__m256)one);
            out.data[i] = __lasx_xvbitsel_v((__m256i)value, zero, missing);
        }
        return out;
    }
    template <int Bits, bool normalized = true>
    static Vec decode_codes(const uint8_t* code) {
        Vec out;
        constexpr int bytes = Bits / 2;
        const uint32_t lo = load_sq_word(code, bytes),
                       hi = load_sq_word(code + bytes, bytes);
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};

        __m256i words[1] = {__lasx_xvpermi_q(
                __lasx_xvreplgr2vr_w(hi), __lasx_xvreplgr2vr_w(lo), 0x20)};
        for (int i = 0; i < 1; ++i) {
            const auto indices = __lasx_xvand_v(
                    __lasx_xvsrl_w(words[i], __lasx_xvld(shifts + i * 8, 0)),
                    __lasx_xvreplgr2vr_w((1 << Bits) - 1));
            auto value = __lasx_xvffint_s_w(indices);
            if constexpr (normalized)
                value = __lasx_xvfdiv_s(
                        __lasx_xvfadd_s(
                                value,
                                (__m256)__lasx_xvreplgr2vr_w(0x3f000000)),
                        (__m256)__lasx_xvreplgr2vr_w(
                                std::bit_cast<int32_t>(
                                        float((1 << Bits) - 1))));
            out.data[i] = (__m256i)value;
        }
        return out;
    }
    template <int Bits, bool normalized = true>
    static void encode_codes(Vec value, uint8_t* code) {
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        for (int i = 0; i < 1; ++i) {
            __m256i indices;
            if constexpr (normalized && Bits != 8) {
                // Codec4bit/Codec6bit multiply in double precision before
                // truncation; retain their behavior at quantization boundaries.
                const auto scale = (__m256d)__lasx_xvreplgr2vr_d(
                        std::bit_cast<int64_t>(double((1 << Bits) - 1)));
                auto lo = __lasx_xvftintrz_l_d(__lasx_xvfmul_d(
                        __lasx_xvfcvtl_d_s((__m256)value.data[i]), scale));
                auto hi = __lasx_xvftintrz_l_d(__lasx_xvfmul_d(
                        __lasx_xvfcvth_d_s((__m256)value.data[i]), scale));
                indices = __lasx_xvpickev_w(hi, lo);
            } else {
                auto v = (__m256)value.data[i];
                if constexpr (normalized)
                    v = __lasx_xvfmul_s(
                            v,
                            (__m256)__lasx_xvreplgr2vr_w(
                                    std::bit_cast<int32_t>(255.0f)));
                indices = __lasx_xvftintrz_w_s(v);
            }
            auto packed = __lasx_xvsll_w(
                    __lasx_xvand_v(
                            indices, __lasx_xvreplgr2vr_w((1 << Bits) - 1)),
                    __lasx_xvld(shifts + i * 8, 0));
            packed = __lasx_xvor_v(packed, __lasx_xvshuf4i_w(packed, 0x4e));
            packed = __lasx_xvor_v(packed, __lasx_xvshuf4i_w(packed, 0xb1));

            store_sq_word(
                    code,
                    __lasx_xvpickve2gr_w(packed, 0),
                    Bits / 2,
                    normalized && Bits != 8);
            store_sq_word(
                    code + Bits / 2,
                    __lasx_xvpickve2gr_w(packed, 4),
                    Bits / 2,
                    normalized && Bits != 8);
        }
    }
    template <bool bf16>
    static Vec decode_half(const uint8_t* code) {
        Vec out;

        const auto raw = __lasx_xvpermi_q(
                __lasx_xvldrepl_d(code, 8), __lasx_xvldrepl_d(code, 0), 0x20);
        if constexpr (bf16)
            out.data[0] =
                    __lasx_xvslli_w(__lasx_xvilvl_h(__lasx_xvldi(0), raw), 16);
        else
            out.data[0] = (__m256i)__lasx_xvfcvtl_s_h(raw);
        return out;
    }
    template <bool bf16>
    static void encode_half(Vec value, uint8_t* code) {
        for (int i = 0; i < 1; ++i) {
            if constexpr (bf16) {
                // Faiss BF16 uses half-up rounding, not ties-to-even.
                value.data[i] = __lasx_xvsrli_w(
                        __lasx_xvadd_w(
                                value.data[i], __lasx_xvreplgr2vr_w(0x8000)),
                        16);
            } else {
                // Match fp16-inl.h, including its rounding at half-way
                // values and subnormals. Hardware FCVT uses ties-to-even.
                const auto abs = __lasx_xvand_v(
                        value.data[i], __lasx_xvreplgr2vr_w(0x7fffffff));
                const auto inf = __lasx_xvreplgr2vr_w(0x7f800000);
                const auto nan = __lasx_xvslt_wu(inf, abs);
                const auto special = __lasx_xvbitsel_v(
                        __lasx_xvreplgr2vr_w(0x7c00),
                        __lasx_xvreplgr2vr_w(0x7e00),
                        nan);
                auto scaled = __lasx_xvfmul_s(
                        (__m256)__lasx_xvand_v(
                                abs, __lasx_xvreplgr2vr_w(0xfffff000u)),
                        (__m256)__lasx_xvreplgr2vr_w(15u << 23));
                scaled = __lasx_xvfmin_s(
                        scaled,
                        (__m256)__lasx_xvreplgr2vr_w((31u << 23) - 0x1000u));
                const auto rounded = __lasx_xvsrli_w(
                        __lasx_xvadd_w(
                                (__m256i)scaled, __lasx_xvreplgr2vr_w(0x1000)),
                        13);
                const auto sign = __lasx_xvsrli_w(
                        __lasx_xvand_v(
                                value.data[i],
                                __lasx_xvreplgr2vr_w(0x80000000u)),
                        16);
                value.data[i] = __lasx_xvor_v(
                        sign,
                        __lasx_xvbitsel_v(
                                special, rounded, __lasx_xvslt_wu(abs, inf)));
            }
        }

        const auto packed = __lasx_xvpickev_h(value.data[0], value.data[0]);
        __lasx_xvstelm_d(packed, code, 0, 0);
        __lasx_xvstelm_d(packed, code, 8, 2);
    }
};
#endif
} // namespace detail
} // namespace faiss::scalar_quantizer
