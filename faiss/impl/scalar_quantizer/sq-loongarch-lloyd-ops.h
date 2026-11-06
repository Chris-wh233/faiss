/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/scalar_quantizer/sq-loongarch-ops.h>
#include <faiss/utils/simd_impl/loongarch_bits.h>
namespace faiss::scalar_quantizer::detail {
template <SIMDLevel SL>
struct LoongArchLloydOps;
#if defined(__loongarch_sx) && !defined(__loongarch_asx)
template <>
struct LoongArchLloydOps<SIMDLevel::LOONGARCH_LSX> {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LSX>;
    using Indices = simd8uint32_tpl<SIMDLevel::LOONGARCH_LSX>;
    static Vec abs(Vec x) {
        for (int k = 0; k < 2; ++k)
            x.data[k] = __lsx_vand_v(x.data[k], __lsx_vreplgr2vr_w(0x7fffffff));
        return x;
    }
    static Vec round_unsigned(Vec x, int maxval) {
        Vec out;
        for (int k = 0; k < 2; ++k) {
            auto v = __lsx_vfmin_s(
                    __lsx_vfmax_s((__m128)x.data[k], (__m128)__lsx_vldi(0)),
                    (__m128)__lsx_vreplgr2vr_w(
                            std::bit_cast<int32_t>(float(maxval))));
            auto integral = __lsx_vftintrz_w_s(v);
            auto trunc = __lsx_vffint_s_w(integral);
            auto frac = __lsx_vfsub_s(v, trunc);
            auto up = __lsx_vfcmp_cle_s(
                    (__m128)__lsx_vreplgr2vr_w(0x3f000000), frac);
            out.data[k] = (__m128i)__lsx_vffint_s_w(__lsx_vsub_w(integral, up));
        }
        return out;
    }

    static Vec divide(Vec a, Vec b) {
        Vec out;
        for (int k = 0; k < 2; ++k)
            out.data[k] = (__m128i)__lsx_vfdiv_s(
                    (__m128)a.data[k], (__m128)b.data[k]);
        return out;
    }
    static Vec gather(Vec indices, const float* table) {
        float idx[8], values[8];
        indices.storeu(idx);
        for (int j = 0; j < 8; ++j)
            values[j] = table[int(idx[j])];
        return Vec(values);
    }
    // LSX/LASX have no indexed float load.
    static Vec gather(Indices indices, const float* table) {
        uint32_t idx[8];
        float values[8];
        indices.storeu(idx);
        for (int j = 0; j < 8; ++j)
            values[j] = table[idx[j]];
        return Vec(values);
    }
    template <int Bits>
    static Vec select(Vec x, const float* boundaries) {
        Vec indices(0.f);
        for (int step = 1 << (Bits - 1); step; step >>= 1) {
            Vec bound = gather(indices + Vec(float(step - 1)), boundaries);
            Vec next = indices + Vec(float(step));
            for (int k = 0; k < 2; ++k) {
                auto less = __lsx_vfcmp_clt_s(
                        (__m128)x.data[k], (__m128)bound.data[k]);
                indices.data[k] =
                        __lsx_vbitsel_v(next.data[k], indices.data[k], less);
            }
        }
        return indices;
    }
    template <int Bits>
    static Indices unpack(const uint8_t* code) {
        uint64_t word = 0;
        memcpy(&word, code, Bits);
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        Indices out;
        __m128i raw[2] = {
                __lsx_vreplgr2vr_w(uint32_t(word)),
                __lsx_vreplgr2vr_w(uint32_t(word >> (4 * Bits)))};
        for (int k = 0; k < 2; ++k)
            out.data[k] = __lsx_vand_v(
                    __lsx_vsrl_w(raw[k], __lsx_vld(shifts + k * 4, 0)),
                    __lsx_vreplgr2vr_w((1 << Bits) - 1));
        return out;
    }
    template <int Bits>
    static void pack(Vec indices, uint8_t* code) {
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        uint32_t words[2];
        for (int k = 0; k < 2; ++k) {
            auto v = __lsx_vsll_w(
                    __lsx_vftintrz_w_s((__m128)indices.data[k]),
                    __lsx_vld(shifts + k * 4, 0));
            v = __lsx_vor_v(v, __lsx_vshuf4i_w(v, 0x4e));
            v = __lsx_vor_v(v, __lsx_vshuf4i_w(v, 0xb1));
            words[k] = __lsx_vpickve2gr_w(v, 0);
        }
        uint64_t word = uint64_t(words[0]) | (uint64_t(words[1]) << (4 * Bits));
        uint64_t old = 0;
        memcpy(&old, code, Bits);
        word |= old;
        memcpy(code, &word, Bits);
    }
    static Vec plane(uint8_t byte) {
        const uint32_t shifts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        Vec out;
        for (int k = 0; k < 2; ++k)
            out.data[k] = (__m128i)__lsx_vffint_s_w(__lsx_vand_v(
                    __lsx_vsrl_w(
                            __lsx_vreplgr2vr_w(byte),
                            __lsx_vld(shifts + k * 4, 0)),
                    __lsx_vreplgr2vr_w(1)));
        return out;
    }
    static uint8_t positive(Vec x) {
        unsigned bits = 0;
        for (int k = 0; k < 2; ++k) {
            auto mask = __lsx_vmskltz_w(__lsx_vfcmp_clt_s(
                    (__m128)__lsx_vldi(0), (__m128)x.data[k]));
            bits |= unsigned(__lsx_vpickve2gr_w(mask, 0)) << (4 * k);
        }
        return uint8_t(bits);
    }
    template <int Bits>
    static void store_planes(Vec indices, uint8_t* code, size_t stride) {
        uint8_t packed[8] = {};
        pack<8>(indices, packed);
        using B = faiss::detail::LoongArchBits<SIMDLevel::LOONGARCH_LSX>;
        auto v = B::load_partial(packed, 8);
        for (int b = 0; b < Bits; ++b)
            code[b * stride] |= uint8_t(B::bit_plane(v, b));
    }
    template <int Bits>
    static Vec load_planes(const uint8_t* code, size_t stride) {
        Vec out(0.f);
        for (int b = 0; b < Bits; ++b)
            out = out + plane(code[b * stride]) * Vec(float(1 << b));
        return out;
    }
    static Vec butterfly(Vec x, int h) {
        Vec other, out;
        for (int k = 0; k < 2; ++k)
            other.data[k] = h == 1 ? __lsx_vshuf4i_w(x.data[k], 0xb1)
                    : h == 2       ? __lsx_vshuf4i_w(x.data[k], 0x4e)
                                   : x.data[1 - k];
        Vec sum = x + other, diff = other - x;
        uint32_t mask[8];
        for (int j = 0; j < 8; ++j)
            mask[j] = (j & h) ? ~0u : 0u;
        for (int k = 0; k < 2; ++k)
            out.data[k] = __lsx_vbitsel_v(
                    sum.data[k], diff.data[k], __lsx_vld(mask + k * 4, 0));
        return out;
    }
};
#endif
#if defined(__loongarch_asx)
template <>
struct LoongArchLloydOps<SIMDLevel::LOONGARCH_LASX> {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>;
    using Indices = simd8uint32_tpl<SIMDLevel::LOONGARCH_LASX>;
    static Vec abs(Vec x) {
        x.data = __lasx_xvand_v(x.data, __lasx_xvreplgr2vr_w(0x7fffffff));
        return x;
    }
    static Vec round_unsigned(Vec x, int maxval) {
        Vec out;
        auto v = __lasx_xvfmin_s(
                __lasx_xvfmax_s((__m256)x.data, (__m256)__lasx_xvldi(0)),
                (__m256)__lasx_xvreplgr2vr_w(
                        std::bit_cast<int32_t>(float(maxval))));
        auto integral = __lasx_xvftintrz_w_s(v);
        auto trunc = __lasx_xvffint_s_w(integral);
        auto frac = __lasx_xvfsub_s(v, trunc);
        auto up = __lasx_xvfcmp_cle_s(
                (__m256)__lasx_xvreplgr2vr_w(0x3f000000), frac);
        out.data = (__m256i)__lasx_xvffint_s_w(__lasx_xvsub_w(integral, up));
        return out;
    }

    static Vec divide(Vec a, Vec b) {
        Vec out;
        out.data = (__m256i)__lasx_xvfdiv_s((__m256)a.data, (__m256)b.data);
        return out;
    }
    static Vec gather(Vec indices, const float* table) {
        float idx[8], values[8];
        indices.storeu(idx);
        for (int j = 0; j < 8; ++j)
            values[j] = table[int(idx[j])];
        return Vec(values);
    }
    // LSX/LASX have no indexed float load.
    static Vec gather(Indices indices, const float* table) {
        uint32_t idx[8];
        float values[8];
        indices.storeu(idx);
        for (int j = 0; j < 8; ++j)
            values[j] = table[idx[j]];
        return Vec(values);
    }
    template <int Bits>
    static Vec select(Vec x, const float* boundaries) {
        Vec indices(0.f);
        for (int step = 1 << (Bits - 1); step; step >>= 1) {
            Vec bound = gather(indices + Vec(float(step - 1)), boundaries);
            Vec next = indices + Vec(float(step));
            auto less = __lasx_xvfcmp_clt_s((__m256)x.data, (__m256)bound.data);
            indices.data = __lasx_xvbitsel_v(next.data, indices.data, less);
        }
        return indices;
    }
    template <int Bits>
    static Indices unpack(const uint8_t* code) {
        uint64_t word = 0;
        memcpy(&word, code, Bits);
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        Indices out;
        __m256i raw = __lasx_xvpermi_q(
                __lasx_xvreplgr2vr_w(uint32_t(word >> (4 * Bits))),
                __lasx_xvreplgr2vr_w(uint32_t(word)),
                0x20);
        out.data = __lasx_xvand_v(
                __lasx_xvsrl_w(raw, __lasx_xvld(shifts, 0)),
                __lasx_xvreplgr2vr_w((1 << Bits) - 1));
        return out;
    }
    template <int Bits>
    static void pack(Vec indices, uint8_t* code) {
        const uint32_t shifts[8] = {
                0, Bits, 2 * Bits, 3 * Bits, 0, Bits, 2 * Bits, 3 * Bits};
        uint32_t words[2];
        auto v = __lasx_xvsll_w(
                __lasx_xvftintrz_w_s((__m256)indices.data),
                __lasx_xvld(shifts, 0));
        v = __lasx_xvor_v(v, __lasx_xvshuf4i_w(v, 0x4e));
        v = __lasx_xvor_v(v, __lasx_xvshuf4i_w(v, 0xb1));
        words[0] = __lasx_xvpickve2gr_w(v, 0);
        words[1] = __lasx_xvpickve2gr_w(v, 4);
        uint64_t word = uint64_t(words[0]) | (uint64_t(words[1]) << (4 * Bits));
        uint64_t old = 0;
        memcpy(&old, code, Bits);
        word |= old;
        memcpy(code, &word, Bits);
    }
    static Vec plane(uint8_t byte) {
        const uint32_t shifts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        Vec out;
        out.data = (__m256i)__lasx_xvffint_s_w(__lasx_xvand_v(
                __lasx_xvsrl_w(
                        __lasx_xvreplgr2vr_w(byte), __lasx_xvld(shifts, 0)),
                __lasx_xvreplgr2vr_w(1)));
        return out;
    }
    static uint8_t positive(Vec x) {
        unsigned bits = 0;
        auto mask = __lasx_xvmskltz_w(
                __lasx_xvfcmp_clt_s((__m256)__lasx_xvldi(0), (__m256)x.data));
        bits = unsigned(__lasx_xvpickve2gr_w(mask, 0)) |
                (unsigned(__lasx_xvpickve2gr_w(mask, 4)) << 4);
        return uint8_t(bits);
    }
    template <int Bits>
    static void store_planes(Vec indices, uint8_t* code, size_t stride) {
        uint8_t packed[8] = {};
        pack<8>(indices, packed);
        using B = faiss::detail::LoongArchBits<SIMDLevel::LOONGARCH_LASX>;
        auto v = B::load_partial(packed, 8);
        for (int b = 0; b < Bits; ++b)
            code[b * stride] |= uint8_t(B::bit_plane(v, b));
    }
    template <int Bits>
    static Vec load_planes(const uint8_t* code, size_t stride) {
        Vec out(0.f);
        for (int b = 0; b < Bits; ++b)
            out = out + plane(code[b * stride]) * Vec(float(1 << b));
        return out;
    }
    static Vec butterfly(Vec x, int h) {
        Vec other, out;
        other.data = h == 1 ? __lasx_xvshuf4i_w(x.data, 0xb1)
                : h == 2    ? __lasx_xvshuf4i_w(x.data, 0x4e)
                            : __lasx_xvpermi_q(x.data, x.data, 0x01);
        Vec sum = x + other, diff = other - x;
        uint32_t mask[8];
        for (int j = 0; j < 8; ++j)
            mask[j] = (j & h) ? ~0u : 0u;
        out.data = __lasx_xvbitsel_v(sum.data, diff.data, __lasx_xvld(mask, 0));
        return out;
    }
};
#endif
} // namespace faiss::scalar_quantizer::detail
