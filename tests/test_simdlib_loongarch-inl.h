/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <source_location>

namespace {
template <typename A, typename B>
void expect_bits(
        const A& a,
        const B& b,
        std::source_location loc = std::source_location::current()) {
    SCOPED_TRACE(loc.line());
    std::array<uint8_t, 32> actual, expected;
    a.storeu(actual.data());
    b.storeu(expected.data());
    EXPECT_EQ(actual, expected);
}

template <faiss::SIMDLevel SL>
void check_simdlib() {
    using namespace faiss;
    constexpr auto NONE = SIMDLevel::NONE;
    using U16 = simd16uint16_tpl<SL>;
    using U8 = simd32uint8_tpl<SL>;
    using U32 = simd8uint32_tpl<SL>;
    using F32 = simd8float32_tpl<SL>;
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 100; ++trial) {
        // Offset loads exercise unaligned access, including full unsigned
        // ranges.
        std::array<uint16_t, 17> aa, bb;
        std::array<uint8_t, 33> cc, dd;
        std::array<uint32_t, 9> ee, ff;
        std::array<float, 9> gg, hh;
        for (auto& x : aa)
            x = rng();
        for (auto& x : bb)
            x = rng();
        for (auto& x : cc)
            x = rng();
        for (auto& x : dd)
            x = rng();
        for (auto& x : ee)
            x = rng();
        for (auto& x : ff)
            x = rng();
        for (auto& x : gg)
            x = int(rng() % 1000) - 500;
        for (auto& x : hh)
            x = int(rng() % 1000) - 500;
        U16 a(aa.data() + 1), b(bb.data() + 1), t(uint16_t(32768));
        simd16uint16_tpl<NONE> ar(aa.data() + 1), br(bb.data() + 1),
                tr(uint16_t(32768));
        expect_bits(a, ar);
        expect_bits(a + b, ar + br);
        expect_bits(a - b, ar - br);
        expect_bits(a * b, ar * br);
        expect_bits(a & b, ar & br);
        expect_bits(a | b, ar | br);
        expect_bits(a ^ b, ar ^ br);
        expect_bits(~a, ~ar);
        expect_bits(a == b, ar == br);
        EXPECT_EQ(a.ge_mask(t), ar.ge_mask(tr));
        EXPECT_EQ(a.le_mask(t), ar.le_mask(tr));
        EXPECT_EQ(a.gt_mask(t), ar.gt_mask(tr));
        EXPECT_EQ(a.all_gt(t), ar.all_gt(tr));
        EXPECT_EQ(cmp_ge32(a, b, t), cmp_ge32(ar, br, tr));
        EXPECT_EQ(cmp_le32(a, b, t), cmp_le32(ar, br, tr));
        for (int shift = 0; shift < 16; ++shift) {
            expect_bits(a >> shift, ar >> shift);
            expect_bits(a << shift, ar << shift);
        }
        expect_bits(a << 16, simd16uint16_tpl<NONE>(0));
        expect_bits(a >> 16, simd16uint16_tpl<NONE>(0));
        expect_bits(min(a, b), min(ar, br));
        expect_bits(max(a, b), max(ar, br));
        expect_bits(hadd(a, b), hadd(ar, br));
        expect_bits(combine2x2(a, b), combine2x2(ar, br));
        auto amin = a, amax = a;
        amin.accu_min(b);
        amax.accu_max(b);
        expect_bits(amin, min(ar, br));
        expect_bits(amax, max(ar, br));
        EXPECT_EQ(a.get_scalar_0(), ar.get_scalar_0());
        U8 c(cc.data() + 1), e(dd.data() + 1);
        simd32uint8_tpl<NONE> cr(cc.data() + 1), er(dd.data() + 1);
        expect_bits(c + e, cr + er);
        expect_bits(c & e, cr & er);
        expect_bits(c.lookup_2_lanes(e), cr.lookup_2_lanes(er));
        expect_bits(blendv(c, e, c), blendv(cr, er, cr));
        EXPECT_EQ(get_MSBs(c), get_MSBs(cr));
        expect_bits(
                uint16_to_uint8_saturate(a, b),
                uint16_to_uint8_saturate(ar, br));
        for (int half = 0; half < 2; ++half) {
            auto wide = half == 0 ? c.lane0_as_uint16() : c.lane1_as_uint16();
            uint16_t values[16];
            wide.storeu(values);
            for (int j = 0; j < 16; ++j)
                EXPECT_EQ(values[j], cc[1 + half * 16 + j]);
        }
        U32 u(ee.data() + 1), v(ff.data() + 1);
        simd8uint32_tpl<NONE> ur(ee.data() + 1), vr(ff.data() + 1);
        expect_bits(u + v, ur + vr);
        expect_bits(u - v, ur - vr);
        expect_bits(u.unzip(), ur.unzip());
        EXPECT_TRUE(u == u);
        EXPECT_FALSE(u != u);
        EXPECT_EQ(u == v, ur == vr);
        F32 x(gg.data() + 1), y(hh.data() + 1);
        simd8float32_tpl<NONE> xr(gg.data() + 1), yr(hh.data() + 1);
        expect_bits(x + y, xr + yr);
        expect_bits(x - y, xr - yr);
        expect_bits(x * y, xr * yr);
        expect_bits(fmadd(x, y, x), fmadd(xr, yr, xr));
        expect_bits(hadd(x, y), hadd(xr, yr));
        expect_bits(unpacklo(x, y), unpacklo(xr, yr));
        expect_bits(unpackhi(x, y), unpackhi(xr, yr));
        expect_bits(geteven(x, y), geteven(xr, yr));
        expect_bits(getodd(x, y), getodd(xr, yr));
        expect_bits(getlow128(x, y), getlow128(xr, yr));
        expect_bits(gethigh128(x, y), gethigh128(xr, yr));
        float sum = 0;
        for (int j = 1; j < 9; ++j)
            sum += gg[j];
        EXPECT_EQ(horizontal_add(x), sum);
        EXPECT_TRUE(x == x);
        EXPECT_FALSE(x != x);
        EXPECT_EQ(x == y, xr == yr);
        auto compare =
                []<typename V, typename I, typename R, typename RI>(
                        V ca, I ci, V cu, I ui, R ra, RI ri, R ru, RI rui) {
                    V lo, hi;
                    I li, hi_i;
                    R rlo, rhi;
                    RI rli, rhi_i;
                    cmplt_min_max_fast(ca, ci, cu, ui, lo, li, hi, hi_i);
                    cmplt_min_max_fast(ra, ri, ru, rui, rlo, rli, rhi, rhi_i);
                    expect_bits(lo, rlo);
                    expect_bits(li, rli);
                    expect_bits(hi, rhi);
                    expect_bits(hi_i, rhi_i);
                };
        compare(a, b, b, a, ar, br, br, ar);
        compare(a, b, a, a, ar, br, ar, ar);
        compare(u, v, v, u, ur, vr, vr, ur);
        compare(u, v, u, u, ur, vr, ur, ur);
        compare(x, u, y, v, xr, ur, yr, vr);
        compare(x, u, x, v, xr, ur, xr, vr);
        cmplt_and_blend_inplace(x, u, y, v);
        cmplt_and_blend_inplace(xr, ur, yr, vr);
        expect_bits(y, yr);
        expect_bits(v, vr);
        a += b;
        ar += br;
        expect_bits(a, ar);
        a -= b;
        ar -= br;
        expect_bits(a, ar);
        c += e;
        cr += er;
        expect_bits(c, cr);
        u += v;
        ur += vr;
        expect_bits(u, ur);
        x += y;
        xr += yr;
        expect_bits(x, xr);
        EXPECT_EQ(a.bin(), ar.bin());
        EXPECT_EQ(a.hex(), ar.hex());
        EXPECT_EQ(a.dec(), ar.dec());
        uint8_t stored[33];
        c.storeu(stored + 1);
        U8 reloaded(stored + 1);
        EXPECT_TRUE(c.is_same_as(reloaded));
        c.clear();
        EXPECT_EQ(get_MSBs(c), 0u);
        expect_bits(c, simd32uint8_tpl<NONE>(0));
    }
    const uint16_t edges[16] = {
            0,
            1,
            127,
            128,
            254,
            255,
            256,
            257,
            32767,
            32768,
            65534,
            65535,
            42,
            0,
            255,
            256};
    expect_bits(
            uint16_to_uint8_saturate(U16(edges), U16(uint16_t(255))),
            uint16_to_uint8_saturate(
                    simd16uint16_tpl<NONE>(edges),
                    simd16uint16_tpl<NONE>(uint16_t(255))));
    F32 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(nan == nan);
    EXPECT_TRUE(F32(0.0f) == F32(-0.0f));
    // Exhaust the table-lookup control byte (only bit 7 and bits 0..3 matter).
    uint8_t table[32];
    for (int j = 0; j < 32; ++j)
        table[j] = j + 1;
    for (int j = 0; j < 256; ++j) {
        expect_bits(
                U8(table).lookup_2_lanes(U8(j)),
                simd32uint8_tpl<NONE>(table).lookup_2_lanes(
                        simd32uint8_tpl<NONE>(j)));
    }
}
} // namespace
