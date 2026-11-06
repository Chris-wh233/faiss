/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <faiss/utils/rabitq_simd.h>
#include <bit>
#include <cmath>
namespace faiss::rabitq {
namespace {
#ifdef FAISS_RABITQ_LSX
struct QuantizeVector {
    using Float = __m128;
    using Int = __m128i;
    static constexpr size_t width = 4;
    static Float load(const float* x) {
        return (Float)__lsx_vld(x, 0);
    }
    static Float splat(float x) {
        return (Float)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(x));
    }
    static Float minimum(Float a, Float b) {
        return (Float)__lsx_vbitsel_v((Int)a, (Int)b, __lsx_vfcmp_clt_s(b, a));
    }
    static Float maximum(Float a, Float b) {
        return (Float)__lsx_vbitsel_v((Int)a, (Int)b, __lsx_vfcmp_clt_s(a, b));
    }
    static void store(float* x, Float v) {
        __lsx_vst((Int)v, x, 0);
    }
    static Int codes(const float* x, float lo, float scale, uint8_t limit) {
        auto v = __lsx_vfmul_s(__lsx_vfsub_s(load(x), splat(lo)), splat(scale));
        v = minimum(maximum(v, splat(0)), splat(limit));
        return __lsx_vftintrz_w_s(__lsx_vfadd_s(v, splat(0.5f)));
    }
    static uint32_t sum(Int v) {
        v = __lsx_vadd_w(v, __lsx_vshuf4i_w(v, 0x4e));
        v = __lsx_vadd_w(v, __lsx_vshuf4i_w(v, 0xb1));
        return __lsx_vpickve2gr_w(v, 0);
    }
    static uint32_t square_sum(Int v, uint8_t limit) {
        auto odd = __lsx_vsub_w(__lsx_vslli_w(v, 1), __lsx_vreplgr2vr_w(limit));
        return sum(__lsx_vmul_w(odd, odd));
    }
    static void pack(uint8_t* out, Int v) {
        v = __lsx_vpickev_h(v, v);
        v = __lsx_vpickev_b(v, v);
        uint32_t lo = __lsx_vpickve2gr_w(v, 0);
        memcpy(out, &lo, 4);
    }
};
#endif
#ifdef FAISS_RABITQ_LASX
struct QuantizeVector {
    using Float = __m256;
    using Int = __m256i;
    static constexpr size_t width = 8;
    static Float load(const float* x) {
        return (Float)__lasx_xvld(x, 0);
    }
    static Float splat(float x) {
        return (Float)__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(x));
    }
    static Float minimum(Float a, Float b) {
        return (Float)__lasx_xvbitsel_v(
                (Int)a, (Int)b, __lasx_xvfcmp_clt_s(b, a));
    }
    static Float maximum(Float a, Float b) {
        return (Float)__lasx_xvbitsel_v(
                (Int)a, (Int)b, __lasx_xvfcmp_clt_s(a, b));
    }
    static void store(float* x, Float v) {
        __lasx_xvst((Int)v, x, 0);
    }
    static Int codes(const float* x, float lo, float scale, uint8_t limit) {
        auto v = __lasx_xvfmul_s(
                __lasx_xvfsub_s(load(x), splat(lo)), splat(scale));
        v = minimum(maximum(v, splat(0)), splat(limit));
        return __lasx_xvftintrz_w_s(__lasx_xvfadd_s(v, splat(0.5f)));
    }
    static uint32_t sum(Int v) {
        v = __lasx_xvadd_w(v, __lasx_xvpermi_q(v, v, 0x01));
        v = __lasx_xvadd_w(v, __lasx_xvshuf4i_w(v, 0x4e));
        v = __lasx_xvadd_w(v, __lasx_xvshuf4i_w(v, 0xb1));
        return __lasx_xvpickve2gr_w(v, 0);
    }
    static uint32_t square_sum(Int v, uint8_t limit) {
        auto odd = __lasx_xvsub_w(
                __lasx_xvslli_w(v, 1), __lasx_xvreplgr2vr_w(limit));
        return sum(__lasx_xvmul_w(odd, odd));
    }
    static void pack(uint8_t* out, Int v) {
        v = __lasx_xvpickev_h(v, v);
        v = __lasx_xvpickev_b(v, v);
        uint32_t lo = __lasx_xvpickve2gr_w(v, 0);
        memcpy(out, &lo, 4);
        uint32_t hi = __lasx_xvpickve2gr_w(v, 4);
        memcpy(out + 4, &hi, 4);
    }
};
#endif
} // namespace

template <>
void minmax_values<THE_SIMD_LEVEL>(
        const float* values,
        size_t n,
        float& mn,
        float& mx) {
    if (n == 0)
        return;
    mn = mx = values[0];
    if (std::isnan(mn))
        return;
    using V = QuantizeVector;
    auto lo = V::splat(mn), hi = lo;
    size_t i = 0;
    for (; i + V::width <= n; i += V::width) {
        auto v = V::load(values + i);
        lo = V::minimum(lo, v);
        hi = V::maximum(hi, v);
    }
    float lows[V::width], highs[V::width];
    V::store(lows, lo);
    V::store(highs, hi);
    for (size_t j = 0; j < V::width; ++j) {
        mn = std::min(mn, lows[j]);
        mx = std::max(mx, highs[j]);
    }
    for (; i < n; ++i) {
        mn = std::min(mn, values[i]);
        mx = std::max(mx, values[i]);
    }
}
template <>
void lut_minmax_16<THE_SIMD_LEVEL>(const float* tab, float& mn, float& mx) {
    minmax_values<THE_SIMD_LEVEL>(tab, 16, mn, mx);
}
template <>
void lut_quantize_16_to_uint8<THE_SIMD_LEVEL>(
        const float* tab,
        float mn,
        float a,
        uint8_t* out) {
    for (size_t i = 0; i < 16; i += QuantizeVector::width)
        QuantizeVector::pack(
                out + i, QuantizeVector::codes(tab + i, mn, a, 255));
}
template <>
void quantize_query_values<THE_SIMD_LEVEL>(
        const float* rq,
        size_t d,
        float v_min,
        float inv_delta,
        uint8_t max_code,
        bool centered,
        uint8_t* rqq,
        size_t& sum_qq,
        int64_t& sum2_signed_odd_int) {
    size_t i = 0;
    for (; i + QuantizeVector::width <= d; i += QuantizeVector::width) {
        auto v = QuantizeVector::codes(rq + i, v_min, inv_delta, max_code);
        QuantizeVector::pack(rqq + i, v);
        sum_qq += QuantizeVector::sum(v);
        if (centered)
            sum2_signed_odd_int += QuantizeVector::square_sum(v, max_code);
    }
    for (; i < d; ++i) {
        auto v = round_clamped_byte_scalar(
                (rq[i] - v_min) * inv_delta, max_code);
        rqq[i] = v;
        sum_qq += v;
        if (centered) {
            int64_t odd = int64_t(v) * 2 - max_code;
            sum2_signed_odd_int += odd * odd;
        }
    }
}
} // namespace faiss::rabitq
