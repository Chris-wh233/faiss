/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>
#ifdef COMPILE_SIMD_LOONGARCH_LSX
#include <faiss/impl/simdlib/simdlib_lsx.h>
#include <lsxintrin.h>
#include <bit>
#include <cmath>
namespace faiss {
namespace {
struct FloatVec {
    struct Mask {
        __m128i data;
        Mask operator&(Mask other) const {
            return {__lsx_vand_v(data, other.data)};
        }
        Mask operator|(Mask other) const {
            return {__lsx_vor_v(data, other.data)};
        }
        Mask operator~() const {
            return {__lsx_vnor_v(data, data)};
        }
        uint32_t bits() const {
            return detail::simdlib_lsx::byte_mask(data);
        }
        size_t count() const {
            return std::popcount(bits()) / 4;
        }
        bool any() const {
            return bits() != 0;
        }
    };
    static constexpr size_t width = 4;
    __m128 data;
    explicit FloatVec(__m128 value) : data(value) {}
    explicit FloatVec(float value)
            : data((__m128)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(value))) {}
    explicit FloatVec(const float* ptr) : data((__m128)__lsx_vld(ptr, 0)) {}
    void store(float* ptr) const {
        __lsx_vst((__m128i)data, ptr, 0);
    }

    FloatVec operator+(FloatVec other) const {
        return FloatVec(__lsx_vfadd_s(data, other.data));
    }
    FloatVec operator-(FloatVec other) const {
        return FloatVec(__lsx_vfsub_s(data, other.data));
    }
    FloatVec operator*(FloatVec other) const {
        return FloatVec(__lsx_vfmul_s(data, other.data));
    }
    FloatVec operator/(FloatVec other) const {
        return FloatVec(__lsx_vfdiv_s(data, other.data));
    }
    FloatVec minimum_number(FloatVec other) const {
        return FloatVec(__lsx_vfmin_s(data, other.data));
    }
    FloatVec maximum_number(FloatVec other) const {
        return FloatVec(__lsx_vfmax_s(data, other.data));
    }
    FloatVec canberra_terms(FloatVec other) const {
        // The scalar metric uses double fabs/add/div for the denominator.
        // In particular, |x| + |y| must not overflow in float.
        const auto a = abs().data, b = other.abs().data;
        const auto delta = (*this - other).abs().data;
        const auto lo = __lsx_vfdiv_d(
                __lsx_vfcvtl_d_s(delta),
                __lsx_vfadd_d(__lsx_vfcvtl_d_s(a), __lsx_vfcvtl_d_s(b)));
        const auto hi = __lsx_vfdiv_d(
                __lsx_vfcvth_d_s(delta),
                __lsx_vfadd_d(__lsx_vfcvth_d_s(a), __lsx_vfcvth_d_s(b)));
        return FloatVec(__lsx_vfcvt_s_d(hi, lo));
    }
    Mask equal(FloatVec other) const {
        return {__lsx_vfcmp_ceq_s(data, other.data)};
    }
    Mask less(FloatVec other) const {
        return {__lsx_vfcmp_clt_s(data, other.data)};
    }
    Mask less_equal(FloatVec other) const {
        return {__lsx_vfcmp_cle_s(data, other.data)};
    }
    FloatVec masked(Mask mask) const {
        return FloatVec((__m128)__lsx_vand_v((__m128i)data, mask.data));
    }
    static FloatVec select(Mask mask, FloatVec yes, FloatVec no) {
        return FloatVec((__m128)__lsx_vbitsel_v(
                (__m128i)no.data, (__m128i)yes.data, mask.data));
    }
    FloatVec negative_log_product(FloatVec args) const {
        // Preserve the scalar metric's double log and multiplication. libm
        // supplies log per lane; conversion and multiplication remain SIMD.
        float values[width];
        args.store(values);
        double logs_lo[width / 2], logs_hi[width / 2];
        for (size_t j = 0; j < width / 2; ++j) {
            const size_t lane = (j / 2) * 4 + j % 2;
            logs_lo[j] = std::log(double(values[lane]));
            logs_hi[j] = std::log(double(values[lane + 2]));
        }
        const auto negative = (__m128)__lsx_vxor_v(
                (__m128i)data, __lsx_vreplgr2vr_w(0x80000000u));
        const auto lo = __lsx_vfmul_d(
                __lsx_vfcvtl_d_s(negative), (__m128d)__lsx_vld(logs_lo, 0));
        const auto hi = __lsx_vfmul_d(
                __lsx_vfcvth_d_s(negative), (__m128d)__lsx_vld(logs_hi, 0));
        return FloatVec(__lsx_vfcvt_s_d(hi, lo));
    }
    static FloatVec madd(FloatVec a, FloatVec b, FloatVec c) {
        return FloatVec(__lsx_vfmadd_s(a.data, b.data, c.data));
    }
    FloatVec abs() const {
        return FloatVec((__m128)__lsx_vand_v(
                (__m128i)data, __lsx_vreplgr2vr_w(0x7fffffff)));
    }
    FloatVec min(FloatVec other) const {
        const auto mask = __lsx_vfcmp_clt_s(other.data, data);
        return FloatVec((__m128)__lsx_vbitsel_v(
                (__m128i)data, (__m128i)other.data, mask));
    }
    FloatVec max(FloatVec other) const {
        const auto mask = __lsx_vfcmp_clt_s(data, other.data);
        return FloatVec((__m128)__lsx_vbitsel_v(
                (__m128i)data, (__m128i)other.data, mask));
    }
    float scalar() const {
        return std::bit_cast<float>(
                static_cast<uint32_t>(__lsx_vpickve2gr_w((__m128i)data, 0)));
    }
    uint32_t equal_mask(FloatVec other) const {
        return detail::simdlib_lsx::byte_mask(
                __lsx_vfcmp_ceq_s(data, other.data));
    }

    float reduce_sum() const {
        FloatVec v = *this;
        v = v + FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0x4e));
        v = v + FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0xb1));
        return v.scalar();
    }
    float reduce_min() const {
        FloatVec v = *this;
        v = v.min(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0x4e)));
        v = v.min(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0xb1)));
        return v.scalar();
    }
    float reduce_max() const {
        FloatVec v = *this;
        v = v.max(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0x4e)));
        v = v.max(FloatVec((__m128)__lsx_vshuf4i_w((__m128i)v.data, 0xb1)));
        return v.scalar();
    }
};
} // namespace
} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LSX
#include <faiss/utils/simd_impl/distances_loongarch-inl.h>
#include <faiss/utils/simd_impl/extra_distances_loongarch-inl.h>

namespace faiss::quantizer_preprocessing {
template <>
void eden_code_stats<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        const float* centroid,
        const float* decoded,
        double& norm,
        double& ip) {
    __m128d n0 = (__m128d)__lsx_vreplgr2vr_w(0), n1 = n0, p0 = n0, p1 = n0;
    size_t j = 0;
    for (; j + 4 <= d; j += 4) {
        const FloatVec q(decoded + j);
        const FloatVec c = centroid ? FloatVec(centroid + j) : FloatVec(0.0f);
        const FloatVec residual = FloatVec(x + j) - c;
        const auto q0 = __lsx_vfcvtl_d_s(q.data), q1 = __lsx_vfcvth_d_s(q.data);
        const auto r0 = __lsx_vfcvtl_d_s(residual.data),
                   r1 = __lsx_vfcvth_d_s(residual.data);
        n0 = __lsx_vfmadd_d(q0, q0, n0);
        n1 = __lsx_vfmadd_d(q1, q1, n1);
        p0 = __lsx_vfmadd_d(q0, r0, p0);
        p1 = __lsx_vfmadd_d(q1, r1, p1);
    }
    double norms[2], ips[2];
    __lsx_vst((__m128i)__lsx_vfadd_d(n0, n1), norms, 0);
    __lsx_vst((__m128i)__lsx_vfadd_d(p0, p1), ips, 0);
    norm = ip = 0.0;
    for (size_t k = 0; k < 2; ++k) {
        norm += norms[k];
        ip += ips[k];
    }
    for (; j < d; ++j) {
        const double q = decoded[j];
        const float residual = x[j] - (centroid ? centroid[j] : 0.0f);
        norm += q * q;
        ip += q * residual;
    }
}
} // namespace faiss::quantizer_preprocessing
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>
#undef THE_SIMD_LEVEL
#endif // COMPILE_SIMD_LOONGARCH_LSX
