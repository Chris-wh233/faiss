/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/distances.h>
#ifdef COMPILE_SIMD_LOONGARCH_LASX
#include <faiss/impl/simdlib/simdlib_lasx.h>
#include <lasxintrin.h>
#include <bit>
#include <cmath>
namespace faiss {
namespace {
struct FloatVec {
    struct Mask {
        __m256i data;
        Mask operator&(Mask other) const {
            return {__lasx_xvand_v(data, other.data)};
        }
        Mask operator|(Mask other) const {
            return {__lasx_xvor_v(data, other.data)};
        }
        Mask operator~() const {
            return {__lasx_xvnor_v(data, data)};
        }
        uint32_t bits() const {
            return detail::simdlib_lasx::byte_mask(data);
        }
        size_t count() const {
            return std::popcount(bits()) / 4;
        }
        bool any() const {
            return bits() != 0;
        }
    };
    static constexpr size_t width = 8;
    __m256 data;
    explicit FloatVec(__m256 value) : data(value) {}
    explicit FloatVec(float value)
            : data((__m256)__lasx_xvreplgr2vr_w(
                      std::bit_cast<int32_t>(value))) {}
    explicit FloatVec(const float* ptr) : data((__m256)__lasx_xvld(ptr, 0)) {}
    void store(float* ptr) const {
        __lasx_xvst((__m256i)data, ptr, 0);
    }

    FloatVec operator+(FloatVec other) const {
        return FloatVec(__lasx_xvfadd_s(data, other.data));
    }
    FloatVec operator-(FloatVec other) const {
        return FloatVec(__lasx_xvfsub_s(data, other.data));
    }
    FloatVec operator*(FloatVec other) const {
        return FloatVec(__lasx_xvfmul_s(data, other.data));
    }
    FloatVec operator/(FloatVec other) const {
        return FloatVec(__lasx_xvfdiv_s(data, other.data));
    }
    FloatVec minimum_number(FloatVec other) const {
        return FloatVec(__lasx_xvfmin_s(data, other.data));
    }
    FloatVec maximum_number(FloatVec other) const {
        return FloatVec(__lasx_xvfmax_s(data, other.data));
    }
    FloatVec canberra_terms(FloatVec other) const {
        // The scalar metric uses double fabs/add/div for the denominator.
        // In particular, |x| + |y| must not overflow in float.
        const auto a = abs().data, b = other.abs().data;
        const auto delta = (*this - other).abs().data;
        const auto lo = __lasx_xvfdiv_d(
                __lasx_xvfcvtl_d_s(delta),
                __lasx_xvfadd_d(__lasx_xvfcvtl_d_s(a), __lasx_xvfcvtl_d_s(b)));
        const auto hi = __lasx_xvfdiv_d(
                __lasx_xvfcvth_d_s(delta),
                __lasx_xvfadd_d(__lasx_xvfcvth_d_s(a), __lasx_xvfcvth_d_s(b)));
        return FloatVec(__lasx_xvfcvt_s_d(hi, lo));
    }
    Mask equal(FloatVec other) const {
        return {__lasx_xvfcmp_ceq_s(data, other.data)};
    }
    Mask less(FloatVec other) const {
        return {__lasx_xvfcmp_clt_s(data, other.data)};
    }
    Mask less_equal(FloatVec other) const {
        return {__lasx_xvfcmp_cle_s(data, other.data)};
    }
    FloatVec masked(Mask mask) const {
        return FloatVec((__m256)__lasx_xvand_v((__m256i)data, mask.data));
    }
    static FloatVec select(Mask mask, FloatVec yes, FloatVec no) {
        return FloatVec((__m256)__lasx_xvbitsel_v(
                (__m256i)no.data, (__m256i)yes.data, mask.data));
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
        const auto negative = (__m256)__lasx_xvxor_v(
                (__m256i)data, __lasx_xvreplgr2vr_w(0x80000000u));
        const auto lo = __lasx_xvfmul_d(
                __lasx_xvfcvtl_d_s(negative), (__m256d)__lasx_xvld(logs_lo, 0));
        const auto hi = __lasx_xvfmul_d(
                __lasx_xvfcvth_d_s(negative), (__m256d)__lasx_xvld(logs_hi, 0));
        return FloatVec(__lasx_xvfcvt_s_d(hi, lo));
    }
    static FloatVec madd(FloatVec a, FloatVec b, FloatVec c) {
        return FloatVec(__lasx_xvfmadd_s(a.data, b.data, c.data));
    }
    FloatVec abs() const {
        return FloatVec((__m256)__lasx_xvand_v(
                (__m256i)data, __lasx_xvreplgr2vr_w(0x7fffffff)));
    }
    FloatVec min(FloatVec other) const {
        const auto mask = __lasx_xvfcmp_clt_s(other.data, data);
        return FloatVec((__m256)__lasx_xvbitsel_v(
                (__m256i)data, (__m256i)other.data, mask));
    }
    FloatVec max(FloatVec other) const {
        const auto mask = __lasx_xvfcmp_clt_s(data, other.data);
        return FloatVec((__m256)__lasx_xvbitsel_v(
                (__m256i)data, (__m256i)other.data, mask));
    }
    float scalar() const {
        return std::bit_cast<float>(
                static_cast<uint32_t>(__lasx_xvpickve2gr_w((__m256i)data, 0)));
    }
    uint32_t equal_mask(FloatVec other) const {
        return detail::simdlib_lasx::byte_mask(
                __lasx_xvfcmp_ceq_s(data, other.data));
    }

    float reduce_sum() const {
        FloatVec v = *this;
        v = v +
                FloatVec((__m256)__lasx_xvpermi_q(
                        (__m256i)v.data, (__m256i)v.data, 0x01));
        v = v + FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0x4e));
        v = v + FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0xb1));
        return v.scalar();
    }
    float reduce_min() const {
        FloatVec v = *this;
        v = v.min(FloatVec((__m256)__lasx_xvpermi_q(
                (__m256i)v.data, (__m256i)v.data, 0x01)));
        v = v.min(FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0x4e)));
        v = v.min(FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0xb1)));
        return v.scalar();
    }
    float reduce_max() const {
        FloatVec v = *this;
        v = v.max(FloatVec((__m256)__lasx_xvpermi_q(
                (__m256i)v.data, (__m256i)v.data, 0x01)));
        v = v.max(FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0x4e)));
        v = v.max(FloatVec((__m256)__lasx_xvshuf4i_w((__m256i)v.data, 0xb1)));
        return v.scalar();
    }
};
} // namespace
} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LASX
#include <faiss/utils/simd_impl/distances_loongarch-inl.h>
#include <faiss/utils/simd_impl/extra_distances_loongarch-inl.h>
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>
#undef THE_SIMD_LEVEL
#endif // COMPILE_SIMD_LOONGARCH_LASX
