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
namespace faiss {
namespace {
// Native register width for distance kernels (4 floats on LSX, 8 on LASX).
struct FloatVec {
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
// Only the four metrics with native LoongArch distance kernels are dispatched.
#define FAISS_IVFFLAT_SCANNER_BASIC_METRICS_ONLY
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>
#undef FAISS_IVFFLAT_SCANNER_BASIC_METRICS_ONLY
#undef THE_SIMD_LEVEL
#endif // COMPILE_SIMD_LOONGARCH_LASX
