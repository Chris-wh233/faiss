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
namespace faiss {
namespace {
// Native register width for distance kernels (4 floats on LSX, 8 on LASX).
struct FloatVec {
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
// Only the four metrics with native LoongArch distance kernels are dispatched.
#define FAISS_IVFFLAT_SCANNER_BASIC_METRICS_ONLY
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>
#undef FAISS_IVFFLAT_SCANNER_BASIC_METRICS_ONLY
#undef THE_SIMD_LEVEL
#endif // COMPILE_SIMD_LOONGARCH_LSX
