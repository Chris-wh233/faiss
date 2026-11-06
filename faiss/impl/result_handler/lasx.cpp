/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX

#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/simdlib/simdlib_lasx.h>

#include <bit>

namespace faiss {
namespace {

template <class C>
__m256i compare(__m256 a, __m256 b) {
    if constexpr (C::is_max) {
        return __lasx_xvfcmp_clt_s(b, a);
    } else {
        return __lasx_xvfcmp_clt_s(a, b);
    }
}

template <class C, bool use_sel>
void top1_add_results_lasx(
        Top1BlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    for (size_t qi = self->i0; qi < self->i1; ++qi) {
        const float* row = dis_tab + (j1 - j0) * (qi - self->i0);
        const float threshold = self->dis_tab[qi];
        float best = threshold;
        __m256 vb = (__m256)__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(best));
        size_t j = 0;
        for (; j + 8 <= j1 - j0; j += 8) {
            const auto v = (__m256)__lasx_xvld(row + j, 0);
            vb = (__m256)__lasx_xvbitsel_v(
                    (__m256i)vb, (__m256i)v, compare<C>(vb, v));
        }
        float lanes[8];
        __lasx_xvst((__m256i)vb, lanes, 0);
        for (float v : lanes) {
            if (C::cmp(best, v)) {
                best = v;
            }
        }
        for (; j < j1 - j0; ++j) {
            if (C::cmp(best, row[j])) {
                best = row[j];
            }
        }
        if (C::cmp(threshold, best)) {
            // Select the first matching ID, including ties across SIMD lanes.
            vb = (__m256)__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(best));
            j = 0;
            for (; j + 8 <= j1 - j0; j += 8) {
                const auto v = (__m256)__lasx_xvld(row + j, 0);
                const uint32_t mask = detail::simdlib_lasx::byte_mask(
                        __lasx_xvfcmp_ceq_s(v, vb));
                if (mask) {
                    j += std::countr_zero(mask) / 4;
                    break;
                }
            }
            for (; row[j] != best; ++j) {
            }
            self->dis_tab[qi] = row[j];
            self->ids_tab[qi] = j0 + j;
        }
    }
}

template <class C, bool use_sel>
void reservoir_add_results_lasx(
        ReservoirBlockResultHandler<C, use_sel>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
#pragma omp parallel for
    for (int64_t qi = self->i0; qi < int64_t(self->i1); ++qi) {
        auto& res = self->reservoirs[qi - self->i0];
        const float* row = dis_tab + (j1 - j0) * (qi - self->i0);
        auto add = [&](float dis, int64_t id) {
            if (C::cmp(res.threshold, dis)) {
                res.vals[res.i] = dis;
                res.ids[res.i++] = id;
                if (res.i == res.capacity) {
                    res.shrink_fuzzy();
                }
            }
        };
        size_t j = 0;
        for (; j + 8 <= j1 - j0; j += 8) {
            const auto threshold = (__m256)__lasx_xvreplgr2vr_w(
                    std::bit_cast<int32_t>(res.threshold));
            const auto v = (__m256)__lasx_xvld(row + j, 0);
            uint32_t mask =
                    detail::simdlib_lasx::byte_mask(compare<C>(threshold, v));
            while (mask) {
                const size_t lane = std::countr_zero(mask) / 4;
                add(row[j + lane], j0 + j + lane);
                mask &= ~(uint32_t(15) << (4 * lane));
            }
        }
        for (; j < j1 - j0; ++j) {
            add(row[j], j0 + j);
        }
    }
}

} // namespace

template <>
void top1_add_results_tpl<
        CMax<float, int64_t>,
        false,
        SIMDLevel::LOONGARCH_LASX>(
        Top1BlockResultHandler<CMax<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<
        CMax<float, int64_t>,
        true,
        SIMDLevel::LOONGARCH_LASX>(
        Top1BlockResultHandler<CMax<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<
        CMin<float, int64_t>,
        false,
        SIMDLevel::LOONGARCH_LASX>(
        Top1BlockResultHandler<CMin<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void top1_add_results_tpl<
        CMin<float, int64_t>,
        true,
        SIMDLevel::LOONGARCH_LASX>(
        Top1BlockResultHandler<CMin<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    top1_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void reservoir_add_results_tpl<
        CMax<float, int64_t>,
        false,
        SIMDLevel::LOONGARCH_LASX>(
        ReservoirBlockResultHandler<CMax<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    reservoir_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void reservoir_add_results_tpl<
        CMax<float, int64_t>,
        true,
        SIMDLevel::LOONGARCH_LASX>(
        ReservoirBlockResultHandler<CMax<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    reservoir_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void reservoir_add_results_tpl<
        CMin<float, int64_t>,
        false,
        SIMDLevel::LOONGARCH_LASX>(
        ReservoirBlockResultHandler<CMin<float, int64_t>, false>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    reservoir_add_results_lasx(self, j0, j1, dis_tab);
}

template <>
void reservoir_add_results_tpl<
        CMin<float, int64_t>,
        true,
        SIMDLevel::LOONGARCH_LASX>(
        ReservoirBlockResultHandler<CMin<float, int64_t>, true>* self,
        size_t j0,
        size_t j1,
        const float* dis_tab) {
    reservoir_add_results_lasx(self, j0, j1, dis_tab);
}

} // namespace faiss

#endif // COMPILE_SIMD_LOONGARCH_LASX
