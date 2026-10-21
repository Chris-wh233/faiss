/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <faiss/utils/simd_levels.h>

namespace faiss::quantizer_preprocessing {
// Keep the finite initial bounds and ordered comparisons of RowwiseMinMax.
template <SIMDLevel SL>
inline void rowwise_minmax(size_t d, const float* x, float& lo, float& hi) {
    lo = std::numeric_limits<float>::max();
    hi = std::numeric_limits<float>::lowest();
    for (size_t j = 0; j < d; ++j) {
        lo = std::min(lo, x[j]);
        hi = std::max(hi, x[j]);
    }
}
template <SIMDLevel SL>
inline void rowwise_normalize(
        size_t d,
        const float* x,
        float lo,
        float scale,
        float* out) {
    if (scale == 0) {
        std::fill_n(out, d, 0.0f);
    } else {
        const float inv = 1.0f / scale;
        for (size_t j = 0; j < d; ++j)
            out[j] = (x[j] - lo) * inv;
    }
}
template <SIMDLevel SL>
inline void eden_normalize(
        size_t d,
        const float* x,
        const float* centroid,
        float sqrt_d,
        float inv_norm,
        float* out) {
    for (size_t j = 0; j < d; ++j)
        out[j] = (x[j] - (centroid ? centroid[j] : 0.0f)) * sqrt_d * inv_norm;
}
template <SIMDLevel SL>
inline void eden_reconstruct(
        size_t d,
        const float* x,
        const float* centroid,
        float scale,
        float* out) {
    for (size_t j = 0; j < d; ++j)
        out[j] = (centroid ? centroid[j] : 0.0f) + scale * x[j];
}
template <SIMDLevel SL>
inline void eden_code_stats(
        size_t d,
        const float* x,
        const float* centroid,
        const float* decoded,
        double& norm,
        double& ip) {
    norm = ip = 0.0;
    for (size_t j = 0; j < d; ++j) {
        const float residual = x[j] - (centroid ? centroid[j] : 0.0f);
        const double q = decoded[j];
        norm += q * q;
        ip += q * residual;
    }
}

#ifdef COMPILE_SIMD_LOONGARCH_LSX
template <>
void rowwise_minmax<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        float& lo,
        float& hi);
template <>
void rowwise_normalize<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        float lo,
        float scale,
        float* out);
template <>
void eden_normalize<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        const float* centroid,
        float sqrt_d,
        float inv_norm,
        float* out);
template <>
void eden_reconstruct<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        const float* centroid,
        float scale,
        float* out);
template <>
void eden_code_stats<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        const float* x,
        const float* centroid,
        const float* decoded,
        double& norm,
        double& ip);
#endif

#ifdef COMPILE_SIMD_LOONGARCH_LASX
template <>
void rowwise_minmax<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const float* x,
        float& lo,
        float& hi);
template <>
void rowwise_normalize<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const float* x,
        float lo,
        float scale,
        float* out);
template <>
void eden_normalize<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const float* x,
        const float* centroid,
        float sqrt_d,
        float inv_norm,
        float* out);
template <>
void eden_reconstruct<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const float* x,
        const float* centroid,
        float scale,
        float* out);
template <>
void eden_code_stats<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        const float* x,
        const float* centroid,
        const float* decoded,
        double& norm,
        double& ip);
#endif
} // namespace faiss::quantizer_preprocessing
