/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace detail {

template <SIMDLevel SL>
inline void fvec_scale(size_t d, float* x, float scale) {
    for (size_t j = 0; j < d; ++j) {
        x[j] *= scale;
    }
}

// Search clamps negative roundoff; the public conversion API retains it.
template <SIMDLevel SL>
inline void fvec_L2sqr_blas(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm) {
    for (size_t j = 0; j < n; ++j) {
        float dis = x_norm + y_norms[j] - 2 * ip[j];
        ip[j] = dis < 0 ? 0 : dis;
    }
}

template <SIMDLevel SL>
inline void fvec_inner_product_to_L2sqr(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm) {
    for (size_t j = 0; j < n; ++j) {
        ip[j] = x_norm + y_norms[j] - 2 * ip[j];
    }
}

// Return the first minimum strictly below limit, or n if none exists.
template <SIMDLevel SL>
inline size_t fvec_L2sqr_blas_cmax(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm,
        float limit) {
    fvec_L2sqr_blas<SL>(n, ip, y_norms, x_norm);
    size_t best = n;
    for (size_t j = 0; j < n; ++j) {
        if (ip[j] < limit) {
            limit = ip[j];
            best = j;
        }
    }
    return best;
}

#ifdef COMPILE_SIMD_LOONGARCH_LSX
template <>
void fvec_scale<SIMDLevel::LOONGARCH_LSX>(size_t d, float* x, float scale);

template <>
void fvec_L2sqr_blas<SIMDLevel::LOONGARCH_LSX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm);

template <>
void fvec_inner_product_to_L2sqr<SIMDLevel::LOONGARCH_LSX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm);

template <>
size_t fvec_L2sqr_blas_cmax<SIMDLevel::LOONGARCH_LSX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm,
        float limit);

#endif

#ifdef COMPILE_SIMD_LOONGARCH_LASX
template <>
void fvec_scale<SIMDLevel::LOONGARCH_LASX>(size_t d, float* x, float scale);

template <>
void fvec_L2sqr_blas<SIMDLevel::LOONGARCH_LASX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm);

template <>
void fvec_inner_product_to_L2sqr<SIMDLevel::LOONGARCH_LASX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm);

template <>
size_t fvec_L2sqr_blas_cmax<SIMDLevel::LOONGARCH_LASX>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm,
        float limit);

#endif

} // namespace detail
} // namespace faiss
