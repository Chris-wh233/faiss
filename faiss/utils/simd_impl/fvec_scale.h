/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/utils/simd_levels.h>
#include <cstddef>
namespace faiss {
template <SIMDLevel SL>
inline void fvec_scale(size_t d, float* x, float scale) {
    for (size_t j = 0; j < d; ++j)
        x[j] *= scale;
}
#ifdef COMPILE_SIMD_LOONGARCH_LSX
template <>
void fvec_scale<SIMDLevel::LOONGARCH_LSX>(size_t d, float* x, float scale);
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
template <>
void fvec_scale<SIMDLevel::LOONGARCH_LASX>(size_t d, float* x, float scale);
#endif
template <SIMDLevel SL>
inline void fvec_affine(size_t d, float* x, float scale, float bias) {
    for (size_t j = 0; j < d; ++j)
        x[j] = x[j] * scale + bias;
}
#ifdef COMPILE_SIMD_LOONGARCH_LSX
template <>
void fvec_affine<SIMDLevel::LOONGARCH_LSX>(
        size_t d,
        float* x,
        float scale,
        float bias);
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
template <>
void fvec_affine<SIMDLevel::LOONGARCH_LASX>(
        size_t d,
        float* x,
        float scale,
        float bias);
#endif
} // namespace faiss
