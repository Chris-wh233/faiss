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

// Replace one BLAS inner-product row with squared L2 distances, clamping
// negative roundoff to zero. The multiply-subtract is fused, as in the
// optimized BLAS top-1 kernels. Ordered comparisons preserve NaNs.
template <SIMDLevel SL>
void fvec_L2sqr_blas(size_t n, float* ip, const float* y_norms, float x_norm);

} // namespace faiss
