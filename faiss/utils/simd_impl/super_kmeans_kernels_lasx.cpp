/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX
#include <faiss/utils/distances.h>
#include <faiss/utils/simd_impl/super_kmeans_kernels.h>
namespace faiss::detail {
template <>
float block_l2<SIMDLevel::LOONGARCH_LASX>(
        const float* x,
        const float* y,
        int n) {
    return fvec_L2sqr<SIMDLevel::LOONGARCH_LASX>(x, y, n);
}
} // namespace faiss::detail
#endif
