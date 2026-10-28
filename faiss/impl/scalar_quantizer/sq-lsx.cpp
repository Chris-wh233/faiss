/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LSX
#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LSX
#include <faiss/impl/scalar_quantizer/sq-loongarch.h>
#endif
