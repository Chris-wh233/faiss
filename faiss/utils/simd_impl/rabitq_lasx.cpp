/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX
#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LASX
#define FAISS_RABITQ_LASX
#include <faiss/utils/simd_impl/rabitq_loongarch-inl.h>
#include <faiss/utils/simd_impl/rabitq_quantize_loongarch-inl.h>
#endif
