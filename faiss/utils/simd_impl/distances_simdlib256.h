/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/distances.h>

namespace faiss {

template <>
void fvec_sub<THE_SIMD_LEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMD_LEVEL> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai - bi;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] - b[i];
    }
}

template <>
void fvec_add<THE_SIMD_LEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i;
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMD_LEVEL> ci, ai, bi;
        ai.loadu(a + i);
        bi.loadu(b + i);
        ci = ai + bi;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b[i];
    }
}

template <>
void fvec_add<THE_SIMD_LEVEL>(size_t d, const float* a, float b, float* c) {
    size_t i;
    simd8float32_tpl<THE_SIMD_LEVEL> bv(b);
    for (i = 0; i + 7 < d; i += 8) {
        simd8float32_tpl<THE_SIMD_LEVEL> ci, ai;
        ai.loadu(a + i);
        ci = ai + bv;
        ci.storeu(c + i);
    }
    for (; i < d; i++) {
        c[i] = a[i] + b;
    }
}

} // namespace faiss

#include <faiss/utils/simd_impl/pq_tables_simdlib256.h>
