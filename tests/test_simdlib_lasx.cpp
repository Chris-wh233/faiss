/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/simdlib/simdlib_lasx.h>
#include "test_simdlib_loongarch-inl.h"

void test_simdlib_lasx() {
    check_simdlib<faiss::SIMDLevel::LOONGARCH_LASX>();
}
