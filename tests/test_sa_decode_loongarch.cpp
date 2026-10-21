/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>
#include <gtest/gtest.h>
#ifdef FAISS_TEST_LSX
void test_sa_decode_lsx();
TEST(TestLoongArchSaDecode, LSX) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::LOONGARCH_LSX))
        GTEST_SKIP();
    test_sa_decode_lsx();
}
#endif
#ifdef FAISS_TEST_LASX
void test_sa_decode_lasx();
TEST(TestLoongArchSaDecode, LASX) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::LOONGARCH_LASX))
        GTEST_SKIP();
    test_sa_decode_lasx();
}
#endif
