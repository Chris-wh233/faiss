/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/auxv.h>

// Keep the hardware checks in a baseline translation unit.
void test_simdlib_lsx();
TEST(TestSIMDLibLoongArch, LSX) {
    if (!(getauxval(AT_HWCAP) & (1UL << 4))) {
        GTEST_SKIP() << "LSX is unavailable";
    }
    test_simdlib_lsx();
}
