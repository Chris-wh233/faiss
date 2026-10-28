/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LSX
#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LSX
// Define native specializations before the scanner templates instantiate them.
#include <faiss/impl/pq_code_distance/pq_code_distance-loongarch.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/pq_tables_simdlib256.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-loongarch.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/PQDistanceComputer_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>
#endif
