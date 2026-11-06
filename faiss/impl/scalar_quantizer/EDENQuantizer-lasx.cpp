/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX
#include <faiss/impl/EDENQuantizer.h>
#include <faiss/impl/simdlib/simdlib_lasx.h>
namespace faiss::eden_distance {
void compute_code_dot_lut_batch_8_lasx(
        const uint8_t* const code[8],
        const float* lut,
        int kind,
        size_t packed_size,
        float dots[8]) {
    using Vec = simd8float32_tpl<SIMDLevel::LOONGARCH_LASX>;
    Vec acc(0.f);
    float values[8];
    for (size_t i = 0; i < packed_size; ++i) {
        if (kind == 2) {
            for (int j = 0; j < 8; ++j)
                values[j] = lut[code[j][i] & 15];
            acc = acc + Vec(values);
            for (int j = 0; j < 8; ++j)
                values[j] = lut[16 + (code[j][i] >> 4)];
            acc = acc + Vec(values);
            lut += 32;
        } else {
            for (int j = 0; j < 8; ++j)
                values[j] = lut[code[j][i]];
            acc = acc + Vec(values);
            lut += 256;
        }
    }
    acc.storeu(dots);
}
} // namespace faiss::eden_distance
#endif
