/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <lasxintrin.h>
#include <bit>
namespace faiss {
namespace {
template <class HC>
int pop_best_lasx(MinimaxHeapT<HC>& heap, float* value) {
    assert(heap.k > 0);
    int last = heap.k - 1;
    while (last >= 0 && heap.ids[last] == -1)
        --last;
    if (last < 0)
        return -1;
    // Seed from a valid entry, preserving the scalar path's infinity and NaN
    // behavior.
    float best = heap.dis[last];
    int index = last;
    auto distances = (__m256)__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(best));
    auto indices = __lasx_xvreplgr2vr_w(index);
    const int32_t lanes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto positions = __lasx_xvld(lanes, 0);
    int j = 0;
    for (; j + 8 <= heap.k; j += 8) {
        auto ids = __lasx_xvld(heap.ids.data() + j, 0);
        auto candidate = (__m256)__lasx_xvld(heap.dis.data() + j, 0);
        auto invalid = __lasx_xvseq_w(ids, __lasx_xvreplgr2vr_w(-1));
        auto better = HC::is_max ? __lasx_xvfcmp_cle_s(candidate, distances)
                                 : __lasx_xvfcmp_cle_s(distances, candidate);
        better = __lasx_xvandn_v(invalid, better);
        distances = (__m256)__lasx_xvbitsel_v(
                (__m256i)distances, (__m256i)candidate, better);
        indices = __lasx_xvbitsel_v(indices, positions, better);
        positions = __lasx_xvadd_w(positions, __lasx_xvreplgr2vr_w(8));
    }
    float values[8];
    int32_t offsets[8];
    __lasx_xvst((__m256i)distances, values, 0);
    __lasx_xvst(indices, offsets, 0);
    for (int lane = 0; lane < 8; ++lane)
        if (HC::cmp(best, values[lane]) ||
            (best == values[lane] && offsets[lane] > index)) {
            best = values[lane];
            index = offsets[lane];
        }
    for (; j < heap.k; ++j)
        if (heap.ids[j] != -1 &&
            (HC::cmp(best, heap.dis[j]) ||
             (best == heap.dis[j] && j > index))) {
            best = heap.dis[j];
            index = j;
        }
    if (value)
        *value = best;
    int id = heap.ids[index];
    heap.ids[index] = -1;
    --heap.nvalid;
    return id;
}
} // namespace
template <>
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::LOONGARCH_LASX>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* value) {
    return pop_best_lasx(*heap, value);
}
template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::LOONGARCH_LASX>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* value) {
    return pop_best_lasx(*heap, value);
}
} // namespace faiss
#endif
