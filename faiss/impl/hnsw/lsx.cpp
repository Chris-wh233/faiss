/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LSX
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <lsxintrin.h>
#include <bit>
namespace faiss {
namespace {
template <class HC>
int pop_best_lsx(MinimaxHeapT<HC>& heap, float* value) {
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
    auto distances = (__m128)__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(best));
    auto indices = __lsx_vreplgr2vr_w(index);
    const int32_t lanes[4] = {0, 1, 2, 3};
    auto positions = __lsx_vld(lanes, 0);
    int j = 0;
    for (; j + 4 <= heap.k; j += 4) {
        auto ids = __lsx_vld(heap.ids.data() + j, 0);
        auto candidate = (__m128)__lsx_vld(heap.dis.data() + j, 0);
        auto invalid = __lsx_vseq_w(ids, __lsx_vreplgr2vr_w(-1));
        auto better = HC::is_max ? __lsx_vfcmp_cle_s(candidate, distances)
                                 : __lsx_vfcmp_cle_s(distances, candidate);
        better = __lsx_vandn_v(invalid, better);
        distances = (__m128)__lsx_vbitsel_v(
                (__m128i)distances, (__m128i)candidate, better);
        indices = __lsx_vbitsel_v(indices, positions, better);
        positions = __lsx_vadd_w(positions, __lsx_vreplgr2vr_w(4));
    }
    float values[4];
    int32_t offsets[4];
    __lsx_vst((__m128i)distances, values, 0);
    __lsx_vst(indices, offsets, 0);
    for (int lane = 0; lane < 4; ++lane)
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
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::LOONGARCH_LSX>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* value) {
    return pop_best_lsx(*heap, value);
}
template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::LOONGARCH_LSX>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* value) {
    return pop_best_lsx(*heap, value);
}
} // namespace faiss
#endif
