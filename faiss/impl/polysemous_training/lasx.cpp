/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_LOONGARCH_LASX

#include <faiss/impl/PolysemousTraining.h>
#include <faiss/impl/polysemous_training/dispatch.h>

#include <lasxintrin.h>
#include <algorithm>
#include <bit>

namespace faiss {
namespace {

template <bool Hamming>
double actual_distance(const double* source, int n, int a, int b) {
    if constexpr (Hamming) {
        return std::popcount(uint64_t(a) ^ uint64_t(b));
    } else {
        return source[a * n + b];
    }
}

template <bool Hamming>
__m256d actual_distance_batch(
        const double* source,
        int n,
        const int64_t* a,
        const int64_t* b) {
    if constexpr (Hamming) {
        return __lasx_xvffint_d_l(__lasx_xvpcnt_d(
                __lasx_xvxor_v(__lasx_xvld(a, 0), __lasx_xvld(b, 0))));
    } else {
        double values[4];
        for (int k = 0; k < 4; ++k) {
            values[k] = source[a[k] * n + b[k]];
        }
        return (__m256d)__lasx_xvld(values, 0);
    }
}

template <bool Hamming>
double compute_cost(
        int n,
        const int* perm,
        const double* target,
        const double* weights,
        const double* source) {
    auto sum = (__m256d)__lasx_xvreplgr2vr_d(0);
    double tail = 0;
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            int64_t a[4], b[4];
            for (int k = 0; k < 4; ++k) {
                a[k] = perm[i];
                b[k] = perm[j + k];
            }
            const auto wanted = (__m256d)__lasx_xvld(target + i * n + j, 0);
            const auto weight = (__m256d)__lasx_xvld(weights + i * n + j, 0);
            const auto diff = __lasx_xvfsub_d(
                    wanted, actual_distance_batch<Hamming>(source, n, a, b));
            sum = __lasx_xvfadd_d(
                    sum, __lasx_xvfmul_d(weight, __lasx_xvfmul_d(diff, diff)));
        }
        for (; j < n; ++j) {
            const double diff = target[i * n + j] -
                    actual_distance<Hamming>(source, n, perm[i], perm[j]);
            tail += weights[i * n + j] * diff * diff;
        }
    }
    double values[4];
    __lasx_xvst((__m256i)sum, values, 0);
    for (double v : values) {
        tail += v;
    }
    return tail;
}

template <bool Hamming>
double cost_update(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target,
        const double* weights,
        const double* source) {
    auto sum = (__m256d)__lasx_xvreplgr2vr_d(0);
    double tail = 0;
    auto swapped = [&](int i) { return i == iw ? jw : i == jw ? iw : i; };
    auto process = [&](int row, int column, bool full_row, int begin, int end) {
        int j = begin;
        for (; j + 4 <= end; j += 4) {
            int64_t a[4], b[4], na[4], nb[4];
            double wanted[4], weight[4];
            for (int k = 0; k < 4; ++k) {
                const int ri = full_row ? row : j + k;
                const int cj = full_row ? j + k : column;
                a[k] = perm[ri];
                b[k] = perm[cj];
                na[k] = perm[swapped(ri)];
                nb[k] = perm[swapped(cj)];
                wanted[k] = target[ri * n + cj];
                weight[k] = weights[ri * n + cj];
            }
            const auto w = (__m256d)__lasx_xvld(weight, 0);
            const auto t = (__m256d)__lasx_xvld(wanted, 0);
            const auto old_diff = __lasx_xvfsub_d(
                    t, actual_distance_batch<Hamming>(source, n, a, b));
            const auto new_diff = __lasx_xvfsub_d(
                    t, actual_distance_batch<Hamming>(source, n, na, nb));
            sum = __lasx_xvfsub_d(
                    sum,
                    __lasx_xvfmul_d(w, __lasx_xvfmul_d(old_diff, old_diff)));
            sum = __lasx_xvfadd_d(
                    sum,
                    __lasx_xvfmul_d(w, __lasx_xvfmul_d(new_diff, new_diff)));
        }
        for (; j < end; ++j) {
            const int ri = full_row ? row : j;
            const int cj = full_row ? j : column;
            const double wanted = target[ri * n + cj];
            const double weight = weights[ri * n + cj];
            const double old_diff = wanted -
                    actual_distance<Hamming>(source, n, perm[ri], perm[cj]);
            const double new_diff =
                    wanted -
                    actual_distance<Hamming>(
                            source, n, perm[swapped(ri)], perm[swapped(cj)]);
            tail -= weight * old_diff * old_diff;
            tail += weight * new_diff * new_diff;
        }
    };
    process(iw, 0, true, 0, n);
    process(jw, 0, true, 0, n);
    const int lo = std::min(iw, jw), hi = std::max(iw, jw);
    for (int column : {iw, jw}) {
        process(0, column, false, 0, lo);
        process(0, column, false, lo + 1, hi);
        process(0, column, false, hi + 1, n);
    }
    double values[4];
    __lasx_xvst((__m256i)sum, values, 0);
    for (double v : values) {
        tail += v;
    }
    return tail;
}

} // namespace

namespace polysemous_training {

template <>
double hamming_compute_cost<SIMDLevel::LOONGARCH_LASX>(
        int n,
        const int* perm,
        const double* target_dis,
        const double* weights) {
    return compute_cost<true>(n, perm, target_dis, weights, nullptr);
}

template <>
double hamming_cost_update<SIMDLevel::LOONGARCH_LASX>(
        int n,
        const int* perm,
        int iw,
        int jw,
        const double* target_dis,
        const double* weights) {
    return cost_update<true>(n, perm, iw, jw, target_dis, weights, nullptr);
}

template <>
double distances_compute_cost<SIMDLevel::LOONGARCH_LASX>(
        const ReproduceDistancesObjective& obj,
        const int* perm) {
    return compute_cost<false>(
            obj.n,
            perm,
            obj.target_dis,
            obj.weights.data(),
            obj.source_dis.data());
}

template <>
double distances_cost_update<SIMDLevel::LOONGARCH_LASX>(
        const ReproduceDistancesObjective& obj,
        const int* perm,
        int iw,
        int jw) {
    return cost_update<false>(
            obj.n,
            perm,
            iw,
            jw,
            obj.target_dis,
            obj.weights.data(),
            obj.source_dis.data());
}

} // namespace polysemous_training
} // namespace faiss

#endif // COMPILE_SIMD_LOONGARCH_LASX
