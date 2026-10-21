/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <faiss/utils/extra_distances.h>

namespace faiss {
namespace {
// FloatVec is the native 4/8-float register supplied by the LSX/LASX TU.
template <MetricType metric>
float loongarch_extra_distance(const float* x, const float* y, size_t d) {
    FloatVec numerator(0.0f), denominator(0.0f);
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        const FloatVec a(x + i), b(y + i);
        if constexpr (metric == METRIC_Canberra) {
            numerator = numerator + a.canberra_terms(b);
        } else if constexpr (metric == METRIC_BrayCurtis) {
            numerator = numerator + (a - b).abs();
            denominator = denominator + (a + b).abs();
        } else {
            numerator = numerator + a.minimum_number(b);
            denominator = denominator + a.maximum_number(b);
        }
    }
    float num = numerator.reduce_sum(), den = denominator.reduce_sum();
    for (; i < d; ++i) {
        if constexpr (metric == METRIC_Canberra) {
            num += fabs(x[i] - y[i]) / (fabs(x[i]) + fabs(y[i]));
        } else if constexpr (metric == METRIC_BrayCurtis) {
            num += fabs(x[i] - y[i]);
            den += fabs(x[i] + y[i]);
        } else {
            num += fmin(x[i], y[i]);
            den += fmax(x[i], y[i]);
        }
    }
    if constexpr (metric == METRIC_Canberra)
        return num;
    else
        return num / den;
}
} // namespace

#define DEFINE_LOONGARCH_EXTRA_DISTANCE(metric)                 \
    template <>                                                 \
    float VectorDistance<metric, THE_SIMD_LEVEL>::operator()(   \
            const float* x, const float* y) const {             \
        return loongarch_extra_distance<metric>(x, y, this->d); \
    }
DEFINE_LOONGARCH_EXTRA_DISTANCE(METRIC_Canberra)
DEFINE_LOONGARCH_EXTRA_DISTANCE(METRIC_BrayCurtis)
DEFINE_LOONGARCH_EXTRA_DISTANCE(METRIC_Jaccard)
#undef DEFINE_LOONGARCH_EXTRA_DISTANCE
} // namespace faiss
