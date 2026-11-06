/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <limits>
#include <faiss/utils/extra_distances.h>

namespace faiss {
namespace {
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

namespace faiss {
template <>
float VectorDistance<METRIC_NaNEuclidean, THE_SIMD_LEVEL>::operator()(
        const float* x,
        const float* y) const {
    FloatVec accumulator(0.0f);
    size_t present = 0, i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        const FloatVec a(x + i), b(y + i);
        const auto valid = a.equal(a) & b.equal(b);
        const auto delta = a.masked(valid) - b.masked(valid);
        accumulator = FloatVec::madd(delta, delta, accumulator);
        present += valid.count();
    }
    float sum = accumulator.reduce_sum();
    for (; i < d; ++i) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            const float delta = x[i] - y[i];
            sum += delta * delta;
            ++present;
        }
    }
    return present ? float(d) / float(present) * sum : NAN;
}

template <>
float VectorDistance<METRIC_GOWER, THE_SIMD_LEVEL>::operator()(
        const float* x,
        const float* y) const {
    const FloatVec zero(0.0f), one(1.0f);
    FloatVec accumulator(0.0f);
    size_t present = 0, i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        const FloatVec a(x + i), b(y + i);
        const auto valid = a.equal(a) & b.equal(b);
        const auto numeric = zero.less_equal(a) & zero.less_equal(b);
        const auto categorical = a.less(zero) & b.less(zero);
        const auto in_range = a.less_equal(one) & b.less_equal(one);
        if ((valid & ~((numeric & in_range) | categorical)).any())
            return std::numeric_limits<float>::quiet_NaN();
        const auto numeric_valid = numeric & valid;
        const auto delta =
                (a.masked(numeric_valid) - b.masked(numeric_valid)).abs();
        const auto unequal = FloatVec::select(a.equal(b), zero, one);
        accumulator = accumulator + delta + unequal.masked(categorical & valid);
        present += valid.count();
    }
    float sum = accumulator.reduce_sum();
    for (; i < d; ++i) {
        if (std::isnan(x[i]) || std::isnan(y[i]))
            continue;
        if (x[i] >= 0 && y[i] >= 0) {
            if (x[i] > 1 || y[i] > 1)
                return std::numeric_limits<float>::quiet_NaN();
            sum += fabs(x[i] - y[i]);
        } else if (x[i] < 0 && y[i] < 0) {
            sum += float(x[i] != y[i]);
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
        ++present;
    }
    return present ? sum / present : std::numeric_limits<float>::quiet_NaN();
}

template <>
float VectorDistance<METRIC_Lp, THE_SIMD_LEVEL>::operator()(
        const float* x,
        const float* y) const {
    if (metric_arg == 1.0f)
        return fvec_L1<THE_SIMD_LEVEL>(x, y, d);
    if (metric_arg == 2.0f)
        return fvec_L2sqr<THE_SIMD_LEVEL>(x, y, d);
    FloatVec accumulator(0.0f);
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        float terms[FloatVec::width];
        (FloatVec(x + i) - FloatVec(y + i)).abs().store(terms);
        // Keep libm powf for arbitrary powers, including its special values.
        for (size_t j = 0; j < FloatVec::width; ++j)
            terms[j] = powf(terms[j], metric_arg);
        accumulator = accumulator + FloatVec(terms);
    }
    float sum = accumulator.reduce_sum();
    for (; i < d; ++i)
        sum += powf(fabs(x[i] - y[i]), metric_arg);
    return sum;
}

template <>
float VectorDistance<METRIC_JensenShannon, THE_SIMD_LEVEL>::operator()(
        const float* x,
        const float* y) const {
    FloatVec accumulator(0.0f);
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        const FloatVec a(x + i), b(y + i);
        const auto mean = FloatVec(0.5f) * (a + b);
        const auto kl1 = a.negative_log_product(mean / a);
        const auto kl2 = b.negative_log_product(mean / b);
        accumulator = accumulator + (kl1 + kl2);
    }
    float sum = accumulator.reduce_sum();
    for (; i < d; ++i) {
        const float mean = 0.5 * (x[i] + y[i]);
        const float kl1 = -x[i] * log(mean / x[i]);
        const float kl2 = -y[i] * log(mean / y[i]);
        sum += kl1 + kl2;
    }
    return 0.5 * sum;
}
} // namespace faiss
