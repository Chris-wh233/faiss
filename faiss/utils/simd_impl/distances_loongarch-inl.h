/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {
namespace {

// FloatVec is supplied by the LSX/LASX translation unit. All full registers
// use native SIMD operations; the final incomplete register uses scalar tails.
template <MetricType metric>
float loongarch_distance(const float* x, const float* y, size_t d) {
    FloatVec acc(0.0f);
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        FloatVec a(x + i), b(y + i);
        if constexpr (metric == METRIC_INNER_PRODUCT) {
            acc = FloatVec::madd(a, b, acc);
        } else {
            auto delta = a - b;
            if constexpr (metric == METRIC_L2) {
                acc = FloatVec::madd(delta, delta, acc);
            } else if constexpr (metric == METRIC_L1) {
                acc = acc + delta.abs();
            } else {
                acc = acc.max(delta.abs());
            }
        }
    }
    float result = metric == METRIC_Linf ? acc.reduce_max() : acc.reduce_sum();
    for (; i < d; ++i) {
        if constexpr (metric == METRIC_INNER_PRODUCT) {
            result += x[i] * y[i];
        } else if constexpr (metric == METRIC_L2) {
            const float delta = x[i] - y[i];
            result += delta * delta;
        } else if constexpr (metric == METRIC_L1) {
            result += std::fabs(x[i] - y[i]);
        } else {
            result = std::max(result, std::fabs(x[i] - y[i]));
        }
    }
    return result;
}

template <bool inner_product>
void loongarch_batch_4(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    FloatVec acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width) {
        FloatVec q(x + i), a(y0 + i), b(y1 + i), c(y2 + i), e(y3 + i);
        if constexpr (inner_product) {
            acc0 = FloatVec::madd(q, a, acc0);
            acc1 = FloatVec::madd(q, b, acc1);
            acc2 = FloatVec::madd(q, c, acc2);
            acc3 = FloatVec::madd(q, e, acc3);
        } else {
            a = q - a;
            b = q - b;
            c = q - c;
            e = q - e;
            acc0 = FloatVec::madd(a, a, acc0);
            acc1 = FloatVec::madd(b, b, acc1);
            acc2 = FloatVec::madd(c, c, acc2);
            acc3 = FloatVec::madd(e, e, acc3);
        }
    }
    dis0 = acc0.reduce_sum();
    dis1 = acc1.reduce_sum();
    dis2 = acc2.reduce_sum();
    dis3 = acc3.reduce_sum();
    for (; i < d; ++i) {
        if constexpr (inner_product) {
            dis0 += x[i] * y0[i];
            dis1 += x[i] * y1[i];
            dis2 += x[i] * y2[i];
            dis3 += x[i] * y3[i];
        } else {
            float a = x[i] - y0[i], b = x[i] - y1[i], c = x[i] - y2[i],
                  e = x[i] - y3[i];
            dis0 += a * a;
            dis1 += b * b;
            dis2 += c * c;
            dis3 += e * e;
        }
    }
}

// Two vector passes preserve the first index on ties, without limiting indices
// to 32 bits. NaNs never replace a minimum; values >= limit are not candidates.
size_t loongarch_argmin(const float* values, size_t n, float limit) {
    FloatVec minima(limit);
    size_t i = 0;
    for (; i + FloatVec::width <= n; i += FloatVec::width) {
        minima = minima.min(FloatVec(values + i));
    }
    float best = minima.reduce_min();
    for (; i < n; ++i) {
        if (values[i] < best)
            best = values[i];
    }
    if (!(best < limit)) {
        return n;
    }
    i = 0;
    const FloatVec target(best);
    for (; i + FloatVec::width <= n; i += FloatVec::width) {
        const uint32_t mask = FloatVec(values + i).equal_mask(target);
        if (mask) {
            return i + __builtin_ctz(mask) / 4;
        }
    }
    for (; i < n; ++i) {
        if (values[i] == best)
            return i;
    }
    return n;
}
} // namespace

#define DEFINE_LOONGARCH_DISTANCE(name, metric)                            \
    template <>                                                            \
    float name<THE_SIMD_LEVEL>(const float* x, const float* y, size_t d) { \
        return loongarch_distance<metric>(x, y, d);                        \
    }                                                                      \
    template <>                                                            \
    float VectorDistance<metric, THE_SIMD_LEVEL>::operator()(              \
            const float* x, const float* y) const {                        \
        return name<THE_SIMD_LEVEL>(x, y, d);                              \
    }
DEFINE_LOONGARCH_DISTANCE(fvec_L2sqr, METRIC_L2)
DEFINE_LOONGARCH_DISTANCE(fvec_inner_product, METRIC_INNER_PRODUCT)
DEFINE_LOONGARCH_DISTANCE(fvec_L1, METRIC_L1)
DEFINE_LOONGARCH_DISTANCE(fvec_Linf, METRIC_Linf)
#undef DEFINE_LOONGARCH_DISTANCE

template <>
float fvec_norm_L2sqr<THE_SIMD_LEVEL>(const float* x, size_t d) {
    return loongarch_distance<METRIC_INNER_PRODUCT>(x, x, d);
}

template <>
void fvec_inner_product_batch_4<THE_SIMD_LEVEL>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    loongarch_batch_4<true>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}
template <>
void fvec_L2sqr_batch_4<THE_SIMD_LEVEL>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    loongarch_batch_4<false>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

template <>
void fvec_inner_products_ny<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        fvec_inner_product_batch_4<THE_SIMD_LEVEL>(
                x,
                y + i * d,
                y + (i + 1) * d,
                y + (i + 2) * d,
                y + (i + 3) * d,
                d,
                dis[i],
                dis[i + 1],
                dis[i + 2],
                dis[i + 3]);
    }
    for (; i < ny; ++i)
        dis[i] = fvec_inner_product<THE_SIMD_LEVEL>(x, y + i * d, d);
}
template <>
void fvec_L2sqr_ny<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        fvec_L2sqr_batch_4<THE_SIMD_LEVEL>(
                x,
                y + i * d,
                y + (i + 1) * d,
                y + (i + 2) * d,
                y + (i + 3) * d,
                d,
                dis[i],
                dis[i + 1],
                dis[i + 2],
                dis[i + 3]);
    }
    for (; i < ny; ++i)
        dis[i] = fvec_L2sqr<THE_SIMD_LEVEL>(x, y + i * d, d);
}
template <>
void fvec_L2sqr_ny_transposed<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    const float norm = fvec_norm_L2sqr<THE_SIMD_LEVEL>(x, d);
    size_t i = 0;
    for (; i + FloatVec::width <= ny; i += FloatVec::width) {
        FloatVec dot(0.0f);
        for (size_t j = 0; j < d; ++j)
            dot = FloatVec::madd(
                    FloatVec(x[j]), FloatVec(y + j * d_offset + i), dot);
        (FloatVec(norm) + FloatVec(y_sqlen + i) - FloatVec(2.0f) * dot)
                .store(dis + i);
    }
    for (; i < ny; ++i) {
        float dot = 0;
        for (size_t j = 0; j < d; ++j)
            dot += x[j] * y[j * d_offset + i];
        dis[i] = norm + y_sqlen[i] - 2 * dot;
    }
}
template <>
size_t fvec_L2sqr_ny_nearest<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    fvec_L2sqr_ny<THE_SIMD_LEVEL>(dis, x, y, d, ny);
    size_t i =
            loongarch_argmin(dis, ny, std::numeric_limits<float>::infinity());
    return i < ny ? i : 0;
}
template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<THE_SIMD_LEVEL>(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    fvec_L2sqr_ny_transposed<THE_SIMD_LEVEL>(
            dis, x, y, y_sqlen, d, d_offset, ny);
    size_t i =
            loongarch_argmin(dis, ny, std::numeric_limits<float>::infinity());
    return i < ny ? i : 0;
}
template <>
void fvec_madd<THE_SIMD_LEVEL>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    size_t i = 0;
    const FloatVec factor(bf);
    for (; i + FloatVec::width <= n; i += FloatVec::width)
        FloatVec::madd(factor, FloatVec(b + i), FloatVec(a + i)).store(c + i);
    for (; i < n; ++i)
        c[i] = a[i] + bf * b[i];
}
template <>
int fvec_madd_and_argmin<THE_SIMD_LEVEL>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    fvec_madd<THE_SIMD_LEVEL>(n, a, bf, b, c);
    const size_t i = loongarch_argmin(c, n, 1e20f);
    return i < n ? static_cast<int>(i) : -1;
}
template <>
void fvec_add<THE_SIMD_LEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width)
        (FloatVec(a + i) + FloatVec(b + i)).store(c + i);
    for (; i < d; ++i)
        c[i] = a[i] + b[i];
}
template <>
void fvec_add<THE_SIMD_LEVEL>(size_t d, const float* a, float b, float* c) {
    size_t i = 0;
    const FloatVec value(b);
    for (; i + FloatVec::width <= d; i += FloatVec::width)
        (FloatVec(a + i) + value).store(c + i);
    for (; i < d; ++i)
        c[i] = a[i] + b;
}
template <>
void fvec_sub<THE_SIMD_LEVEL>(
        size_t d,
        const float* a,
        const float* b,
        float* c) {
    size_t i = 0;
    for (; i + FloatVec::width <= d; i += FloatVec::width)
        (FloatVec(a + i) - FloatVec(b + i)).store(c + i);
    for (; i < d; ++i)
        c[i] = a[i] - b[i];
}
} // namespace faiss
