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
#include <faiss/utils/simd_impl/distances_kernels.h>
#include <faiss/utils/simd_impl/quantizer_preprocessing.h>

namespace faiss {
template <>
void detail::fvec_affine<THE_SIMD_LEVEL>(
        size_t d,
        float* x,
        float scale,
        float bias) {
    const FloatVec factor(scale), offset(bias);
    size_t j = 0;
    for (; j + FloatVec::width <= d; j += FloatVec::width)
        FloatVec::madd(FloatVec(x + j), factor, offset).store(x + j);
    for (; j < d; ++j)
        x[j] = x[j] * scale + bias;
}

template <>
void detail::fvec_scale<THE_SIMD_LEVEL>(size_t d, float* x, float scale) {
    FloatVec factor(scale);
    size_t j = 0;
    for (; j + FloatVec::width <= d; j += FloatVec::width)
        (FloatVec(x + j) * factor).store(x + j);
    for (; j < d; ++j)
        x[j] *= scale;
}

namespace {
template <bool clamp>
void loongarch_L2sqr_blas(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm) {
    const FloatVec x(x_norm), minus_two(-2.0f), zero(0.0f);
    size_t j = 0;
    for (; j + FloatVec::width <= n; j += FloatVec::width) {
        FloatVec dis = FloatVec::madd(
                minus_two, FloatVec(ip + j), x + FloatVec(y_norms + j));
        if constexpr (clamp) {
            dis = FloatVec::select(dis.less(zero), zero, dis);
        }
        dis.store(ip + j);
    }
    for (; j < n; ++j) {
        float dis = std::fma(-2.0f, ip[j], x_norm + y_norms[j]);
        if constexpr (clamp) {
            if (dis < 0)
                dis = 0;
        }
        ip[j] = dis;
    }
}
} // namespace

template <>
void detail::fvec_L2sqr_blas<THE_SIMD_LEVEL>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm) {
    loongarch_L2sqr_blas<true>(n, ip, y_norms, x_norm);
}

template <>
void detail::fvec_inner_product_to_L2sqr<THE_SIMD_LEVEL>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm) {
    loongarch_L2sqr_blas<false>(n, ip, y_norms, x_norm);
}

namespace {

// FloatVec is supplied by the LSX/LASX translation unit.
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

template <>
size_t detail::fvec_L2sqr_blas_cmax<THE_SIMD_LEVEL>(
        size_t n,
        float* ip,
        const float* y_norms,
        float x_norm,
        float limit) {
    fvec_L2sqr_blas<THE_SIMD_LEVEL>(n, ip, y_norms, x_norm);
    return loongarch_argmin(ip, n, limit);
}

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

namespace faiss::quantizer_preprocessing {
template <>
float eden_residual_norm<THE_SIMD_LEVEL>(
        size_t d,
        const float* x,
        const float* centroid) {
    return centroid ? fvec_L2sqr<THE_SIMD_LEVEL>(x, centroid, d)
                    : fvec_norm_L2sqr<THE_SIMD_LEVEL>(x, d);
}
template <>
void rowwise_minmax<THE_SIMD_LEVEL>(
        size_t d,
        const float* x,
        float& lo,
        float& hi) {
    FloatVec vmin(std::numeric_limits<float>::max());
    FloatVec vmax(std::numeric_limits<float>::lowest());
    size_t j = 0;
    for (; j + FloatVec::width <= d; j += FloatVec::width) {
        const FloatVec v(x + j);
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }
    lo = vmin.reduce_min();
    hi = vmax.reduce_max();
    for (; j < d; ++j) {
        lo = std::min(lo, x[j]);
        hi = std::max(hi, x[j]);
    }
    // Ordered scalar min/max retain the first zero's sign on equal values.
    // Locate that zero with vector masks when a bound is zero.
    if (lo == 0.0f || hi == 0.0f) {
        size_t first = 0;
        for (; first + FloatVec::width <= d; first += FloatVec::width) {
            const auto mask = FloatVec(x + first).equal_mask(FloatVec(0.0f));
            if (mask) {
                first += std::countr_zero(mask) / 4;
                break;
            }
        }
        for (; first < d && x[first] != 0.0f; ++first) {
        }
        if (first < d) {
            if (lo == 0.0f)
                lo = x[first];
            if (hi == 0.0f)
                hi = x[first];
        }
    }
}
template <>
void rowwise_normalize<THE_SIMD_LEVEL>(
        size_t d,
        const float* x,
        float lo,
        float scale,
        float* out) {
    size_t j = 0;
    if (scale == 0.0f) {
        const FloatVec zero(0.0f);
        for (; j + FloatVec::width <= d; j += FloatVec::width)
            zero.store(out + j);
        for (; j < d; ++j)
            out[j] = 0.0f;
    } else {
        const float inv = 1.0f / scale;
        const FloatVec offset(lo), factor(inv);
        for (; j + FloatVec::width <= d; j += FloatVec::width)
            ((FloatVec(x + j) - offset) * factor).store(out + j);
        for (; j < d; ++j)
            out[j] = (x[j] - lo) * inv;
    }
}
template <>
void eden_normalize<THE_SIMD_LEVEL>(
        size_t d,
        const float* x,
        const float* centroid,
        float sqrt_d,
        float inv_norm,
        float* out) {
    size_t j = 0;
    const FloatVec root(sqrt_d), inv(inv_norm);
    for (; j + FloatVec::width <= d; j += FloatVec::width) {
        const FloatVec c = centroid ? FloatVec(centroid + j) : FloatVec(0.0f);
        ((FloatVec(x + j) - c) * root * inv).store(out + j);
    }
    for (; j < d; ++j)
        out[j] = (x[j] - (centroid ? centroid[j] : 0.0f)) * sqrt_d * inv_norm;
}
template <>
void eden_reconstruct<THE_SIMD_LEVEL>(
        size_t d,
        const float* x,
        const float* centroid,
        float scale,
        float* out) {
    size_t j = 0;
    const FloatVec factor(scale);
    for (; j + FloatVec::width <= d; j += FloatVec::width) {
        const FloatVec c = centroid ? FloatVec(centroid + j) : FloatVec(0.0f);
        FloatVec::madd(factor, FloatVec(x + j), c).store(out + j);
    }
    for (; j < d; ++j)
        out[j] = (centroid ? centroid[j] : 0.0f) + scale * x[j];
}
} // namespace faiss::quantizer_preprocessing
