/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/sq-loongarch-advanced.h>
#include <faiss/impl/scalar_quantizer/sq-loongarch-ops.h>
#include <faiss/impl/scalar_quantizer/sq-turboquant-dc.h>
#include <faiss/utils/simd_impl/masked_sum-loongarch.h>
namespace faiss::scalar_quantizer {
namespace detail {
template <SIMDLevel SL, QuantizerType QT>
struct LoongArchQuantizer : ScalarQuantizer::SQuantizer {
    using Ops = LoongArchSQOps<SL>;
    using Vec = typename Ops::Vec;
    static constexpr bool normalized = QT == ScalarQuantizer::QT_8bit ||
            QT == ScalarQuantizer::QT_4bit || QT == ScalarQuantizer::QT_6bit ||
            QT == ScalarQuantizer::QT_8bit_uniform ||
            QT == ScalarQuantizer::QT_4bit_uniform;
    static constexpr bool uniform = QT == ScalarQuantizer::QT_8bit_uniform ||
            QT == ScalarQuantizer::QT_4bit_uniform;
    static constexpr bool half =
            QT == ScalarQuantizer::QT_fp16 || QT == ScalarQuantizer::QT_bf16;
    static constexpr int bits = half ? 16
            : (QT == ScalarQuantizer::QT_4bit ||
               QT == ScalarQuantizer::QT_4bit_uniform)
            ? 4
            : QT == ScalarQuantizer::QT_6bit ? 6
                                             : 8;
    using Codec = std::conditional_t<
            bits == 4,
            Codec4bit<SIMDLevel::NONE>,
            std::conditional_t<
                    bits == 6,
                    Codec6bit<SIMDLevel::NONE>,
                    Codec8bit<SIMDLevel::NONE>>>;
    const size_t d;
    const float* trained;
    float vmin = 0, vdiff = 0;
    LoongArchQuantizer(size_t dim, const std::vector<float>& data)
            : d(dim), trained(data.data()) {
        if constexpr (uniform) {
            vmin = data[0];
            vdiff = data[1];
        }
    }
    float minimum(size_t i) const {
        if constexpr (uniform)
            return vmin;
        else
            return trained[i];
    }
    float range(size_t i) const {
        if constexpr (uniform)
            return vdiff;
        else
            return trained[d + i];
    }
    Vec minimum8(size_t i) const {
        if constexpr (uniform)
            return Vec(vmin);
        else
            return Vec(trained + i);
    }
    Vec range8(size_t i) const {
        if constexpr (uniform)
            return Vec(vdiff);
        else
            return Vec(trained + d + i);
    }
    Vec reconstruct_8_components(const uint8_t* code, size_t i) const {
        auto block = code + i * bits / 8;
        if constexpr (half)
            return Ops::template decode_half<QT == ScalarQuantizer::QT_bf16>(
                    block);
        else {
            Vec value = Ops::template decode_codes<bits, normalized>(block);
            if constexpr (normalized)
                return minimum8(i) + value * range8(i);
            else if constexpr (QT == ScalarQuantizer::QT_8bit_direct_signed)
                return value - Vec(128.0f);
            else
                return value;
        }
    }
    float reconstruct_component(const uint8_t* code, size_t i) const {
        if constexpr (half) {
            uint16_t value;
            memcpy(&value, code + 2 * i, 2);
            if constexpr (QT == ScalarQuantizer::QT_bf16)
                return decode_bf16(value);
            else
                return decode_fp16(value);
        } else if constexpr (normalized)
            return minimum(i) + Codec::decode_component(code, i) * range(i);
        else if constexpr (QT == ScalarQuantizer::QT_8bit_direct_signed)
            return float(code[i]) - 128;
        else
            return code[i];
    }
    void encode_vector(const float* x, uint8_t* code) const final {
        size_t i = 0;
        for (; i + 8 <= d; i += 8) {
            Vec value(x + i);
            auto block = code + i * bits / 8;
            if constexpr (half)
                Ops::template encode_half<QT == ScalarQuantizer::QT_bf16>(
                        value, block);
            else {
                if constexpr (normalized)
                    value = Ops::normalize(value, minimum8(i), range8(i));
                else if constexpr (QT == ScalarQuantizer::QT_8bit_direct_signed)
                    value = value + Vec(128.0f);
                Ops::template encode_codes<bits, normalized>(value, block);
            }
        }
        for (; i < d; ++i) {
            if constexpr (half) {
                uint16_t value;
                if constexpr (QT == ScalarQuantizer::QT_bf16)
                    value = encode_bf16(x[i]);
                else
                    value = encode_fp16(x[i]);
                memcpy(code + 2 * i, &value, 2);
            } else if constexpr (normalized) {
                float value =
                        range(i) == 0 ? 0 : (x[i] - minimum(i)) / range(i);
                value = std::min(std::max(value, 0.0f), 1.0f);
                Codec::encode_component(value, code, i);
            } else if constexpr (QT == ScalarQuantizer::QT_8bit_direct_signed)
                code[i] = uint8_t(x[i] + 128.0f);
            else
                code[i] = uint8_t(x[i]);
        }
    }
    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        for (; i + 8 <= d; i += 8)
            reconstruct_8_components(code, i).storeu(x + i);
        for (; i < d; ++i)
            x[i] = reconstruct_component(code, i);
    }
};

template <SIMDLevel SL, class Action>
auto with_loongarch_quantizer(QuantizerType qt, Action action) {
    switch (qt) {
        case ScalarQuantizer::QT_8bit:
            return action.template
            operator()<LoongArchQuantizer<SL, ScalarQuantizer::QT_8bit>>();
        case ScalarQuantizer::QT_4bit:
            return action.template
            operator()<LoongArchQuantizer<SL, ScalarQuantizer::QT_4bit>>();
        case ScalarQuantizer::QT_6bit:
            return action.template
            operator()<LoongArchQuantizer<SL, ScalarQuantizer::QT_6bit>>();
        case ScalarQuantizer::QT_8bit_uniform:
            return action.template operator()<
                    LoongArchQuantizer<SL, ScalarQuantizer::QT_8bit_uniform>>();
        case ScalarQuantizer::QT_4bit_uniform:
            return action.template operator()<
                    LoongArchQuantizer<SL, ScalarQuantizer::QT_4bit_uniform>>();
        case ScalarQuantizer::QT_fp16:
            return action.template
            operator()<LoongArchQuantizer<SL, ScalarQuantizer::QT_fp16>>();
        case ScalarQuantizer::QT_bf16:
            return action.template
            operator()<LoongArchQuantizer<SL, ScalarQuantizer::QT_bf16>>();
        case ScalarQuantizer::QT_8bit_direct:
            return action.template operator()<
                    LoongArchQuantizer<SL, ScalarQuantizer::QT_8bit_direct>>();
        case ScalarQuantizer::QT_8bit_direct_signed:
            return action.template operator()<LoongArchQuantizer<
                    SL,
                    ScalarQuantizer::QT_8bit_direct_signed>>();
        case ScalarQuantizer::QT_1bit_eden:
        case ScalarQuantizer::QT_1bit_tqmse:
            return action.template operator()<LoongArchLloydMax<SL, 1>>();
        case ScalarQuantizer::QT_2bit_eden:
        case ScalarQuantizer::QT_2bit_tqmse:
            return action.template operator()<LoongArchLloydMax<SL, 2>>();
        case ScalarQuantizer::QT_3bit_eden:
        case ScalarQuantizer::QT_3bit_tqmse:
            return action.template operator()<LoongArchLloydMax<SL, 3>>();
        case ScalarQuantizer::QT_4bit_eden:
        case ScalarQuantizer::QT_4bit_tqmse:
            return action.template operator()<LoongArchLloydMax<SL, 4>>();
        case ScalarQuantizer::QT_5bit_eden:
            return action.template operator()<LoongArchLloydMax<SL, 5>>();
        case ScalarQuantizer::QT_6bit_eden:
            return action.template operator()<LoongArchLloydMax<SL, 6>>();
        case ScalarQuantizer::QT_7bit_eden:
            return action.template operator()<LoongArchLloydMax<SL, 7>>();
        case ScalarQuantizer::QT_8bit_eden:
        case ScalarQuantizer::QT_8bit_tqmse:
            return action.template operator()<LoongArchLloydMax<SL, 8>>();
        default:
            return decltype(action.template operator()<LoongArchQuantizer<
                                    SL,
                                    ScalarQuantizer::QT_8bit>>())(nullptr);
    }
}
template <class Quantizer, MetricType metric>
struct LoongArchSQDistanceComputer : SQDistanceComputer {
    using Vec = typename Quantizer::Vec;
    struct Sim {
        static constexpr MetricType metric_type = metric;
    };
    Quantizer quant;
    LoongArchSQDistanceComputer(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}
    template <bool symmetric>
    float distance(const float* x, const uint8_t* a, const uint8_t* b) const {
        Vec acc(0.0f);
        size_t i = 0;
        for (; i + 8 <= quant.d; i += 8) {
            Vec v = quant.reconstruct_8_components(a, i);
            Vec u;
            if constexpr (symmetric)
                u = quant.reconstruct_8_components(b, i);
            else
                u = Vec(x + i);
            if constexpr (metric == METRIC_L2) {
                auto delta = u - v;
                acc = fmadd(delta, delta, acc);
            } else
                acc = fmadd(u, v, acc);
        }
        float sum = horizontal_add(acc);
        for (; i < quant.d; ++i) {
            const float v = quant.reconstruct_component(a, i);
            const float u =
                    symmetric ? quant.reconstruct_component(b, i) : x[i];
            if constexpr (metric == METRIC_L2) {
                float delta = u - v;
                sum += delta * delta;
            } else
                sum += u * v;
        }
        return sum;
    }
    float compute_distance(const float* x, const uint8_t* code) const {
        return distance<false>(x, code, nullptr);
    }
    float compute_code_distance(const uint8_t* a, const uint8_t* b) const {
        return distance<true>(nullptr, a, b);
    }
    void set_query(const float* x) final {
        q = x;
    }
    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
    float symmetric_dis(idx_t i, idx_t j) final {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }
};
} // namespace detail

template <>
struct TurboQuantQueryOps<THE_SIMD_LEVEL> {
    using Ops = detail::LoongArchLloydOps<THE_SIMD_LEVEL>;
    using Vec = typename Ops::Vec;
    static float sum(const float* x, size_t d) {
        Vec acc(0.f);
        size_t j = 0;
        for (; j + 8 <= d; j += 8)
            acc = acc + Vec(x + j);
        float s = horizontal_add(acc);
        for (; j < d; ++j)
            s += x[j];
        return s;
    }
    static void projection(
            float* x,
            size_t d,
            float scale,
            float& sum,
            float& l1) {
        Vec acc(0.f), absacc(0.f);
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            Vec v = Vec(x + j) * Vec(scale);
            v.storeu(x + j);
            acc = acc + v;
            absacc = absacc + Ops::abs(v);
        }
        sum = horizontal_add(acc);
        l1 = horizontal_add(absacc);
        for (; j < d; ++j) {
            x[j] *= scale;
            sum += x[j];
            l1 += std::abs(x[j]);
        }
    }
    static void quantize(
            const float* x,
            size_t d,
            uint8_t qb,
            uint8_t* out,
            float& minimum,
            float& delta) {
        float hi;
        rabitq::minmax_values<THE_SIMD_LEVEL>(x, d, minimum, hi);
        float range = std::max(hi - minimum, 1e-30f),
              maxval = float((1 << qb) - 1), scale = maxval / range;
        delta = range / maxval;
        size_t stride = (d + 7) / 8, j = 0;
        using B = faiss::detail::LoongArchBits<THE_SIMD_LEVEL>;
        for (; j + 8 <= d; j += 8) {
            Vec v = Ops::round_unsigned(
                    (Vec(x + j) - Vec(minimum)) * Vec(scale), int(maxval));
            uint8_t packed[8] = {};
            Ops::template pack<8>(v, packed);
            auto bytes = B::load_partial(packed, 8);
            for (int b = 0; b < qb; ++b)
                out[b * stride + j / 8] = uint8_t(B::bit_plane(bytes, b));
        }
        for (; j < d; ++j) {
            int v = std::max(
                    0,
                    std::min(
                            int(maxval),
                            int(std::round((x[j] - minimum) * scale))));
            for (int b = 0; b < qb; ++b)
                if (v & (1 << b))
                    out[b * stride + j / 8] |= 1 << (j % 8);
        }
    }
    template <typename Quantizer>
    static float mse_dot(
            const Quantizer& quant,
            const float* query,
            const uint8_t* code,
            float scale) {
        return quant.mse_dot(query, code, scale);
    }
    static void intersect(
            const uint8_t* a,
            const uint8_t* b,
            uint8_t* out,
            size_t n) {
        using V = simd32uint8_tpl<THE_SIMD_LEVEL>;
        size_t j = 0;
        for (; j + 32 <= n; j += 32)
            (V(a + j) & V(b + j)).storeu(out + j);
        for (; j < n; ++j)
            out[j] = a[j] & b[j];
    }
};
template <int Bits, MetricType metric>
using LoongArchTurboDC = DCTurboQuantFull<
        Bits,
        std::conditional_t<
                metric == METRIC_L2,
                SimilarityL2<SIMDLevel::NONE>,
                SimilarityIP<SIMDLevel::NONE>>,
        THE_SIMD_LEVEL>;
template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<THE_SIMD_LEVEL>(
        QuantizerType qt,
        size_t d,
        const std::vector<float>& trained) {
    switch (qt) {
        case ScalarQuantizer::QT_2bit_tq:
            return new QuantizerTurboQuantFull<2, THE_SIMD_LEVEL>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            return new QuantizerTurboQuantFull<3, THE_SIMD_LEVEL>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            return new QuantizerTurboQuantFull<4, THE_SIMD_LEVEL>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            return new QuantizerTurboQuantFull<5, THE_SIMD_LEVEL>(d, trained);
        default:
            break;
    }
    return detail::with_loongarch_quantizer<THE_SIMD_LEVEL>(
            qt, [&]<class Quantizer>() -> ScalarQuantizer::SQuantizer* {
                return new Quantizer(d, trained);
            });
}
template <>
SQDistanceComputer* sq_select_distance_computer<THE_SIMD_LEVEL>(
        MetricType metric,
        QuantizerType qt,
        size_t d,
        const std::vector<float>& trained) {
    switch (qt) {
        case ScalarQuantizer::QT_2bit_tq:
            if (metric == METRIC_L2)
                return new LoongArchTurboDC<2, METRIC_L2>(d, trained);
            return new LoongArchTurboDC<2, METRIC_INNER_PRODUCT>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            if (metric == METRIC_L2)
                return new LoongArchTurboDC<3, METRIC_L2>(d, trained);
            return new LoongArchTurboDC<3, METRIC_INNER_PRODUCT>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            if (metric == METRIC_L2)
                return new LoongArchTurboDC<4, METRIC_L2>(d, trained);
            return new LoongArchTurboDC<4, METRIC_INNER_PRODUCT>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            if (metric == METRIC_L2)
                return new LoongArchTurboDC<5, METRIC_L2>(d, trained);
            return new LoongArchTurboDC<5, METRIC_INNER_PRODUCT>(d, trained);
        default:
            break;
    }
    return detail::with_loongarch_quantizer<THE_SIMD_LEVEL>(
            qt, [&]<class Quantizer>() -> SQDistanceComputer* {
                if (metric == METRIC_L2)
                    return new detail::
                            LoongArchSQDistanceComputer<Quantizer, METRIC_L2>(
                                    d, trained);
                return new detail::LoongArchSQDistanceComputer<
                        Quantizer,
                        METRIC_INNER_PRODUCT>(d, trained);
            });
}
template <>
InvertedListScanner* sq_select_InvertedListScanner<THE_SIMD_LEVEL>(
        QuantizerType qt,
        MetricType metric,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    switch (qt) {
        case ScalarQuantizer::QT_2bit_tq:
            if (metric == METRIC_L2)
                return new IVFSQScannerL2<LoongArchTurboDC<2, METRIC_L2>>(
                        int(d),
                        trained,
                        code_size,
                        quantizer,
                        store_pairs,
                        sel,
                        by_residual);
            return new IVFSQScannerIP<
                    LoongArchTurboDC<2, METRIC_INNER_PRODUCT>>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        case ScalarQuantizer::QT_3bit_tq:
            if (metric == METRIC_L2)
                return new IVFSQScannerL2<LoongArchTurboDC<3, METRIC_L2>>(
                        int(d),
                        trained,
                        code_size,
                        quantizer,
                        store_pairs,
                        sel,
                        by_residual);
            return new IVFSQScannerIP<
                    LoongArchTurboDC<3, METRIC_INNER_PRODUCT>>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        case ScalarQuantizer::QT_4bit_tq:
            if (metric == METRIC_L2)
                return new IVFSQScannerL2<LoongArchTurboDC<4, METRIC_L2>>(
                        int(d),
                        trained,
                        code_size,
                        quantizer,
                        store_pairs,
                        sel,
                        by_residual);
            return new IVFSQScannerIP<
                    LoongArchTurboDC<4, METRIC_INNER_PRODUCT>>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        case ScalarQuantizer::QT_5bit_tq:
            if (metric == METRIC_L2)
                return new IVFSQScannerL2<LoongArchTurboDC<5, METRIC_L2>>(
                        int(d),
                        trained,
                        code_size,
                        quantizer,
                        store_pairs,
                        sel,
                        by_residual);
            return new IVFSQScannerIP<
                    LoongArchTurboDC<5, METRIC_INNER_PRODUCT>>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        default:
            break;
    }
    return detail::with_loongarch_quantizer<
            THE_SIMD_LEVEL>(qt, [&]<class Quantizer>() -> InvertedListScanner* {
        if (metric == METRIC_L2)
            return new IVFSQScannerL2<
                    detail::LoongArchSQDistanceComputer<Quantizer, METRIC_L2>>(
                    int(d),
                    trained,
                    code_size,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
        return new IVFSQScannerIP<detail::LoongArchSQDistanceComputer<
                Quantizer,
                METRIC_INNER_PRODUCT>>(
                int(d), trained, code_size, store_pairs, sel, by_residual);
    });
}
} // namespace faiss::scalar_quantizer
