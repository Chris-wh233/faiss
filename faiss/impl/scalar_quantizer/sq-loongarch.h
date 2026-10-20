/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/sq-loongarch-ops.h>
namespace faiss::scalar_quantizer {
namespace detail {
// Eight logical lanes on both ISAs; packed codes are processed without
// reading beyond their allocation. Only the final partial block is scalar.
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

// Unsupported families retain the dispatcher's existing fallback chain.
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
        default:
            return decltype(action.template operator()<LoongArchQuantizer<
                                    SL,
                                    ScalarQuantizer::QT_8bit>>())(nullptr);
    }
}
} // namespace detail

template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<THE_SIMD_LEVEL>(
        QuantizerType qt,
        size_t d,
        const std::vector<float>& trained) {
    return detail::with_loongarch_quantizer<THE_SIMD_LEVEL>(
            qt, [&]<class Quantizer>() -> ScalarQuantizer::SQuantizer* {
                return new Quantizer(d, trained);
            });
}
} // namespace faiss::scalar_quantizer
