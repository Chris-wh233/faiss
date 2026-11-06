/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/impl/scalar_quantizer/sq-turboquant-dc.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <limits>

#ifndef THE_LEVEL_TO_DISPATCH
#error "THE_LEVEL_TO_DISPATCH should be set on input to this header"
#endif

namespace faiss {

namespace scalar_quantizer {

// Define SL as alias for THE_LEVEL_TO_DISPATCH for use in this file
constexpr SIMDLevel SL = THE_LEVEL_TO_DISPATCH;

// True for every level that runs the 512-bit kernels. AVX512_VPOPCNT is
// unreachable here until an SQ translation unit is compiled at that level;
// naming it keeps the 512-bit alignment rule correct when one is.
template <SIMDLevel SL2>
constexpr bool is_avx512_family = SL2 == SIMDLevel::AVX512 ||
        SL2 == SIMDLevel::AVX512_VPOPCNT || SL2 == SIMDLevel::AVX512_SPR;

// Returns true if dimension d is compatible with the given SIMD level
template <SIMDLevel SL2>
constexpr bool is_dimension_compatible(size_t d) {
    if constexpr (is_avx512_family<SL2>) {
        return d % 16 == 0;
    } else if constexpr (SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
        return d % 8 == 0;
    } else {
        return true; // SIMDLevel::NONE has no alignment requirements
    }
}

// Distance computers and scanners. At AVX512 the 16-bit float codecs decode
// a partial last block, so they take any d.
template <SIMDLevel SL2>
constexpr bool is_dimension_compatible(
        size_t d,
        ScalarQuantizer::QuantizerType qtype) {
    return (SL2 == SIMDLevel::AVX512 &&
            (qtype == ScalarQuantizer::QT_fp16 ||
             qtype == ScalarQuantizer::QT_bf16)) ||
            is_dimension_compatible<SL2>(d);
}

/// Calls f with the DCTemplate type for (Quantizer, Sim, SL2) and d. The
/// 16-bit float quantizers and the QT_8bit inner product use it. Four
/// accumulators per code hide the FMA latency at a large d. At a small d their
/// reduction costs more than it saves. At AVX512 they are faster only on AMD
/// (SIMDConfig::sq_avx512_4_accumulators).
template <class Quantizer, class Sim, SIMDLevel SL2, class F>
auto with_dc_template(size_t d, F f) {
    if constexpr (SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::AVX512) {
        if (d >= 384 &&
            (SL2 == SIMDLevel::AVX2 || SIMDConfig::sq_avx512_4_accumulators)) {
            return f.template
            operator()<DCTemplate<Quantizer, Sim, SL2, true>>();
        }
    }
    return f.template operator()<DCTemplate<Quantizer, Sim, SL2>>();
}

/// with_dc_template for QT_8bit. Only its inner product has a four-accumulator
/// loop.
template <class Sim, SIMDLevel SL2, class F>
auto with_8bit_dc_template(size_t d, F f) {
    using Quantizer = QuantizerTemplate<
            Codec8bit<SL2>,
            QuantizerTemplateScaling::NON_UNIFORM,
            SL2>;
    if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
        return with_dc_template<Quantizer, Sim, SL2>(d, f);
    } else {
        return f.template operator()<DCTemplate<Quantizer, Sim, SL2>>();
    }
}

/*******************************************************************
 * sq_select_quantizer: the big switch returning SQuantizer*
 *******************************************************************/

template <>
ScalarQuantizer::SQuantizer* sq_select_quantizer<THE_LEVEL_TO_DISPATCH>(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // Return nullptr for incompatible dimensions in SIMD cases
    if constexpr (SL != SIMDLevel::NONE) {
        if (!is_dimension_compatible<SL>(d)) {
            return nullptr;
        }
    }
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<
                    Codec8bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<
                    Codec6bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<
                    Codec4bit<SL>,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<
                    Codec8bit<SL>,
                    QuantizerTemplateScaling::UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<
                    Codec4bit<SL>,
                    QuantizerTemplateScaling::UNIFORM,
                    SL>(d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SL>(d, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SL>(d, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned<SL>(d, trained);
        case ScalarQuantizer::QT_0bit:
            FAISS_THROW_MSG(
                    "QT_0bit does not support standalone quantization, use IndexIVFScalarQuantizer");
        case ScalarQuantizer::QT_1bit_tqmse:
            return new QuantizerTurboQuantMSE<1, SL>(d, trained);
        case ScalarQuantizer::QT_2bit_tqmse:
            return new QuantizerTurboQuantMSE<2, SL>(d, trained);
        case ScalarQuantizer::QT_3bit_tqmse:
            return new QuantizerTurboQuantMSE<3, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_tqmse:
            return new QuantizerTurboQuantMSE<4, SL>(d, trained);
        case ScalarQuantizer::QT_8bit_tqmse:
            return new QuantizerTurboQuantMSE<8, SL>(d, trained);
        case ScalarQuantizer::QT_1bit_eden:
            return new QuantizerLloydMax<1, SL>(d, trained);
        case ScalarQuantizer::QT_2bit_eden:
            return new QuantizerLloydMax<2, SL>(d, trained);
        case ScalarQuantizer::QT_3bit_eden:
            return new QuantizerLloydMax<3, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_eden:
            return new QuantizerLloydMax<4, SL>(d, trained);
        case ScalarQuantizer::QT_5bit_eden:
            if constexpr (SL != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new QuantizerLloydMax<5, SL>(d, trained);
            }
        case ScalarQuantizer::QT_6bit_eden:
            if constexpr (SL != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new QuantizerLloydMax<6, SL>(d, trained);
            }
        case ScalarQuantizer::QT_7bit_eden:
            if constexpr (SL != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new QuantizerLloydMax<7, SL>(d, trained);
            }
        case ScalarQuantizer::QT_8bit_eden:
            return new QuantizerLloydMax<8, SL>(d, trained);
        case ScalarQuantizer::QT_2bit_tq:
            return new QuantizerTurboQuantFull<2, SL>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            return new QuantizerTurboQuantFull<3, SL>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            return new QuantizerTurboQuantFull<4, SL>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            return new QuantizerTurboQuantFull<5, SL>(d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

/*******************************************************************
 * select_distance_computer_body: helper for sq_select_distance_computer
 *******************************************************************/

template <class Sim, SIMDLevel SL2>
SQDistanceComputer* select_distance_computer_body(
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    // Return nullptr for incompatible dimensions in SIMD cases
    if constexpr (SL2 != SIMDLevel::NONE) {
        if (!is_dimension_compatible<SL2>(d, qtype)) {
            return nullptr;
        }
    }
    auto new_dc = [&]<class DC>() -> SQDistanceComputer* {
        return new DC(d, trained);
    };
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec8bit<SL2>,
                            QuantizerTemplateScaling::UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit<SL2>,
                            QuantizerTemplateScaling::UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return with_8bit_dc_template<Sim, SL2>(d, new_dc);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec6bit<SL2>,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<
                            Codec4bit<SL2>,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SL2>,
                    Sim,
                    SL2>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return with_dc_template<QuantizerFP16<SL2>, Sim, SL2>(d, new_dc);

        case ScalarQuantizer::QT_bf16:
            return with_dc_template<QuantizerBF16<SL2>, Sim, SL2>(d, new_dc);

        case ScalarQuantizer::QT_8bit_direct:
            if constexpr (is_avx512_family<SL2>) {
                if (d % 32 == 0) {
                    return new DistanceComputerByte<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            } else if constexpr (
                    SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
                if (d % 16 == 0) {
                    return new DistanceComputerByte<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            }
            return new DCTemplate<Quantizer8bitDirect<SL2>, Sim, SL2>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct_signed:
            if constexpr (SL2 == SIMDLevel::AVX512_SPR) {
                if (d % 64 == 0) {
                    return new DistanceComputerByteSigned<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            } else if constexpr (is_avx512_family<SL2>) {
                if (d % 32 == 0) {
                    return new DistanceComputerByteSigned<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            } else if constexpr (
                    SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
                if (d % 16 == 0) {
                    return new DistanceComputerByteSigned<Sim, SL2>(
                            static_cast<int>(d), trained);
                }
            }
            return new DCTemplate<Quantizer8bitDirectSigned<SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_0bit:
            FAISS_THROW_MSG(
                    "QT_0bit does not support standalone distance computation, use IndexIVFScalarQuantizer");
        case ScalarQuantizer::QT_1bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<1, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_2bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<2, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_3bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<3, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<4, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_8bit_tqmse:
            return new DCTemplate<QuantizerTurboQuantMSE<8, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_1bit_eden:
            return new DCTemplate<QuantizerLloydMax<1, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_2bit_eden:
            return new DCTemplate<QuantizerLloydMax<2, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_3bit_eden:
            return new DCTemplate<QuantizerLloydMax<3, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_eden:
            return new DCTemplate<QuantizerLloydMax<4, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_5bit_eden:
            if constexpr (SL2 != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new DCTemplate<QuantizerLloydMax<5, SL2>, Sim, SL2>(
                        d, trained);
            }
        case ScalarQuantizer::QT_6bit_eden:
            if constexpr (SL2 != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new DCTemplate<QuantizerLloydMax<6, SL2>, Sim, SL2>(
                        d, trained);
            }
        case ScalarQuantizer::QT_7bit_eden:
            if constexpr (SL2 != SIMDLevel::NONE) {
                return nullptr;
            } else {
                return new DCTemplate<QuantizerLloydMax<7, SL2>, Sim, SL2>(
                        d, trained);
            }
        case ScalarQuantizer::QT_8bit_eden:
            return new DCTemplate<QuantizerLloydMax<8, SL2>, Sim, SL2>(
                    d, trained);
        case ScalarQuantizer::QT_2bit_tq:
            // FRICTION: bypasses DCTemplate entirely — custom DC
            // that doesn't fit the Quantizer+Similarity decomposition
            return new DCTurboQuantFull<2, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_3bit_tq:
            return new DCTurboQuantFull<3, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_4bit_tq:
            return new DCTurboQuantFull<4, Sim, SL2>(d, trained);
        case ScalarQuantizer::QT_5bit_tq:
            return new DCTurboQuantFull<5, Sim, SL2>(d, trained);
        default:
            FAISS_THROW_MSG("unknown qtype");
    }
}

/*******************************************************************
 * sq_select_distance_computer: returns SQDistanceComputer*
 *******************************************************************/

template <>
SQDistanceComputer* sq_select_distance_computer<THE_LEVEL_TO_DISPATCH>(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        return select_distance_computer_body<SimilarityL2<SL>, SL>(
                qtype, d, trained);
    } else {
        return select_distance_computer_body<SimilarityIP<SL>, SL>(
                qtype, d, trained);
    }
}

/*******************************************************************
 * sq_select_InvertedListScanner: returns InvertedListScanner*
 *******************************************************************/

template <>
InvertedListScanner* sq_select_InvertedListScanner<THE_LEVEL_TO_DISPATCH>(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    auto scan = [&]<class DCClass>() -> InvertedListScanner* {
        if constexpr (DCClass::Sim::metric_type == METRIC_L2) {
            return new IVFSQScannerL2<DCClass>(
                    int(d),
                    trained,
                    code_size,
                    quantizer,
                    store_pairs,
                    sel,
                    by_residual);
        } else if constexpr (
                DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
            return new IVFSQScannerIP<DCClass>(
                    int(d), trained, code_size, store_pairs, sel, by_residual);
        } else {
            FAISS_THROW_MSG("unsupported metric type");
        }
    };

    auto select_by_simd_and_metric =
            [&]<SIMDLevel SL2, class Similarity>() -> InvertedListScanner* {
        // Return nullptr for incompatible dimensions in SIMD cases
        if constexpr (SL2 != SIMDLevel::NONE) {
            if (!is_dimension_compatible<SL2>(d, qtype)) {
                return nullptr;
            }
        }
        switch (qtype) {
            case ScalarQuantizer::QT_8bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec8bit<SL2>,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_uniform:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit<SL2>,
                                QuantizerTemplateScaling::UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit:
                return with_8bit_dc_template<Similarity, SL2>(d, scan);
            case ScalarQuantizer::QT_4bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec4bit<SL2>,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_6bit:
                return scan.template operator()<DCTemplate<
                        QuantizerTemplate<
                                Codec6bit<SL2>,
                                QuantizerTemplateScaling::NON_UNIFORM,
                                SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_fp16:
                return with_dc_template<QuantizerFP16<SL2>, Similarity, SL2>(
                        d, scan);
            case ScalarQuantizer::QT_bf16:
                return with_dc_template<QuantizerBF16<SL2>, Similarity, SL2>(
                        d, scan);
            case ScalarQuantizer::QT_8bit_direct:
                if constexpr (is_avx512_family<SL2>) {
                    if (d % 32 == 0) {
                        return scan.template
                        operator()<DistanceComputerByte<Similarity, SL2>>();
                    }
                } else if constexpr (
                        SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
                    if (d % 16 == 0) {
                        return scan.template
                        operator()<DistanceComputerByte<Similarity, SL2>>();
                    }
                }
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirect<SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit_direct_signed:
                if constexpr (SL2 == SIMDLevel::AVX512_SPR) {
                    if (d % 64 == 0) {
                        return scan.template operator()<
                                DistanceComputerByteSigned<Similarity, SL2>>();
                    }
                } else if constexpr (is_avx512_family<SL2>) {
                    if (d % 32 == 0) {
                        return scan.template operator()<
                                DistanceComputerByteSigned<Similarity, SL2>>();
                    }
                } else if constexpr (
                        SL2 == SIMDLevel::AVX2 || SL2 == SIMDLevel::ARM_NEON) {
                    if (d % 16 == 0) {
                        return scan.template operator()<
                                DistanceComputerByteSigned<Similarity, SL2>>();
                    }
                }
                return scan.template operator()<DCTemplate<
                        Quantizer8bitDirectSigned<SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_0bit:
                return new IVFCoarseDistanceScanner(
                        Similarity::metric_type != METRIC_L2, store_pairs, sel);
            case ScalarQuantizer::QT_1bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<1, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_2bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<2, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_3bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<3, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<4, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_8bit_tqmse:
                return scan.template operator()<DCTemplate<
                        QuantizerTurboQuantMSE<8, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_1bit_eden:
                return scan.template operator()<DCTemplate<
                        QuantizerLloydMax<1, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_2bit_eden:
                return scan.template operator()<DCTemplate<
                        QuantizerLloydMax<2, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_3bit_eden:
                return scan.template operator()<DCTemplate<
                        QuantizerLloydMax<3, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_4bit_eden:
                return scan.template operator()<DCTemplate<
                        QuantizerLloydMax<4, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_5bit_eden:
                if constexpr (SL2 != SIMDLevel::NONE) {
                    return nullptr;
                } else {
                    return scan.template operator()<DCTemplate<
                            QuantizerLloydMax<5, SL2>,
                            Similarity,
                            SL2>>();
                }
            case ScalarQuantizer::QT_6bit_eden:
                if constexpr (SL2 != SIMDLevel::NONE) {
                    return nullptr;
                } else {
                    return scan.template operator()<DCTemplate<
                            QuantizerLloydMax<6, SL2>,
                            Similarity,
                            SL2>>();
                }
            case ScalarQuantizer::QT_7bit_eden:
                if constexpr (SL2 != SIMDLevel::NONE) {
                    return nullptr;
                } else {
                    return scan.template operator()<DCTemplate<
                            QuantizerLloydMax<7, SL2>,
                            Similarity,
                            SL2>>();
                }
            case ScalarQuantizer::QT_8bit_eden:
                return scan.template operator()<DCTemplate<
                        QuantizerLloydMax<8, SL2>,
                        Similarity,
                        SL2>>();
            case ScalarQuantizer::QT_2bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<2, Similarity, SL2>>();
            case ScalarQuantizer::QT_3bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<3, Similarity, SL2>>();
            case ScalarQuantizer::QT_4bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<4, Similarity, SL2>>();
            case ScalarQuantizer::QT_5bit_tq:
                return scan.template
                operator()<DCTurboQuantFull<5, Similarity, SL2>>();
            default:
                FAISS_THROW_MSG("unknown qtype");
        }
    };

    if (mt == METRIC_L2) {
        return select_by_simd_and_metric
                .template operator()<SL, SimilarityL2<SL>>();
    } else if (mt == METRIC_INNER_PRODUCT) {
        return select_by_simd_and_metric
                .template operator()<SL, SimilarityIP<SL>>();
    }
    FAISS_THROW_MSG("unsupported metric type");
}

} // namespace scalar_quantizer

} // namespace faiss
