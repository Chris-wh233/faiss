/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/utils/rabitq_simd.h>
#include <limits>
namespace faiss::scalar_quantizer {
// The SPR scalar-quantizer implementation uses VPOPCNTDQ for its RaBitQ
// popcount helpers, but those helpers belong to the narrower VPOPCNT
// capability. Keep the SQ implementation at AVX512_SPR while explicitly
// reusing the independently dispatched VPOPCNT kernels.
template <SIMDLevel SL0>
inline constexpr SIMDLevel rabitq_popcount_level =
        SL0 == SIMDLevel::AVX512_SPR ? SIMDLevel::AVX512_VPOPCNT : SL0;

/*******************************************************************
 * TurboQuant SIMD kernel: masked_sum
 * Compute sum of arr[j] where bit j of the bitmask is set.
 * NONE specialization is inline; AVX2/AVX512/NEON specializations
 * live in sq-avx2.cpp / sq-avx512.cpp / sq-neon.cpp.
 *******************************************************************/

template <SIMDLevel SL0>
float turboq_masked_sum(const float* arr, const uint8_t* bits, size_t d);

template <>
inline float turboq_masked_sum<SIMDLevel::NONE>(
        const float* arr,
        const uint8_t* bits,
        size_t d) {
    float result = 0;
    for (size_t byte_idx = 0; byte_idx < (d + 7) / 8; byte_idx++) {
        uint8_t b = bits[byte_idx];
        size_t base = byte_idx * 8;
        size_t end = std::min(base + 8, d);
        for (size_t j = base; j < end; j++) {
            if (b & (1 << (j - base))) {
                result += arr[j];
            }
        }
    }
    return result;
}

template <SIMDLevel SL>
struct TurboQuantQueryOps {
    static float sum(const float* x, size_t d) {
        float s = 0;
        for (size_t j = 0; j < d; ++j)
            s += x[j];
        return s;
    }
    static void projection(
            float* x,
            size_t d,
            float scale,
            float& sum,
            float& l1) {
        sum = l1 = 0;
        for (size_t j = 0; j < d; ++j) {
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
        minimum = *std::min_element(x, x + d);
        float hi = *std::max_element(x, x + d);
        float range = std::max(hi - minimum, 1e-30f),
              maxval = float((1 << qb) - 1), scale = maxval / range;
        delta = range / maxval;
        size_t stride = (d + 7) / 8;
        for (size_t j = 0; j < d; ++j) {
            int v = int(std::round((x[j] - minimum) * scale));
            v = std::max(0, std::min(int(maxval), v));
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
        float sum = 0;
        for (size_t j = 0; j < quant.d; j++) {
            uint8_t idx = quant.load_mse_index(code, j);
            sum += query[j] * quant.centroids[idx] * scale;
        }
        return sum;
    }
    static void intersect(
            const uint8_t* a,
            const uint8_t* b,
            uint8_t* out,
            size_t n) {
        for (size_t j = 0; j < n; ++j)
            out[j] = a[j] & b[j];
    }
};
template <int NBits, class Similarity, SIMDLevel SL2>
struct DCTurboQuantFull : ScalarQuantizer::TurboQuantRefine::DistanceComputer {
    using Sim = Similarity;
    QuantizerTurboQuantFull<NBits, SL2> quant;
    std::vector<float> query;
    std::vector<float> query_proj;
    float q_norm_sq = 0;
    float qjl_coeff = 0;
    float total_qproj_sum = 0;

    // Pre-screening state
    const float* threshold_ptr = nullptr;
    bool prescreen_l2 = false;
    float qjl_error_coeff = 0;
    mutable size_t n_total = 0;
    mutable size_t n_skipped = 0;

    // Integer popcount state
    uint8_t qb = 0;
    bool int_qjl = false;
    std::vector<uint8_t> rearranged_q;
    float mse_base = 0;
    float mse_int_scale = 0;
    float mse_popcnt_scale = 0;

    // Integer QJL popcount state
    std::vector<uint8_t> rearranged_qproj;
    float qjl_int_scale = 0;
    float qjl_popcnt_scale = 0;

    // Scaled centroids for 1-bit MSE fast path (NBits==2)
    float scaled_c0 = 0;
    float scaled_c1 = 0;
    float delta_centroid = 0;
    float total_q_sum = 0;

    // Multi-bit MSE decomposed coefficients (NBits==3, kMSEBits==2)
    float mse_multi_base = 0;
    float mse_coeff_s0 = 0;
    float mse_coeff_s1 = 0;
    float mse_coeff_s01 = 0;
    mutable std::vector<uint8_t> scratch_and;

    DCTurboQuantFull(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {
        qjl_coeff = std::sqrt(M_PI / 2.0f) / static_cast<float>(d);
    }

    void configure(uint8_t qb_in, bool int_qjl_in) override {
        qb = qb_in;
        int_qjl = int_qjl_in;
    }

    void set_prescreen_threshold(const float* ptr, bool l2) override {
        threshold_ptr = ptr;
        prescreen_l2 = l2;
    }

    void clear_prescreen_threshold() override {
        threshold_ptr = nullptr;
    }

    void set_query(const float* x) final {
        q = x;
        size_t d = quant.d;
        query.assign(x, x + d);
        q_norm_sq = fvec_norm_L2sqr(x, d);

        // Project query
        query_proj.resize(d);
        quant.project_forward(x, query_proj.data());
        float inv_sqrt_pd =
                1.0f / std::sqrt(static_cast<float>(quant.padded_d));
        float qproj_l1 = 0;
        TurboQuantQueryOps<SL2>::projection(
                query_proj.data(), d, inv_sqrt_pd, total_qproj_sum, qproj_l1);
        qjl_error_coeff = qjl_coeff * qproj_l1;

        // Pre-compute for 1-bit MSE fast path
        if constexpr (NBits == 2) {
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            scaled_c0 = quant.centroids[0] * inv_sqrt_d;
            scaled_c1 = quant.centroids[1] * inv_sqrt_d;
            delta_centroid = scaled_c1 - scaled_c0;
            total_q_sum = TurboQuantQueryOps<SL2>::sum(query.data(), d);

            // Integer popcount setup
            if (qb > 0) {
                size_t byte_size = (d + 7) / 8;
                float q_min, delta_q;
                rearranged_q.assign(byte_size * qb, 0);
                TurboQuantQueryOps<SL2>::quantize(
                        query.data(),
                        d,
                        qb,
                        rearranged_q.data(),
                        q_min,
                        delta_q);
                mse_base = scaled_c0 * total_q_sum;
                mse_int_scale = delta_centroid * delta_q;
                mse_popcnt_scale = delta_centroid * q_min;
            }
        }

        // Pre-compute for 2-bit MSE decomposed path (NBits==3)
        if constexpr (NBits == 3) {
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
            const float* c = quant.centroids;
            total_q_sum = TurboQuantQueryOps<SL2>::sum(query.data(), d);
            mse_multi_base = c[0] * inv_sqrt_d * total_q_sum;
            mse_coeff_s0 = (c[1] - c[0]) * inv_sqrt_d;
            mse_coeff_s1 = (c[2] - c[0]) * inv_sqrt_d;
            mse_coeff_s01 = (c[3] - c[2] - c[1] + c[0]) * inv_sqrt_d;
            scratch_and.resize((d + 7) / 8);
        }

        // Integer QJL: quantize projected query into bit-planes
        if (qb > 0 && int_qjl) {
            size_t byte_size = (d + 7) / 8;
            float qp_min, delta_qp;
            rearranged_qproj.assign(byte_size * qb, 0);
            TurboQuantQueryOps<SL2>::quantize(
                    query_proj.data(),
                    d,
                    qb,
                    rearranged_qproj.data(),
                    qp_min,
                    delta_qp);
            qjl_popcnt_scale = qp_min;
            qjl_int_scale = delta_qp;
        }

        n_total = 0;
        n_skipped = 0;
    }

    float query_to_code(const uint8_t* code) const final {
        size_t d = quant.d;
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
        const auto* factors = reinterpret_cast<const SQTurboQFactors*>(
                code + quant.mse_total_bytes + quant.qjl_plane_bytes);
        float norm = factors->norm;
        float gamma = factors->gamma;

        // Stage 1: MSE dot product
        float mse_dot = 0;
        if constexpr (NBits == 2) {
            if (qb > 0) {
                // Integer popcount path for 1-bit MSE
                size_t byte_size = (d + 7) / 8;
                uint64_t and_result = rabitq::bitwise_and_dot_product<
                        rabitq_popcount_level<SL2>>(
                        rearranged_q.data(), code, byte_size, qb);
                uint64_t pop = rabitq::popcount<rabitq_popcount_level<SL2>>(
                        code, byte_size);
                mse_dot = mse_base +
                        mse_int_scale * static_cast<float>(and_result) +
                        mse_popcnt_scale * static_cast<float>(pop);
            } else {
                // Float path: masked accumulation
                float pos_sum = turboq_masked_sum<SL2>(query.data(), code, d);
                mse_dot = scaled_c0 * total_q_sum + delta_centroid * pos_sum;
            }
        } else if constexpr (NBits == 3) {
            // 2-bit MSE: decompose into 3 masked sums over bit-planes.
            size_t pb = quant.mse_plane_bytes;
            float s0 = turboq_masked_sum<SL2>(query.data(), code, d);
            float s1 = turboq_masked_sum<SL2>(query.data(), code + pb, d);
            TurboQuantQueryOps<SL2>::intersect(
                    code, code + pb, scratch_and.data(), pb);
            float s01 =
                    turboq_masked_sum<SL2>(query.data(), scratch_and.data(), d);
            mse_dot = mse_multi_base + mse_coeff_s0 * s0 + mse_coeff_s1 * s1 +
                    mse_coeff_s01 * s01;
        } else {
            mse_dot = TurboQuantQueryOps<SL2>::mse_dot(
                    quant, query.data(), code, inv_sqrt_d);
        }

        // Pre-screening
        if (threshold_ptr != nullptr) {
            n_total++;
            float bound = qjl_error_coeff * gamma * norm;
            float mse_ip = norm * mse_dot;

            if constexpr (Similarity::metric_type == METRIC_INNER_PRODUCT) {
                if (mse_ip + bound <= *threshold_ptr) {
                    n_skipped++;
                    return -std::numeric_limits<float>::infinity();
                }
            } else {
                float best_possible =
                        q_norm_sq + norm * norm - 2.0f * (mse_ip + bound);
                if (best_possible >= *threshold_ptr) {
                    n_skipped++;
                    return std::numeric_limits<float>::infinity();
                }
            }
        }

        // Stage 2: QJL dot product
        const uint8_t* qjl_code = code + quant.mse_total_bytes;
        float qjl_dot;
        if (qb > 0 && int_qjl) {
            size_t byte_size = (d + 7) / 8;
            uint64_t and_result =
                    rabitq::bitwise_and_dot_product<rabitq_popcount_level<SL2>>(
                            rearranged_qproj.data(), qjl_code, byte_size, qb);
            uint64_t pop = rabitq::popcount<rabitq_popcount_level<SL2>>(
                    qjl_code, byte_size);
            float pos_sum = qjl_popcnt_scale * static_cast<float>(pop) +
                    qjl_int_scale * static_cast<float>(and_result);
            qjl_dot = qjl_coeff * gamma * (2.0f * pos_sum - total_qproj_sum);
        } else {
            float pos_sum =
                    turboq_masked_sum<SL2>(query_proj.data(), qjl_code, d);
            qjl_dot = qjl_coeff * gamma * (2.0f * pos_sum - total_qproj_sum);
        }

        float estimated_ip = norm * (mse_dot + qjl_dot);

        if constexpr (Similarity::metric_type == METRIC_INNER_PRODUCT) {
            return estimated_ip;
        } else {
            return q_norm_sq + norm * norm - 2.0f * estimated_ip;
        }
    }

    float symmetric_dis(idx_t, idx_t) override {
        FAISS_THROW_MSG("Not implemented");
    }
};

} // namespace faiss::scalar_quantizer
