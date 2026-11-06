/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/sq-loongarch-lloyd-ops.h>
namespace faiss::scalar_quantizer {
namespace detail {
template <SIMDLevel SL, int Bits>
struct LoongArchLloydMax : QuantizerLloydMax<Bits, SIMDLevel::NONE> {
    using Base = QuantizerLloydMax<Bits, SIMDLevel::NONE>;
    using Ops = LoongArchLloydOps<SL>;
    using Vec = typename Ops::Vec;
    using Base::Base;
    Vec reconstruct_8_components(const uint8_t* code, size_t i) const {
        return Ops::gather(
                Ops::template unpack<Bits>(code + i * Bits / 8),
                this->centroids);
    }
    void encode_vector(const float* x, uint8_t* code) const final {
        size_t i = 0;
        for (; i + 8 <= this->d; i += 8)
            Ops::template pack<Bits>(
                    Ops::template select<Bits>(Vec(x + i), this->boundaries),
                    code + i * Bits / 8);
        for (; i < this->d; ++i)
            this->encode_index(this->select_index(x[i]), code, i);
    }
    void decode_vector(const uint8_t* code, float* x) const final {
        size_t i = 0;
        for (; i + 8 <= this->d; i += 8)
            reconstruct_8_components(code, i).storeu(x + i);
        for (; i < this->d; ++i)
            x[i] = this->reconstruct_component(code, i);
    }
};
template <SIMDLevel SL, int Bits>
struct LoongArchTurboQuant : QuantizerTurboQuantFull<Bits, SIMDLevel::NONE> {
    using Base = QuantizerTurboQuantFull<Bits, SIMDLevel::NONE>;
    using Base::Base;
    using Ops = LoongArchLloydOps<SL>;
    using Vec = typename Ops::Vec;
    using Base::boundaries;
    using Base::centroids;
    using Base::d;
    using Base::mse_plane_bytes;
    using Base::mse_total_bytes;
    using Base::padded_d;
    using Base::qjl_plane_bytes;
    static constexpr int MB = Bits - 1;
    void fwht_inplace(float* x, size_t n) const {
        // Small transforms use a zero-padded vector; its first n outputs are
        // FWHT(n).
        if (n < 8) {
            float tmp[8] = {};
            memcpy(tmp, x, n * sizeof(float));
            Vec v(tmp);
            for (int h = 1; h < int(n); h *= 2)
                v = Ops::butterfly(v, h);
            v.storeu(tmp);
            memcpy(x, tmp, n * sizeof(float));
            return;
        }
        for (size_t j = 0; j < n; j += 8) {
            Vec v(x + j);
            for (int h = 1; h < 8; h *= 2)
                v = Ops::butterfly(v, h);
            v.storeu(x + j);
        }
        for (size_t h = 8; h < n; h *= 2)
            for (size_t i = 0; i < n; i += 2 * h)
                for (size_t j = i; j < i + h; j += 8) {
                    Vec a(x + j), b(x + j + h);
                    (a + b).storeu(x + j);
                    (a - b).storeu(x + j + h);
                }
    }
    void project_forward(const float* x, float* out) const {
        if (this->qjl_type != 0) {
            this->rr_forward(x, out);
            return;
        }
        std::vector<float> buf(padded_d, 0.f);
        size_t j = 0;
        for (; j + 8 <= d; j += 8)
            (Vec(x + j) * Vec(this->fwht_signs.data() + j))
                    .storeu(buf.data() + j);
        for (; j < d; ++j)
            buf[j] = x[j] * this->fwht_signs[j];
        fwht_inplace(buf.data(), padded_d);
        memcpy(out, buf.data(), d * sizeof(float));
    }
    void project_inverse(float* x, float* out) const {
        if (this->qjl_type != 0) {
            this->rr_inverse(x, out);
            return;
        }
        fwht_inplace(x, padded_d);
        size_t j = 0;
        for (; j + 8 <= d; j += 8)
            (Vec(x + j) * Vec(this->fwht_signs.data() + j)).storeu(out + j);
        for (; j < d; ++j)
            out[j] = x[j] * this->fwht_signs[j];
    }
    Vec mse8(const uint8_t* code, size_t j) const {
        return Ops::gather(
                Ops::template load_planes<MB>(code + j / 8, mse_plane_bytes),
                centroids);
    }
    float mse_dot(const float* query, const uint8_t* code, float scale) const {
        Vec acc(0.f);
        size_t j = 0;
        for (; j + 8 <= d; j += 8)
            acc = acc + (Vec(query + j) * mse8(code, j)) * Vec(scale);
        float sum = horizontal_add(acc);
        for (; j < d; ++j)
            sum += query[j] * centroids[this->load_mse_index(code, j)] * scale;
        return sum;
    }
    void encode_vector(const float* x, uint8_t* code) const final {
        const float sd = std::sqrt(float(d)), inv = 1.f / sd;
        const float norm = std::max(std::sqrt(fvec_norm_L2sqr(x, d)), 1e-30f);
        std::vector<float> residual(padded_d, 0.f);
        size_t j = 0;
        for (; j + 8 <= d; j += 8) {
            Vec v = Ops::divide(Vec(x + j), Vec(norm));
            Vec idx = Ops::template select<MB>(v * Vec(sd), boundaries);
            Ops::template store_planes<MB>(idx, code + j / 8, mse_plane_bytes);
            (v - Ops::gather(idx, centroids) * Vec(inv))
                    .storeu(residual.data() + j);
        }
        for (; j < d; ++j) {
            float v = x[j] / norm;
            auto idx =
                    uint8_t(std::upper_bound(
                                    boundaries,
                                    boundaries + Base::kCentroidsCount - 1,
                                    v * sd) -
                            boundaries);
            this->store_mse_index(idx, code, j);
            residual[j] = v - centroids[idx] * inv;
        }
        std::vector<float> proj(d);
        project_forward(residual.data(), proj.data());
        j = 0;
        for (; j + 8 <= d; j += 8)
            code[mse_total_bytes + j / 8] |=
                    Ops::positive(Vec(proj.data() + j));
        for (; j < d; ++j)
            if (proj[j] > 0)
                code[mse_total_bytes + j / 8] |= 1 << (j % 8);
        SQTurboQFactors factors;
        factors.norm = norm;
        factors.gamma = std::sqrt(fvec_norm_L2sqr(residual.data(), d));
        memcpy(code + mse_total_bytes + qjl_plane_bytes,
               &factors,
               sizeof(factors));
    }
    void decode_vector(const uint8_t* code, float* x) const final {
        SQTurboQFactors factors;
        memcpy(&factors,
               code + mse_total_bytes + qjl_plane_bytes,
               sizeof(factors));
        float inv = 1.f / std::sqrt(float(d)),
              invpd = 1.f / std::sqrt(float(padded_d));
        std::vector<float> signs(padded_d, 0.f), reconstructed(d);
        size_t j = 0;
        for (; j + 8 <= d; j += 8)
            ((Ops::plane(code[mse_total_bytes + j / 8]) * Vec(2.f) - Vec(1.f)) *
             Vec(invpd))
                    .storeu(signs.data() + j);
        for (; j < d; ++j)
            signs[j] = (code[mse_total_bytes + j / 8] & (1 << (j % 8)))
                    ? invpd
                    : -invpd;
        project_inverse(signs.data(), reconstructed.data());
        float coeff = std::sqrt(M_PI / 2.f) / float(d) * factors.gamma;
        j = 0;
        for (; j + 8 <= d; j += 8)
            ((mse8(code, j) * Vec(inv) +
              Vec(coeff) * Vec(reconstructed.data() + j)) *
             Vec(factors.norm))
                    .storeu(x + j);
        for (; j < d; ++j)
            x[j] = (centroids[this->load_mse_index(code, j)] * inv +
                    coeff * reconstructed[j]) *
                    factors.norm;
    }
};
} // namespace detail
#if defined(__loongarch_sx) && !defined(__loongarch_asx)
template <int Bits>
struct QuantizerTurboQuantFull<Bits, SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchTurboQuant<SIMDLevel::LOONGARCH_LSX, Bits> {
    using detail::LoongArchTurboQuant<SIMDLevel::LOONGARCH_LSX, Bits>::
            LoongArchTurboQuant;
};
#endif
#if defined(__loongarch_asx)
template <int Bits>
struct QuantizerTurboQuantFull<Bits, SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchTurboQuant<SIMDLevel::LOONGARCH_LASX, Bits> {
    using detail::LoongArchTurboQuant<SIMDLevel::LOONGARCH_LASX, Bits>::
            LoongArchTurboQuant;
};
#endif
} // namespace faiss::scalar_quantizer
