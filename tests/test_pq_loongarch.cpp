/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>
#include <faiss/utils/distances.h>
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace faiss;
template <SIMDLevel SL>
void check_pq() {
    if (!SIMDConfig::is_simd_level_available(SL))
        GTEST_SKIP();
    struct Guard {
        SIMDLevel saved = SIMDConfig::get_level();
        ~Guard() {
            SIMDConfig::set_level(saved);
        }
    } guard;
    std::mt19937 rng(144);
    for (size_t M : {1, 7, 8, 9, 15, 16, 17, 33, 129}) {
        std::vector<float> table(M * 256);
        std::vector<uint8_t> codes(4 * M + 1);
        for (auto& v : table)
            v = float(int(rng() % 2001) - 1000) / 1000;
        for (auto& v : codes)
            v = rng();
        float actual[4];
        pq_code_distance::pq_code_distance_8bit_four_impl<SL>(
                M,
                table.data(),
                codes.data() + 1,
                codes.data() + 1 + M,
                codes.data() + 1 + 2 * M,
                codes.data() + 1 + 3 * M,
                actual[0],
                actual[1],
                actual[2],
                actual[3]);
        for (int j = 0; j < 4; ++j) {
            double expected = 0;
            for (size_t m = 0; m < M; ++m)
                expected += table[m * 256 + codes[1 + j * M + m]];
            EXPECT_NEAR(actual[j], expected, 2e-5f);
            EXPECT_NEAR(
                    pq_code_distance::pq_code_distance_8bit_single_impl<SL>(
                            M, table.data(), codes.data() + 1 + j * M),
                    expected,
                    2e-5f);
        }
    }
    for (size_t M : {1, 3, 4, 5, 8, 9})
        for (size_t ksub : {8, 16, 64, 256})
            for (bool ip : {false, true}) {
                constexpr size_t nq = 3;
                std::vector<float> centroids(M * ksub * 2), x(nq * M * 2),
                        out(nq * M * ksub + 2, 123);
                for (auto& v : centroids)
                    v = float(int(rng() % 2001) - 1000) / 1000;
                for (auto& v : x)
                    v = float(int(rng() % 2001) - 1000) / 1000;
                compute_PQ_dis_tables_dsub2<SL>(
                        M * 2,
                        ksub,
                        centroids.data(),
                        nq,
                        x.data(),
                        ip,
                        out.data() + 1);
                EXPECT_EQ(out.front(), 123);
                EXPECT_EQ(out.back(), 123);
                for (size_t q = 0; q < nq; ++q)
                    for (size_t m = 0; m < M; ++m)
                        for (size_t k = 0; k < ksub; ++k) {
                            float expected = 0;
                            for (size_t j = 0; j < 2; ++j) {
                                float a = x[(q * M + m) * 2 + j],
                                      b = centroids[(m * ksub + k) * 2 + j];
                                expected += ip ? a * b : (a - b) * (a - b);
                            }
                            EXPECT_NEAR(
                                    out[1 + (q * M + m) * ksub + k],
                                    expected,
                                    1e-6f);
                        }
            }
    for (size_t nbits : {1, 2, 4, 6, 8, 16})
        for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
            SCOPED_TRACE(nbits);
            SCOPED_TRACE(metric);
            constexpr int M = 9, d = 18, nb = 73, nq = 3, k = 5;
            IndexPQ index(d, M, nbits, metric);
            for (auto& v : index.pq.centroids)
                v = float(rng() % 10000) / 10000;
            index.is_trained = true;
            std::vector<float> xb(nb * d), xq(nq * d);
            for (auto& v : xb)
                v = float(rng() % 10000) / 10000;
            for (auto& v : xq)
                v = float(rng() % 10000) / 10000;
            SIMDConfig::set_level(SL);
            index.add(nb, xb.data());
            IndexFlat coarse(d, metric);
            std::vector<float> center(d, 0);
            coarse.add(1, center.data());
            IndexIVFPQ ivf(&coarse, d, 1, M, nbits, metric);
            ivf.by_residual = false;
            ivf.is_trained = true;
            ivf.pq.centroids = index.pq.centroids;
            ivf.add(nb, xb.data());
            for (Index* idx :
                 {static_cast<Index*>(&index), static_cast<Index*>(&ivf)}) {
                std::vector<float> expected(nq * k), actual(nq * k);
                std::vector<idx_t> ei(nq * k), ai(nq * k);
                SIMDConfig::set_level(SIMDLevel::NONE);
                idx->search(nq, xq.data(), k, expected.data(), ei.data());
                SIMDConfig::set_level(SL);
                idx->search(nq, xq.data(), k, actual.data(), ai.data());
                for (size_t i = 0; i < actual.size(); ++i)
                    EXPECT_NEAR(
                            actual[i],
                            expected[i],
                            std::abs(expected[i]) * 2e-5f + 1e-6f);
            }
            SIMDConfig::set_level(SIMDLevel::NONE);
            std::unique_ptr<FlatCodesDistanceComputer> ref(
                    index.get_FlatCodesDistanceComputer());
            SIMDConfig::set_level(SL);
            std::unique_ptr<FlatCodesDistanceComputer> simd(
                    index.get_FlatCodesDistanceComputer());
            for (auto dc : {ref.get(), simd.get()})
                dc->set_query(xq.data());
            for (int i = 0; i < nb; ++i)
                EXPECT_NEAR((*ref)(i), (*simd)(i), 2e-5f);
        }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchPQ, LSX) {
    check_pq<SIMDLevel::LOONGARCH_LSX>();
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchPQ, LASX) {
    check_pq<SIMDLevel::LOONGARCH_LASX>();
}
#endif
