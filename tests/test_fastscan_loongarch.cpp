/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/AlignedTable.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
namespace {
using namespace faiss;
void check_fastscan(SIMDLevel level) {
    if (!SIMDConfig::is_simd_level_available(level))
        GTEST_SKIP();
    struct Guard {
        SIMDLevel saved = SIMDConfig::get_level();
        ~Guard() {
            SIMDConfig::set_level(saved);
        }
    } guard;
    std::mt19937 rng(333);
    for (int nq : {1, 2, 3, 4})
        for (int nsq : {2, 6, 16, 64, 256})
            for (size_t nb : {32, 64, 96}) {
                std::vector<uint8_t> codes(nb * nsq / 2), lut(nq * nsq * 16);
                for (auto& v : codes)
                    v = rng();
                for (auto& v : lut)
                    v = nsq == 256 ? 255 : rng();
                AlignedTable<uint8_t> packed(codes.size()), plut(lut.size());
                pq4_pack_codes(
                        codes.data(), nb, nsq, nb, 32, nsq, packed.get());
                pq4_pack_LUT(nq, nsq, lut.data(), plut.get());
                std::vector<uint16_t> actual(nq * nb);
                SIMDConfig::set_level(level);
                accumulate_to_mem(
                        nq, nb, nsq, packed.get(), plut.get(), actual.data());
                for (int q = 0; q < nq; ++q)
                    for (size_t j = 0; j < nb; ++j) {
                        unsigned sum = 0;
                        for (int m = 0; m < nsq; ++m) {
                            unsigned c = (codes[j * nsq / 2 + m / 2] >>
                                          ((m % 2) * 4)) &
                                    15;
                            sum += lut[(q * nsq + m) * 16 + c];
                        }
                        EXPECT_EQ(actual[q * nb + j], sum);
                    }
            }
    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT})
        for (int bbs : {32, 64, 96}) {
            constexpr int M = 9, d = 18, nb = 77, nq = 17;
            IndexPQ orig(d, M, 4, metric);
            orig.is_trained = true;
            for (auto& v : orig.pq.centroids)
                v = float(rng() % 10000) / 10000;
            std::vector<float> xb(nb * d), xq(nq * d);
            for (auto& v : xb)
                v = float(rng() % 10000) / 10000;
            for (auto& v : xq)
                v = float(rng() % 10000) / 10000;
            SIMDConfig::set_level(SIMDLevel::NONE);
            orig.add(nb, xb.data());
            IndexPQFastScan flat(orig, bbs);
            IndexFlat coarse(d, metric);
            std::vector<float> center(d, 0);
            coarse.add(1, center.data());
            IndexIVFPQ orig_ivf(&coarse, d, 1, M, 4, metric);
            orig_ivf.by_residual = false;
            orig_ivf.pq.centroids = orig.pq.centroids;
            orig_ivf.is_trained = true;
            orig_ivf.add(nb, xb.data());
            IndexIVFPQFastScan ivf(orig_ivf, bbs);
            IDSelectorRange sel(2, nb - 3);
            SearchParametersIVF ivfparams;
            ivfparams.sel = &sel;
            for (int impl : {12, 13, 14, 15}) {
                if (bbs != 32 && impl < 14)
                    continue;
                flat.implem = impl;
                ivf.implem = impl >= 14 ? impl - 4 : impl;
                for (int k : {1, 7, 50})
                    for (bool use_ivf : {false, true}) {
                        SCOPED_TRACE(int(level));
                        SCOPED_TRACE(bbs);
                        SCOPED_TRACE(impl);
                        SCOPED_TRACE(k);
                        SCOPED_TRACE(use_ivf);
                        Index* idx = use_ivf ? static_cast<Index*>(&ivf)
                                             : static_cast<Index*>(&flat);
                        const SearchParameters* parameters =
                                use_ivf ? &ivfparams : nullptr;
                        std::vector<float> expected(nq * k), actual(nq * k);
                        std::vector<idx_t> ei(nq * k), ai(nq * k);
                        SIMDConfig::set_level(SIMDLevel::NONE);
                        idx->search(
                                nq,
                                xq.data(),
                                k,
                                expected.data(),
                                ei.data(),
                                parameters);
                        SIMDConfig::set_level(level);
                        idx->search(
                                nq,
                                xq.data(),
                                k,
                                actual.data(),
                                ai.data(),
                                parameters);
                        for (size_t i = 0; i < actual.size(); ++i) {
                            EXPECT_NEAR(
                                    actual[i],
                                    expected[i],
                                    std::abs(expected[i]) * 3e-5f + 1e-5f);
                            EXPECT_EQ(ai[i], ei[i]);
                        }
                    }
            }
            // Range handlers share the accumulation kernels but maintain
            // separate result buffers and apply the ID selector before
            // appending matches.
            for (int impl : {10, 12}) {
                if (bbs != 32 && impl == 12)
                    continue;
                ivf.implem = impl;
                Index* idx = &ivf;
                const SearchParameters* parameters = &ivfparams;
                RangeSearchResult expected(nq), actual(nq);
                float radius = metric == METRIC_L2 ? 3.0f : 4.0f;
                SIMDConfig::set_level(SIMDLevel::NONE);
                idx->range_search(nq, xq.data(), radius, &expected, parameters);
                SIMDConfig::set_level(level);
                idx->range_search(nq, xq.data(), radius, &actual, parameters);
                for (int q = 0; q < nq; ++q) {
                    ASSERT_EQ(
                            expected.lims[q + 1] - expected.lims[q],
                            actual.lims[q + 1] - actual.lims[q]);
                    std::vector<std::pair<idx_t, float>> e, a;
                    for (size_t i = expected.lims[q]; i < expected.lims[q + 1];
                         ++i)
                        e.emplace_back(
                                expected.labels[i], expected.distances[i]);
                    for (size_t i = actual.lims[q]; i < actual.lims[q + 1]; ++i)
                        a.emplace_back(actual.labels[i], actual.distances[i]);
                    std::sort(e.begin(), e.end());
                    std::sort(a.begin(), a.end());
                    for (size_t i = 0; i < e.size(); ++i) {
                        EXPECT_EQ(a[i].first, e[i].first);
                        EXPECT_NEAR(a[i].second, e[i].second, 2e-4f);
                    }
                }
            }
        }
}
} // namespace
#ifdef COMPILE_SIMD_LOONGARCH_LSX
TEST(TestLoongArchFastScan, LSX) {
    check_fastscan(SIMDLevel::LOONGARCH_LSX);
}
#endif
#ifdef COMPILE_SIMD_LOONGARCH_LASX
TEST(TestLoongArchFastScan, LASX) {
    check_fastscan(SIMDLevel::LOONGARCH_LASX);
}
#endif
