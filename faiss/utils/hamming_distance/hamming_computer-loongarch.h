/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/utils/hamming_distance/hamming_computer.h>
#include <faiss/utils/simd_impl/loongarch_bits.h>
#include <cassert>
namespace faiss {
namespace detail {
template <typename Bits, bool generalized>
inline int loongarch_hamming_register(
        typename Bits::Register a,
        typename Bits::Register b) {
    if constexpr (generalized)
        return Bits::different_bytes(a, b);
    else
        return Bits::popcount(Bits::bit_xor(a, b));
}

template <SIMDLevel SL, bool generalized>
inline int loongarch_hamming(const uint8_t* a, const uint8_t* b, size_t size) {
    using Bits = LoongArchBits<SL>;
    int result = 0;
    size_t i = 0;
    for (; i + Bits::width <= size; i += Bits::width)
        result += loongarch_hamming_register<Bits, generalized>(
                Bits::load(a + i), Bits::load(b + i));
    if constexpr (SL == SIMDLevel::LOONGARCH_LASX) {
        using Narrow = LoongArchBits<SIMDLevel::LOONGARCH_LSX>;
        if (i + 16 <= size) {
            result += loongarch_hamming_register<Narrow, generalized>(
                    Narrow::load(a + i), Narrow::load(b + i));
            i += 16;
        }
    }
    for (; i < size; ++i) {
        if constexpr (generalized)
            result += a[i] != b[i];
        else
            result += popcount32(a[i] ^ b[i]);
    }
    return result;
}

template <SIMDLevel SL, int Size, bool generalized = false>
struct LoongArchHammingFixed {
    using Bits = LoongArchBits<SL>;
    static constexpr int full = Size / Bits::width;
    static constexpr int tail = Size % Bits::width;
    typename Bits::Register data[full + (tail != 0)];
    LoongArchHammingFixed() = default;
    LoongArchHammingFixed(const uint8_t* a, int size) {
        set(a, size);
    }
    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == Size);
        for (int i = 0; i < full; ++i)
            data[i] = Bits::load(a + i * Bits::width);
        if constexpr (tail)
            data[full] = Bits::load_partial(a + full * Bits::width, tail);
    }
    int hamming(const uint8_t* b) const {
        int result = 0;
        for (int i = 0; i < full; ++i)
            result += loongarch_hamming_register<Bits, generalized>(
                    data[i], Bits::load(b + i * Bits::width));
        if constexpr (tail)
            result += loongarch_hamming_register<Bits, generalized>(
                    data[full],
                    Bits::load_partial(b + full * Bits::width, tail));
        return result;
    }
    static constexpr int get_code_size() {
        return Size;
    }
};

template <SIMDLevel SL, bool generalized = false>
struct LoongArchHammingDefault {
    const uint8_t* a;
    int size;
    LoongArchHammingDefault() = default;
    LoongArchHammingDefault(const uint8_t* a, int size) {
        set(a, size);
    }
    void set(const uint8_t* a_in, int size_in) {
        a = a_in;
        size = size_in;
    }
    int hamming(const uint8_t* b) const {
        return loongarch_hamming<SL, generalized>(a, b, size);
    }
    int get_code_size() const {
        return size;
    }
};
} // namespace detail
#if defined(__loongarch_sx)
template <>
struct HammingComputer16_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, false>;
    using Base::Base;
};
template <>
struct HammingComputer20_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 20, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 20, false>;
    using Base::Base;
};
template <>
struct HammingComputer32_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 32, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 32, false>;
    using Base::Base;
};
template <>
struct HammingComputer64_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 64, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 64, false>;
    using Base::Base;
};
template <>
struct GenHammingComputer8_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 8, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 8, true>;
    using Base::Base;
};
template <>
struct GenHammingComputer16_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, true>;
    using Base::Base;
};
template <>
struct GenHammingComputer32_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 32, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 32, true>;
    using Base::Base;
};
template <>
struct HammingComputerDefault_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LSX, false> {
    using Base =
            detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LSX, false>;
    using Base::Base;
};
template <>
struct GenHammingComputerM8_tpl<SIMDLevel::LOONGARCH_LSX>
        : detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LSX, true> {
    using Base =
            detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LSX, true>;
    using Base::Base;
};
#endif
#if defined(__loongarch_asx)
template <>
struct HammingComputer16_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, false>;
    using Base::Base;
};
template <>
struct HammingComputer20_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 20, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 20, false>;
    using Base::Base;
};
template <>
struct HammingComputer32_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 32, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 32, false>;
    using Base::Base;
};
template <>
struct HammingComputer64_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 64, false> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 64, false>;
    using Base::Base;
};
template <>
struct GenHammingComputer8_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 8, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 8, true>;
    using Base::Base;
};
template <>
struct GenHammingComputer16_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LSX, 16, true>;
    using Base::Base;
};
template <>
struct GenHammingComputer32_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 32, true> {
    using Base =
            detail::LoongArchHammingFixed<SIMDLevel::LOONGARCH_LASX, 32, true>;
    using Base::Base;
};
template <>
struct HammingComputerDefault_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LASX, false> {
    using Base =
            detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LASX, false>;
    using Base::Base;
};
template <>
struct GenHammingComputerM8_tpl<SIMDLevel::LOONGARCH_LASX>
        : detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LASX, true> {
    using Base =
            detail::LoongArchHammingDefault<SIMDLevel::LOONGARCH_LASX, true>;
    using Base::Base;
};
#endif
} // namespace faiss
