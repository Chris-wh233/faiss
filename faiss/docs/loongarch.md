# LoongArch64 CPU support

Linux LoongArch64 supports `FAISS_OPT_LEVEL=generic` and runtime dispatch with
`FAISS_OPT_LEVEL=dd`. The DD build selects LASX, LSX or NONE from the compiled
implementations and Linux hardware capabilities. DD builds require a compiler
supporting both LSX and LASX intrinsics and compile both implementations. See
[INSTALL.md](../../INSTALL.md#building-on-linux-loongarch64) for dependencies
and build commands.

Common code and Python bindings use `-mno-lsx -mno-lasx`; native translation
units use `-mlsx -mno-lasx` or `-mlasx`. Avoid global SIMD flags in builds
intended to run on CPUs with different extension support. The 256-bit simdlib
abstraction uses two LSX registers or one LASX register; distance kernels use
the native four- or eight-float width.

## Coverage

Native kernels cover basic and extra distances, Flat distance computers,
IVFFlat scanners, normalization and distance conversions. BLAS remains an
external library. Faiss vectorizes the norm broadcasts, L2 conversion,
negative-distance clamp and top-1 reduction around BLAS calls, including
FP16 query tiles and database-parallel search. FP16 query conversion uses
the existing common implementation.

Native kernels also cover Hamming and binary scanners; SQ4/SQ6/SQ8,
uniform and direct SQ variants, FP16/BF16 codecs and IVF scanners; PQ/IVFPQ
distance tables, packed-code scans and SDC; PQ4 FastScan; and RaBitQ bitwise
operations, multi-bit inner products, query/LUT quantization and FastScan.
FP16 encoding follows the rounding used by Faiss's scalar codec.

Additional kernels cover EDEN/TQMSE/TurboQuant, quantizer preprocessing,
partitioning and approximate top-k, fused distances, HNSW heap selection,
and SuperKMeans block distances. Clustering, additive/residual/local-search
quantization, lattice search and bulk decoding use the corresponding SIMD
operations. Dispatch masks enable implemented kernel families.

Standalone PQ and two-level decoders, including MinMax/MinMaxFP16 wrappers,
are available through the installed `faiss/cppcontrib/SaDecodeKernels.h`.
These headers select their implementation at compile time: consumers use
`-mlsx -mno-lasx` or `-mlasx` on hardware supporting that extension.

EDEN and PQ retain scalar indexed table loads because LSX/LASX have no gather
instruction. Unpacking and arithmetic use SIMD. Lookup-heavy paths and wider
vectors need not improve every workload; compare identical indexes and data
with fixed thread counts when measuring performance.

SIMD reduction order and FMA can change floating-point rounding. BLAS search
clamps negative roundoff; the public inner-product-to-L2 conversion retains
negative results. Ordered comparisons preserve NaN and first-minimum rules.
Lp and JensenShannon use system `powf` and double `log` per lane, with SIMD
arithmetic around them. Scalar tails and irregular accesses are retained.

## Validation

Build and run `faiss_test` with `BUILD_TESTING=ON` for these configurations:

| Configuration | CMake settings |
|---|---|
| Scalar | `-DFAISS_OPT_LEVEL=generic` |
| Runtime dispatch | `-DFAISS_OPT_LEVEL=dd` |

On supported hardware, also check DD with `FAISS_SIMD_LEVEL=NONE`,
`LOONGARCH_LSX` or `LOONGARCH_LASX`. This override bypasses hardware detection,
so only force extensions supported by the CPU.

The Python test helper includes available LoongArch levels in parameterized
tests. Use the newly built Python package for these tests. Common library and
Python objects should contain no SIMD instructions, and LSX objects should
contain no LASX instructions. An installed CMake consumer checks exported
headers and dependencies.
