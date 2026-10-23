# LoongArch64 CPU support

Linux LoongArch64 supports `FAISS_OPT_LEVEL=generic` and runtime dispatch with
`FAISS_OPT_LEVEL=dd`. The DD build selects LASX, LSX or NONE from the compiled
implementations and Linux hardware capabilities. CMake probes LSX and LASX
separately. See [INSTALL.md](../../INSTALL.md#building-on-linux-loongarch64)
for dependencies and build commands.

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

Binary distance and quantizer-specific kernels, selection and graph helpers,
and standalone decoding retain their existing implementations. Dispatch masks
enable LoongArch only for implemented kernel families.

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
| LSX only | `-DFAISS_OPT_LEVEL=dd -DFAISS_HAVE_LASX=OFF` |
| No SIMD implementations | `-DFAISS_OPT_LEVEL=dd -DFAISS_HAVE_LSX=OFF -DFAISS_HAVE_LASX=OFF` |

The `FAISS_HAVE_*` overrides disable compiled implementations; they do not
emulate a CPU lacking those extensions. On supported hardware, DD can also be
checked with `FAISS_SIMD_LEVEL=LOONGARCH_LSX` or `LOONGARCH_LASX`. This override
bypasses hardware detection, so only force extensions supported by the CPU.

The Python test helper includes available LoongArch levels in parameterized
tests. Use the newly built Python package for these tests. Common library and
Python objects should contain no SIMD instructions, and LSX objects should
contain no LASX instructions. An installed CMake consumer checks exported
headers and dependencies.
