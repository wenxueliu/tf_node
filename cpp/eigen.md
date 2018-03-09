


http://eigen.tuxfamily.org/index.php?title=Main_Page



If you just want to use Eigen, you can use the header files right away. There is no binary library to link to, and no configured header file. Eigen is a pure template library defined in the headers.



Eigen is versatile.
    It supports all matrix sizes, from small fixed-size matrices to arbitrarily large dense matrices, and even sparse matrices.
    It supports all standard numeric types, including std::complex, integers, and is easily extensible to custom numeric types.
    It supports various matrix decompositions and geometry features.
    Its ecosystem of unsupported modules provides many specialized features such as non-linear optimization, matrix functions, a polynomial solver, FFT, and much more.

Eigen is fast.
    Expression templates allow to intelligently remove temporaries and enable lazy evaluation, when that is appropriate.
    Explicit vectorization is performed for SSE 2/3/4, AVX, FMA, AVX512, ARM NEON (32-bit and 64-bit), PowerPC AltiVec/VSX (32-bit and 64-bit) instruction sets, and now S390x SIMD (ZVector) with graceful fallback to non-vectorized code.
    Fixed-size matrices are fully optimized: dynamic memory allocation is avoided, and the loops are unrolled when that makes sense.
    For large matrices, special attention is paid to cache-friendliness.

Eigen is reliable.
    Algorithms are carefully selected for reliability. Reliability trade-offs are clearly documented and extremely safe decompositions are available.
    Eigen is thoroughly tested through its own test suite (over 500 executables), the standard BLAS test suite, and parts of the LAPACK test suite.

Eigen is elegant.
    The API is extremely clean and expressive while feeling natural to C++ programmers, thanks to expression templates.
    Implementing an algorithm on top of Eigen feels like just copying pseudocode.
    Eigen has good compiler support as we run our test suite against many compilers to guarantee reliability and work around any compiler bugs. Eigen also is standard C++98 and maintains very reasonable compilation times.