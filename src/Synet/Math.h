/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include "Synet/Common.h"

namespace Synet
{
    namespace Detail
    {
        template<class T> void CpuGemmNN(size_t M, size_t N, size_t K, T alpha, const T * A, const T * B, T * C)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    register T a = alpha * A[i*K + k];
                    for (size_t j = 0; j < N; ++j)
                        C[i*N + j] += a * B[k*N + j];
                }
            }
        }

        template<class T> void CpuGemmNT(size_t M, size_t N, size_t K, T alpha, const T * A, const T * B, T * C)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    register T sum = 0;
                    for (size_t k = 0; k < K; ++k)
                        sum += alpha * A[i*K + k] * B[j*K + k];
                    C[i*N + j] += sum;
                }
            }
        }

        template<class T> void CpuGemmTN(size_t M, size_t N, size_t K, float alpha, const T * A, const T * B, T * C)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    register T a = alpha * A[k*M + i];
                    for (size_t j = 0; j < N; ++j)
                        C[i*N + j] += a * B[k*N + j];
                }
            }
        }

        template<class T> void CpuGemmTT(size_t M, size_t N, size_t K, T alpha, const T * A, const T * B, T * C)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    register T sum = 0;
                    for (size_t k = 0; k < K; ++k)
                        sum += alpha * A[i + k * M] * B[k + j * K];
                    C[i*N + j] += sum;
                }
            }
        }

        template<class T> void CpuGemvN(size_t M, size_t N, T alpha, const T * A, const T * x, T * y)
        {
            for (size_t i = 0; i < M; ++i)
            {
                register T sum = 0;
                for (size_t j = 0; j < N; ++j)
                    sum += x[j] * A[i*N + j];
                y[i] += alpha*sum;
            }
        }

        template<class T> void CpuGemvT(size_t M, size_t N, T alpha, const T * A, const T * x, T * y)
        {
            for (size_t j = 0; j < N; ++j)
            {
                register T ax = alpha*x[j];
                for (size_t i = 0; i < M; ++i)
                    y[i] += ax * A[j*M + i];
            }
        }
    }

    enum CblasTranspose
    {
        CblasNoTrans = 111, 
        CblasTrans = 112, 
        CblasConjTrans = 113, 
        CblasConjNoTrans = 114,
    };

    template <typename T> void CpuGemm(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, T alpha, const T * A, const T * B, T beta, T * C)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i*N + j] *= beta;

        if (transA == CblasNoTrans && transB == CblasNoTrans)
            Detail::CpuGemmNN(M, N, K, alpha, A, B, C);
        if (transA == CblasTrans && transB == CblasNoTrans)
            Detail::CpuGemmTN(M, N, K, alpha, A, B, C);
        if (transA == CblasNoTrans && transB == CblasTrans)
            Detail::CpuGemmNT(M, N, K, alpha, A, B, C);
        if (transA == CblasTrans && transB == CblasTrans)
            Detail::CpuGemmTT(M, N, K, alpha, A, B, C);
    }

    template <typename T> void CpuGemv(CblasTranspose transA, size_t M, size_t N, T alpha, const T * A, const T * x, T beta, T * y)
    {
        for (size_t i = 0; i < M; ++i)
            y[i] *= beta;

        if (transA == CblasNoTrans)
            Detail::CpuGemvN(M, N, alpha, A, x, y);
        if (transA == CblasTrans)
            Detail::CpuGemvT(N, M, alpha, A, x, y);
    }

    template <typename T> void CpuSet(size_t size, T value, T * dst)
    {
        if (value == T(0))
        {
            memset(dst, 0, size * sizeof(T));
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = value;
        }
    }

    template <typename T> SYNET_INLINE void CpuCopy(const T * src, size_t size, T * dst)
    {
        ::memcpy(dst, src, size * sizeof(T));
    }

    template <typename T> void ImToCol(const T * src, size_t channels, size_t srcY, size_t srcX, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, T * dst)
    {
        SYNET_PERF_FUNC();

        size_t dstY = (srcY + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        size_t dstX = (srcX + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        size_t channelSize = srcX * srcY;
        if (dilationX == 1 && dilationY == 1 && strideX == 2 && strideY == 2 && padX == 0 && padY == 0 && kernelX == 1 && kernelY == 1)
        {
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t dy = 0; dy < dstY; ++dy)
                {
                    const T * psrc = src + 2*dy*srcX;
                    for (size_t dx = 0, sx = 0; dx < dstX; ++dx, sx += 2)
                        *(dst++) = psrc[sx];
                }
                src += channelSize;
            }
        }
        else if (dilationX*dilationY*strideX*strideY != 1)
        {
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ky++)
                {
                    for (size_t kx = 0; kx < kernelX; kx++)
                    {
                        size_t sy = ky * dilationY - padY;
                        for (size_t dy = 0; dy < dstY; ++dy)
                        {
                            if (sy < srcY)
                            {
                                size_t sx = kx * dilationX - padX;
                                for (size_t dx = 0; dx < dstX; ++dx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = src[sy * srcX + sx];
                                    else
                                        *(dst++) = 0;
                                    sx += strideX;
                                }
                            }
                            else
                            {
                                for (size_t dx = 0; dx < dstX; ++dx)
                                    *(dst++) = 0;
                            }
                            sy += strideY;
                        }
                    }
                }
                src += channelSize;
            }        
        }
        else
        {
            const size_t bodySize = dstX - padX * 2;
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < kernelX; ++kx)
                    {
                        size_t sy = ky - padY;
                        for (size_t dy = 0; dy < dstY; ++dy, ++sy)
                        {
                            if (sy < srcY)
                            {
                                size_t sx = kx - padX, dx = 0;
                                const T * psrc = src + sy*srcX;
                                for (; dx < padX; ++dx, ++sx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = psrc[sx];
                                    else
                                        *(dst++) = 0;
                                }
                                memcpy(dst, psrc + sx, bodySize * sizeof(T));
                                dst += bodySize;
                                dx += bodySize;
                                sx += bodySize;
                                for (; dx < dstX; ++dx, ++sx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = psrc[sx];
                                    else
                                        *(dst++) = 0;
                                }
                            }
                            else
                            {
                                memset(dst, 0, dstX * sizeof(T));
                                dst += dstX;
                            }
                        }
                    }
                }
                src += channelSize;
            }
        }
    }

    template <typename T> void CpuAdd(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] + b[i];
    }

    template <typename T> void CpuSub(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] - b[i];
    }

    template <typename T> void CpuMul(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] * b[i];
    }

    template <typename T> void CpuDiv(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] / b[i];
    }

    template <typename T> void CpuMax(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = std::max(a[i], b[i]);
    }

    template <typename T> void CpuSqr(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[i] * src[i];
    }

    template <typename T> void CpuAxpy(const T * x, size_t size, const T & alpha, T * y)
    {
        for (size_t i = 0; i < size; ++i)
            y[i] += x[i] * alpha;
    }

    template <typename T> void CpuAxpby(size_t size, const T & alpha, const T * x, const T & beta, T * y)
    {
        for (size_t i = 0; i < size; ++i)
            y[i] = alpha*x[i] + beta*y[i];
    }

    template <typename T> void CpuPow(const T * src, size_t size, const T & exp, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::pow(src[i], exp);
    }

    template <typename T> void CpuExp(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::exp(src[i]);
    }

    template <typename T> void CpuSigmoid(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = T(1) / (T(1) + ::exp(-src[i]));
    }

    template <typename T> T CpuSigmoid(T value)
    {
        return T(1) / (T(1) + ::exp(-value));
    }

    template <typename T> void CpuTanh(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::tanh(src[i]);
    }

    template <typename T> void CpuRelu(const T * src, size_t size, const T & negativeSlope, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = std::max(src[i], T(0)) + negativeSlope * std::min(src[i], T(0));
    }

    template <typename T> void CpuAdd(const T & value, T * dst, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] += value;
    }

    template <typename T> void CpuScale(const T * src, size_t size, const T & scale, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[i]*scale;
    }

    template <typename T> void CpuAddBias(const T * bias, size_t count, size_t size, T * dst)
    {
        for (size_t i = 0; i < count; ++i)
        {
            const T value = bias[i];
            for (size_t j = 0; j < size; ++j)
                dst[j] += value;
            dst += size;
        }
    }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template <> SYNET_INLINE void CpuAxpy<float>(const float * x, size_t size, const float & alpha, float * y)
    {
        ::SimdNeuralAddVectorMultipliedByValue(x, size, &alpha, y);
    }

    template <> SYNET_INLINE void CpuPow<float>(const float * src, size_t size, const float & exp, float * dst)
    {
        ::SimdNeuralPow(src, size, &exp, dst);
    }

    template <> SYNET_INLINE void CpuSigmoid<float>(const float * src, size_t size, float * dst)
    {
        float slope = 1.0f;
        ::SimdNeuralSigmoid(src, size, &slope, dst);
    }

    template <> SYNET_INLINE void CpuTanh<float>(const float * src, size_t size, float * dst)
    {
        float slope = 1.0f;
        ::SimdNeuralTanh(src, size, &slope, dst);
    }

    template <> SYNET_INLINE void CpuRelu<float>(const float * src, size_t size, const float & negativeSlope, float * dst)
    {
        ::SimdNeuralRelu(src, size, &negativeSlope, dst);
    }

    template <> SYNET_INLINE void CpuAdd<float>(const float & value, float * dst, size_t size)
    {
        ::SimdNeuralAddValue( &value, dst, size);
    }

    template <> SYNET_INLINE void CpuAddBias<float>(const float * bias, size_t count, size_t size, float * dst)
    {
        ::SimdSynetAddBias(bias, count, size, dst);
    }
#endif

    size_t GetThreadNumber()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        return ::SimdGetThreadNumber();
#elif defined(SYNET_OPEN_BLAS_ENABLE)
        return ::openblas_get_num_threads();
#else
        return 1;
#endif
    }

    void SetThreadNumber(size_t threadNumber)
    {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        ::SimdSetThreadNumber(threadNumber);
#endif
#ifdef SYNET_OPEN_BLAS_ENABLE
        ::openblas_set_num_threads((int)threadNumber);
        ::goto_set_num_threads((int)threadNumber);
#endif
    }

#if defined(SYNET_GEMM_COMPARE) && defined(SYNET_SIMD_LIBRARY_ENABLE) && defined(SYNET_OPEN_BLAS_ENABLE)
    inline void CpuGemmNN(int M, int N, int K, const float * A, const float * B, float * C)
    {
        const float alpha = 1.0f, beta = 0.0f;
        {
            std::stringstream ss;
            ss << M << "-" << N << "-" << K << " blas";
            SYNET_PERF_BLOCK(ss.str().c_str());
            ::cblas_sgemm(::CblasRowMajor, ::CblasNoTrans, ::CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
        }
        {
            std::stringstream ss;
            ss << M << "-" << N << "-" << K << " simd";
            SYNET_PERF_BLOCK(ss.str().c_str());
            ::SimdGemm32fNN(M, N, K, &alpha, A, K, B, N, &beta, C, N);
        }
    }

    template <> void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        assert(transA == CblasNoTrans && transB == CblasNoTrans && alpha == 1.0f && beta == 0.0f);
        CpuGemmNN((int)M, (int)N, (int)K, A, B, C);
    }
#elif defined(SYNET_GEMM_DYNAMIC) && defined(SYNET_SIMD_LIBRARY_ENABLE) && defined(SYNET_OPEN_BLAS_ENABLE)
    template <> void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        size_t threadNumber = GetThreadNumber();
        if (transA == CblasNoTrans && transB == CblasNoTrans && (threadNumber == 1 || N * M * K > threadNumber * 4 * 256 * 256 * 256))
        {
            ::SimdGemm32fNN(M, N, K, &alpha, A, K, B, N, &beta, C, N);
        }
        else
        {
            size_t lda = (transA == CblasNoTrans) ? K : M;
            size_t ldb = (transB == CblasNoTrans) ? N : K;
            ::cblas_sgemm(::CblasRowMajor, (::CBLAS_TRANSPOSE)transA, (::CBLAS_TRANSPOSE)transB,
                (int)M, (int)N, (int)K, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)N);
        }
    }
#elif defined(SYNET_GEMM_SIMD_LIBRARY) && defined(SYNET_SIMD_LIBRARY_ENABLE)
    template <> void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        if (transA == CblasNoTrans && transB == CblasNoTrans)
        {
            ::SimdGemm32fNN(M, N, K, &alpha, A, K, B, N, &beta, C, N);
        }
        else
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    C[i*N + j] *= beta;
            if (transA == CblasTrans && transB == CblasNoTrans)
                Detail::CpuGemmTN(M, N, K, alpha, A, B, C);
            if (transA == CblasNoTrans && transB == CblasTrans)
                Detail::CpuGemmNT(M, N, K, alpha, A, B, C);
            if (transA == CblasTrans && transB == CblasTrans)
                Detail::CpuGemmTT(M, N, K, alpha, A, B, C);        
        }
    }
#elif defined(SYNET_OPEN_BLAS_ENABLE)
    template <> SYNET_INLINE void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        size_t lda = (transA == CblasNoTrans) ? K : M;
        size_t ldb = (transB == CblasNoTrans) ? N : K;
        ::cblas_sgemm(::CblasRowMajor, (::CBLAS_TRANSPOSE)transA, (::CBLAS_TRANSPOSE)transB,
            (int)M, (int)N, (int)K, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)N);
    }
#endif

#ifdef SYNET_OPEN_BLAS_ENABLE
    template <> SYNET_INLINE void CpuGemv<float>(CblasTranspose transA, size_t M, size_t N, float alpha, const float * A, const float * x, float beta, float * y)
    {
        ::cblas_sgemv(::CblasRowMajor, (::CBLAS_TRANSPOSE)transA, (int)M, (int)N, alpha, A, (int)N, x, 1, beta, y, 1);
    }
#endif
}