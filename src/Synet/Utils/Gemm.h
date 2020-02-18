/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
        template<class T> void CpuGemmNN(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T * C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    T a = alpha * A[i*lda + k];
                    for (size_t j = 0; j < N; ++j)
                        C[i*ldc + j] += a * B[k*ldb + j];
                }
            }
        }

        template<class T> void CpuGemmNT(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T * C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k)
                        sum += alpha * A[i*lda + k] * B[j*ldb + k];
                    C[i*ldc + j] += sum;
                }
            }
        }

        template<class T> void CpuGemmTN(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T * C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    T a = alpha * A[k*lda + i];
                    for (size_t j = 0; j < N; ++j)
                        C[i*ldc + j] += a * B[k*ldb + j];
                }
            }
        }

        template<class T> void CpuGemmTT(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T * C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k)
                        sum += alpha * A[i + k * lda] * B[k + j * ldb];
                    C[i*ldc + j] += sum;
                }
            }
        }

        template<class T> void CpuGemvN(size_t M, size_t N, T alpha, const T * A, const T * x, T * y)
        {
            for (size_t i = 0; i < M; ++i)
            {
                T sum = 0;
                for (size_t j = 0; j < N; ++j)
                    sum += x[j] * A[i*N + j];
                y[i] += alpha*sum;
            }
        }

        template<class T> void CpuGemvT(size_t M, size_t N, T alpha, const T * A, const T * x, T * y)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T ax = alpha*x[j];
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
        size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T beta, T * C, size_t ldc)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i*ldc + j] *= beta;

        if (transA == CblasNoTrans && transB == CblasNoTrans)
            Detail::CpuGemmNN(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        if (transA == CblasTrans && transB == CblasNoTrans)
            Detail::CpuGemmTN(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        if (transA == CblasNoTrans && transB == CblasTrans)
            Detail::CpuGemmNT(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        if (transA == CblasTrans && transB == CblasTrans)
            Detail::CpuGemmTT(M, N, K, alpha, A, lda, B, ldb, C, ldc);
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

    template < class TA, class TB> inline void CpuGemmNN(size_t M, size_t N, size_t K, const TA * A, size_t lda, const TB * B, size_t ldb, int32_t * C, size_t ldc)
    {
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
                C[i*ldc + j] = 0;
            for (size_t k = 0; k < K; ++k)
            {
                int32_t a = A[i*lda + k];
                for (size_t j = 0; j < N; ++j)
                    C[i*ldc + j] += a * B[k*ldb + j];
            }
        }
    }

    inline void CpuGemm8iNN(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t lda, const int8_t* weight, size_t ldb, int32_t* dst, size_t ldc, int neg)
    {
#ifdef SYNET_INT8_INT8_DISABLE 
        const size_t C2 = C / 2 * 2;
#else
        const size_t C2 = neg ? 0 : C / 2 * 2;
#endif
        for (size_t i = 0; i < S; ++i)
        {
            for (size_t j = 0; j < D; ++j)
                dst[i * ldc + j] = 0;
            for (size_t k = 0, o = 0; k < K; k++)
            {
                size_t c = 0;
                for (; c < C2; c += 2, o += 2)
                {
                    int32_t s0 = src[i * lda + o + 0];
                    int32_t s1 = src[i * lda + o + 1];
                    const int8_t* w0 = weight + (o + 0) * ldb;
                    const int8_t* w1 = weight + (o + 1) * ldb;
                    int32_t* d = dst + i * ldc;
                    for (size_t j = 0; j < D; ++j)
                    {
                        int sum = s0 * w0[j] + s1 * w1[j];
#if defined(SYNET_INT8_INT16_OWERFLOW)
                        sum = std::min(std::max(SHRT_MIN, sum), SHRT_MAX);
#endif
                        d[j] += sum;
                    }
                }
                for (; c < C; ++c, ++o)
                {
                    int32_t s0 = src[i * lda + o];
                    const int8_t* w0 = weight + o * ldb;
                    int32_t* d = dst + i * ldc;
                    if (neg)
                    {
                        for (size_t j = 0; j < D; ++j)
                        {
                            int _w0 = w0[j];
#ifdef SYNET_INT8_INT8_DISABLE
                            int _s0 = int(s0);
#else
                            if (_w0 & 1)
                                _w0 = Round(_w0 * 0.25f) * 4;
                            int _s0 = int(s0) - 128;
#endif
                            int dp = _w0 * _s0;
                            d[j] += dp;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < D; ++j)
                            d[j] += s0 * w0[j];
                    }
                }
            }
        }
    }

    inline void CpuGemm8iNN(size_t D, size_t S, size_t C, size_t K, const int8_t* weight, size_t lda, const uint8_t* src, size_t ldb, int32_t* dst, size_t ldc, int neg)
    {
#ifdef SYNET_INT8_INT8_DISABLE 
        const size_t C2 = C / 2 * 2;
#else
        const size_t C2 = neg ? 0 : C / 2 * 2;
#endif
        for (size_t i = 0; i < D; ++i)
        {
            for (size_t j = 0; j < S; ++j)
                dst[i * ldc + j] = 0;
            size_t c = 0;
            for (; c < C2; c += 2)
            {
                for (size_t k = 0; k < K; k++)
                {
                    int32_t w0 = weight[i * lda + (c + 0) * K + k];
                    int32_t w1 = weight[i * lda + (c + 1) * K + k];
                    const uint8_t* s0 = src + ((c + 0) * K + k) * ldb;
                    const uint8_t* s1 = src + ((c + 1) * K + k) * ldb;
                    int32_t* d = dst + i * ldc;
                    for (size_t j = 0; j < S; ++j)
                    {
                        int sum = s0[j] * w0 + s1[j] * w1;
#if defined(SYNET_INT8_INT16_OWERFLOW)
                        sum = std::min(std::max(SHRT_MIN, sum), SHRT_MAX);
#endif
                        d[j] += sum;
                    }
                }
            }
            for (; c < C; ++c)
            {
                for (size_t k = 0; k < K; k++)
                {
                    int32_t w0 = weight[i * lda + (c + 0) * K + k];
                    const uint8_t* s0 = src + ((c + 0) * K + k) * ldb;
                    int32_t* d = dst + i * ldc;
                    if (neg)
                    {
                        for (size_t j = 0; j < S; ++j)
                        {
                            int _w0 = w0;
#ifdef SYNET_INT8_INT8_DISABLE
                            int _s0 = int(s0[j]);
#else
                            if (_w0 & 1)
                                _w0 = Round(_w0 * 0.25f) * 4;
                            int _s0 = int(s0[j]) - 128;
#endif
                            int dp = _w0 * _s0;
                            d[j] += dp;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < S; ++j)
                            d[j] += s0[j] * w0;
                    }
                }
            }
        }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
    template <> SYNET_INLINE void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float beta, float * C, size_t ldc)
    {
        if (transA == CblasNoTrans && transB == CblasNoTrans)
        {
            ::SimdGemm32fNN(M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
        }
        else if (transA == CblasNoTrans && transB == CblasTrans)
        {
            ::SimdGemm32fNT(M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
        }
        else
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    C[i*ldc + j] *= beta;
            if (transA == CblasTrans && transB == CblasNoTrans)
                Detail::CpuGemmTN(M, N, K, alpha, A, lda, B, ldb, C, ldc);
            if (transA == CblasTrans && transB == CblasTrans)
                Detail::CpuGemmTT(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        }
    }
#endif

#if defined(SYNET_BLIS_ENABLE)
SYNET_INLINE void BlisGemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
{
#if defined(SYNET_SIZE_STATISTIC) && 0
    std::stringstream ss;
    ss << M << "-" << N << "-" << K;
    SYNET_PERF_BLOCK(ss.str().c_str());
#endif
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, M, N, K, (float*)alpha, (float*)A, lda, 1, (float*)B, ldb, 1, (float*)beta, C, ldc, 1);
}
#define SYNET_EXTERNAL_GEMM Synet::BlisGemm32fNN
#else
#define SYNET_EXTERNAL_GEMM NULL
#endif
}