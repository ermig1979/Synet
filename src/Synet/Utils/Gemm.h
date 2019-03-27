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
        template<class T> void CpuGemmNN(size_t M, size_t N, size_t K, T alpha, const T * A, size_t lda, const T * B, size_t ldb, T * C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    register T a = alpha * A[i*lda + k];
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
                    register T sum = 0;
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
                    register T a = alpha * A[k*lda + i];
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
                    register T sum = 0;
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
#endif
}