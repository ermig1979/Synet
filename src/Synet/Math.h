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
        size_t dstY = (srcY + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        size_t dstX = (srcX + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        size_t channelSize = srcX * srcY;
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

    template <class T> void PoolingMax(const T * src, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t strideY, size_t strideX, T * dst, size_t dstX, size_t dstY)
    {
        for (size_t dy = 0; dy < dstY; ++dy)
        {
            size_t yStart = dy * strideY - padY;
            size_t yEnd = std::min(yStart + kernelY, srcY);
            yStart = std::max<ptrdiff_t>(0, yStart);
            for (size_t dx = 0; dx < dstX; ++dx)
            {
                size_t xStart = dx * strideX - padX;
                size_t xEnd = std::min(xStart + kernelX, srcX);
                xStart = std::max<ptrdiff_t>(0, xStart);
                T max = -std::numeric_limits<T>::max();
                for (size_t sy = yStart; sy < yEnd; ++sy)
                    for (size_t sx = xStart; sx < xEnd; ++sx)
                        max = std::max(max, src[sy * srcX + sx]);
                dst[dy*dstX + dx] = max;
            }
        }
    }

    template <typename T> void CpuMul(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] * b[i];
    }

    template <typename T> void CpuSqr(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[i] * src[i];
    }

    template <typename T> void CpuAxpy(const T * src, size_t size, const T & alpha, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] += src[i] * alpha;
    }

    template <typename T> void CpuPow(const T * src, size_t size, const T & exp, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::pow(src[i], exp);
    }

    template <typename T> void CpuSigmoid(const T * src, size_t size, const T & slope, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = T(1) / (T(1) + ::exp(-src[i] * slope));
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

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template <> SYNET_INLINE void PoolingMax<float>(const float * src, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t strideY, size_t strideX, float * dst, size_t dstX, size_t dstY)
    {
        if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && padY == 1 && padX == 1)
        {
            ::SimdNeuralPooling1x1Max3x3(src, srcX, srcX, srcY, dst, dstX);
            return;
        }
        if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0)
        {
            ::SimdNeuralPooling2x2Max3x3(src, srcX, srcX, srcY, dst, dstX);
            return;
        }
        for (size_t dy = 0; dy < dstY; ++dy)
        {
            size_t yStart = dy * strideY - padY;
            size_t yEnd = std::min(yStart + kernelY, srcY);
            yStart = std::max<ptrdiff_t>(0, yStart);
            for (size_t dx = 0; dx < dstX; ++dx)
            {
                size_t xStart = dx * strideX - padX;
                size_t xEnd = std::min(xStart + kernelX, srcX);
                xStart = std::max<ptrdiff_t>(0, xStart);
                float max = -std::numeric_limits<float>::max();
                for (size_t sy = yStart; sy < yEnd; ++sy)
                    for (size_t sx = xStart; sx < xEnd; ++sx)
                        max = std::max(max, src[sy * srcX + sx]);
                dst[dy*dstX + dx] = max;
            }
        }
    }

    template <> SYNET_INLINE void CpuAxpy<float>(const float * src, size_t size, const float & alpha, float * dst)
    {
        ::SimdNeuralAddVectorMultipliedByValue(src, size, &alpha, dst);
    }

    template <> SYNET_INLINE void CpuPow<float>(const float * src, size_t size, const float & exp, float * dst)
    {
        ::SimdNeuralPow(src, size, &exp, dst);
    }

    template <> SYNET_INLINE void CpuSigmoid<float>(const float * src, size_t size, const float & slope, float * dst)
    {
        ::SimdNeuralSigmoid(src, size, &slope, dst);
    }

    template <> SYNET_INLINE void CpuRelu<float>(const float * src, size_t size, const float & negativeSlope, float * dst)
    {
        ::SimdNeuralRelu(src, size, &negativeSlope, dst);
    }

    template <> SYNET_INLINE void CpuAdd<float>(const float & value, float * dst, size_t size)
    {
        ::SimdNeuralAddValue( &value, dst, size);
    }
#endif

#ifdef SYNET_OPEN_BLAS_ENABLE
    template <> void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB, 
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        size_t lda = (transA == CblasNoTrans) ? K : M;
        size_t ldb = (transB == CblasNoTrans) ? N : K;
        ::cblas_sgemm(::CblasRowMajor, (::CBLAS_TRANSPOSE)transA, (::CBLAS_TRANSPOSE)transB, 
            (int)M, (int)N, (int)K, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)N);
    }
#endif
}