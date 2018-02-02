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

#include "Synet/Math.h"

namespace Synet
{
    static void CpuGemmNN(size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float * C)
    {
        for (size_t i = 0; i < M; ++i) 
        {
            for (size_t k = 0; k < K; ++k) 
            {
                register float a = alpha * A[i*K + k];
                for (size_t j = 0; j < N; ++j) 
                    C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    static void CpuGemmNT(size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float * C)
    {
        for (size_t i = 0; i < M; ++i) 
        {
            for (size_t j = 0; j < N; ++j) 
            {
                register float sum = 0;
                for (size_t k = 0; k < K; ++k) 
                    sum += alpha * A[i*K + k] * B[j*K + k];
                C[i*N + j] += sum;
            }
        }
    }

    static void CpuGemmTN(size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float * C)
    {
        for (size_t i = 0; i < M; ++i) 
        {
            for (size_t k = 0; k < K; ++k) 
            {
                register float a = alpha * A[k*M + i];
                for (size_t j = 0; j < N; ++j) 
                    C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    static void CpuGemmTT(size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float * C)
    {
        for (size_t i = 0; i < M; ++i) 
        {
            for (size_t j = 0; j < N; ++j) 
            {
                register float sum = 0;
                for (size_t k = 0; k < K; ++k) 
                    sum += alpha * A[i + k * M] * B[k + j * K];
                C[i*N + j] += sum;
            }
        }
    }

    template <> void CpuGemm<float>(CblasTranspose transA, CblasTranspose transB,
        size_t M, size_t N, size_t K, float alpha, const float * A, const float * B, float beta, float * C)
    {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                C[i*N + j] *= beta;

        if (transA == CblasNoTrans && transB == CblasNoTrans)
            CpuGemmNN(M, N, K, alpha, A, B, C);
        if (transA == CblasTrans && transB == CblasNoTrans)
            CpuGemmTN(M, N, K, alpha, A, B, C);
        if (transA == CblasNoTrans && transB == CblasTrans)
            CpuGemmNT(M, N, K, alpha, A, B, C);
        if (transA == CblasTrans && transB == CblasTrans)
            CpuGemmTT(M, N, K, alpha, A, B, C);
    }

    template <> void CpuSet<float>(size_t size, float value, float * dst)
    {
        if (value == 0)
        {
            memset(dst, 0, size * sizeof(float));
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = value;
        }
    }

    template <> void ImToCol<float>(const float * src, size_t channels, size_t srcY, size_t srcX, size_t kernelY, size_t kernelX, 
        size_t padY, size_t padX, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, float * dst)
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
                                if (sx < dstX) 
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
                        sy += strideX;
                    }
                }
            }
            src += channelSize;
        }
    }

    template <> void CpuSqr<float>(const float * src, size_t size, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[i] * src[i];
    }

    template <> void CpuMul<float>(const float * a, const float * b, size_t size, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] * b[i];
    }

    template <> void CpuAxpy<float>(const float * src, size_t size, const float & alpha, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] += src[i]*alpha;
    }

    template <> void CpuPow<float>(const float * src, size_t size, const float & exp, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::pow(src[i], exp);
    }
}