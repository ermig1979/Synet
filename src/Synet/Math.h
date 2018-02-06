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
    enum CblasTranspose
    {
        CblasNoTrans = 111, 
        CblasTrans = 112, 
        CblasConjTrans = 113, 
        CblasConjNoTrans = 114,
    };

    template <typename T> void CpuGemm(CblasTranspose transA, CblasTranspose transB, 
        size_t M, size_t N, size_t K, T alpha, const T * A, const T * B, T beta, T * C);

    template <typename T> void CpuSet(size_t size, T value, T * dst);

    template <typename T> SYNET_INLINE void CpuCopy(const T * src, size_t size, T * dst)
    {
        ::memcpy(dst, src, size * sizeof(T));
    }

    template <typename T> void ImToCol(const T * src, size_t channels, size_t srcY, size_t srcX, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, T * dst);

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
}