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
#include "Synet/Gemm.h"

namespace Synet
{
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
            const ptrdiff_t bodySize = dstX - padX * 2;
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
                                if (bodySize > 0)
                                {
                                    memcpy(dst, psrc + sx, bodySize * sizeof(T));
                                    dst += bodySize;
                                    dx += bodySize;
                                    sx += bodySize;
                                }
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

    template <typename T> T CpuAbsSum(const T * src, size_t size)
    {
        T sum = 0;
        for (size_t i = 0; i < size; ++i)
            sum += ::abs(src[i]);
        return sum;
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
}