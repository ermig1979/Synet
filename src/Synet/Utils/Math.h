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

    template <typename T> void CpuMin(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = std::min(a[i], b[i]);
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

    template <typename T> void CpuSigmoid(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = T(1) / (T(1) + ::exp(-src[i]));
    }

    template <typename T> T CpuSigmoid(T value)
    {
        return T(1) / (T(1) + ::exp(-value));
    }

    template <typename T> SYNET_INLINE T CpuRelu(T value, T slope)
    {
        return std::max(value, T(0)) + slope * std::min(value, T(0));
    }

    template <typename T> void CpuRelu(const T * src, size_t size, const T & slope, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuRelu(src[i], slope);
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

    template <typename T> T CpuDotProduct(const T * a, const T * b, size_t size)
    {
        T sum = 0;
        for (size_t i = 0; i < size; ++i)
           sum += a[i]*b[i];
        return sum;
    }

    template<class T> void CpuRestrictRange(const T * src, size_t size, T lower, T upper, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = std::min(std::max(lower, src[i]), upper);
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
        ::SimdSynetAddBias(bias, count, size, dst, ::SimdFalse);
    }

    template <> SYNET_INLINE float CpuDotProduct<float>(const float * a, const float * b, size_t size)
    {
        float sum = 0;
        ::SimdNeuralProductSum(a, b, size, &sum);
        return sum;
    }

    template<> SYNET_INLINE void CpuRestrictRange<float>(const float * src, size_t size, float lower, float upper, float * dst)
    {
        ::SimdSynetRestrictRange(src, size, &lower, &upper, dst);
    }
#endif
}