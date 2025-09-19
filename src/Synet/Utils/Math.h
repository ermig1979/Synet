/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
    SYNET_INLINE size_t DivHi(size_t value, size_t divider)
    {
        return (value + divider - 1) / divider;
    }

    SYNET_INLINE int Round(float value)
    {
        return (int)(value + (value >= 0 ? 0.5f : -0.5f));
    }

    SYNET_INLINE int Quantize(float value)
    {
        return Round(value);
    }

    template <typename T> SYNET_INLINE T Square(T value)
    {
        return value*value;
    }

    template <class T> SYNET_INLINE T Min(T a, T b)
    {
        return a < b ? a : b;
    }

    template <class T> SYNET_INLINE T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <class T> SYNET_INLINE T Abs(T a)
    {
        return a < 0 ? -a : a;
    }

    template <class T> T Not(T i)
    {
        return ~i;
    }

    template<> SYNET_INLINE float Not<float>(float f)
    {
        int i = ~(int&)f;
        return (float&)i;
    }

    template <> SYNET_INLINE bool Not<bool>(bool i)
    {
        return !i;
    }

    template <class T> T And(T a, T b)
    {
        return a & b;
    }

    template<> SYNET_INLINE float And<float>(float a, float b)
    {
        int _a = (int&)a;
        int _b = (int&)b;
        int _c = _a & _b;
        return (float&)(_c);
    }

    template<> SYNET_INLINE bool And<bool>(bool a, bool b)
    {
        return a && b;
    }

    template <class T> SYNET_INLINE T RestrictRange(T value, T min, T max)
    {
        return Max(min, Min(max, value));
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
            dst[i] = Max(a[i], b[i]);
    }

    template <typename T> void CpuMin(const T * a, const T * b, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = Min(a[i], b[i]);
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

    template <typename T> void CpuExp(const T* src, size_t size, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::exp(src[i]);
    }

    template <typename T> void CpuLog(const T* src, size_t size, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::log(src[i]);
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

    template <typename T> void CpuAddBias(const T * bias, size_t count, size_t size, T * dst, int trans = 0)
    {
        if (trans)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] += bias[i];
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                const T value = bias[i];
                for (size_t j = 0; j < size; ++j)
                    dst[j] += value;
                dst += size;
            }
        }
    }

    template <typename T> T CpuAbsSum(const T * src, size_t size)
    {
        T sum = 0;
        for (size_t i = 0; i < size; ++i)
            sum += Abs(src[i]);
        return sum;
    }

    template <typename T> T CpuDotProduct(const T * a, const T * b, size_t size)
    {
        T sum = 0;
        for (size_t i = 0; i < size; ++i)
           sum += a[i]*b[i];
        return sum;
    }

    SYNET_INLINE float Fmadd(float a, float b, float c)
    {
        return float(double(a) * double(b) + double(c));
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> SYNET_INLINE void CpuSet<float>(size_t size, float value, float * dst)
    {
        ::SimdFill32f(dst, size, &value);
    }

    template <> SYNET_INLINE void CpuAxpy<float>(const float * x, size_t size, const float & alpha, float * y)
    {
        ::SimdNeuralAddVectorMultipliedByValue(x, size, &alpha, y);
    }

    template <> SYNET_INLINE void CpuPow<float>(const float * src, size_t size, const float & exp, float * dst)
    {
        ::SimdNeuralPow(src, size, &exp, dst);
    }

    template <> SYNET_INLINE void CpuAdd<float>(const float & value, float * dst, size_t size)
    {
        ::SimdNeuralAddValue( &value, dst, size);
    }

    template <> SYNET_INLINE void CpuAddBias<float>(const float * bias, size_t count, size_t size, float * dst, int trans)
    {
        ::SimdSynetAddBias(bias, count, size, dst, (::SimdTensorFormatType)trans);
    }

    template <> SYNET_INLINE float CpuDotProduct<float>(const float * a, const float * b, size_t size)
    {
        float sum = 0;
        ::SimdNeuralProductSum(a, b, size, &sum);
        return sum;
    }
#endif
}