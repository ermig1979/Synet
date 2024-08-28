/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Utils/Math.h"

namespace Synet
{
    void Elu32f(const float* src, size_t size, float alpha, float* dst);

    void Gelu32f(const float* src, size_t size, float* dst);

    void HardSigmoid32f(const float* src, size_t size, float scale, float shift, float* dst);

    void Hswish32f(const float* src, size_t size, float shift, float scale, float* dst);

    void Mish32f(const float* src, size_t size, float threshold, float* dst);

    void Relu32f(const float* src, size_t size, float slope, float* dst);

    void RestrictRange32f(const float* src, size_t size, float lower, float upper, float* dst);

    void Sigmoid32f(const float* src, size_t size, float* dst);

    void Softplus32f(const float* src, size_t size, float beta, float threshold, float* dst);

    void Swish32f(const float* src, size_t size, float* dst);

    //-------------------------------------------------------------------------------------------------

    void Relu16b(const uint16_t* src, size_t size, float slope, uint16_t* dst);

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuElu(T value, T alpha)
    {
        return value >= T(0) ? value : alpha * (exp(value) - T(1));
    }

    template <typename T> void CpuElu(const T * src, size_t size, T alpha, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuElu(src[i], alpha);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuGelu(T value)
    {
        return value * (::erf(value * T(M_SQRT1_2)) + T(1)) * T(0.5);
    }

    template <typename T> void CpuGelu(const T* src, size_t size, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuGelu(src[i]);
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T CpuHardSigmoid(T value, T scale, T shift)
    {
        return Max(T(0), Min(value * scale + shift, T(1)));
    }

    template <class T> void CpuHardSigmoid(const T* src, size_t size, T scale, T shift, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuHardSigmoid(src[i], scale, shift);
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T CpuHswish(T value, T shift, T scale)
    {
        return Max(Min(value, shift) + shift, T(0))*scale*value;
    }

    template <class T> void CpuHswish(const T * src, size_t size, T shift, T scale, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuHswish(src[i], shift, scale);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuMish(T value, T threshold)
    {
        return value > threshold ? value : value * (T(1) - T(2) / (Square(::exp(value) + T(1)) + T(1)));
    }

    template <typename T> void CpuMish(const T* src, size_t size, T threshold, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuMish(src[i], threshold);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuRelu(T value, T slope)
    {
        return Max(value, T(0)) + slope * Min(value, T(0));
    }

    template <typename T> void CpuRelu(const T * src, size_t size, T slope, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuRelu(src[i], slope);
    }      
    
    //-------------------------------------------------------------------------------------------------

    template<class T> SYNET_INLINE T CpuRestrictRange(T value, T lower, T upper)
    {
        return Min(Max(lower, value), upper);
    }

    template<class T> void CpuRestrictRange(const T * src, size_t size, T lower, T upper, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuRestrictRange(src[i], lower, upper);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuSigmoid(T value)
    {
        return T(1) / (T(1) + ::exp(-value));
    }   

    template <typename T> void CpuSigmoid(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuSigmoid(src[i]);
    }
    
    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuSoftplus(T value, T beta, T threshold)
    {
        return value > threshold ? value : ::log(T(1) + ::exp(value * beta)) / beta;
    }

    template <typename T> void CpuSoftplus(const T * src, size_t size, T beta, T threshold, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuSoftplus(src[i], beta, threshold);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuSwish(T value)
    {
        return value / (T(1) + ::exp(-value));
    }

    template <typename T> void CpuSwish(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuSwish(src[i]);
    }


    //-------------------------------------------------------------------------------------------------

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> SYNET_INLINE void CpuElu<float>(const float* src, size_t size, float alpha, float* dst)
    {
        ::SimdSynetElu32f(src, size, &alpha, dst);
    }

    template <> SYNET_INLINE void CpuGelu<float>(const float* src, size_t size, float* dst)
    {
        ::SimdSynetGelu32f(src, size, dst);
    }

    template <> SYNET_INLINE void CpuHardSigmoid(const float* src, size_t size, float scale, float shift, float* dst)
    {
        ::SimdSynetHardSigmoid32f(src, size, &scale, &shift, dst);
    }

    template <> SYNET_INLINE void CpuHswish<float>(const float* src, size_t size, float shift, float scale, float* dst)
    {
        ::SimdSynetHswish32f(src, size, &shift, &scale, dst);
    }

    template<> SYNET_INLINE void CpuMish<float>(const float* src, size_t size, float threshold, float* dst)
    {
        ::SimdSynetMish32f(src, size, &threshold, dst);
    }

    template <> SYNET_INLINE void CpuRelu<float>(const float* src, size_t size, float negativeSlope, float* dst)
    {
        ::SimdSynetRelu32f(src, size, &negativeSlope, dst);
    }

    template<> SYNET_INLINE void CpuRestrictRange<float>(const float* src, size_t size, float lower, float upper, float* dst)
    {
        ::SimdSynetRestrictRange32f(src, size, &lower, &upper, dst);
    }

    template <> SYNET_INLINE void CpuSigmoid<float>(const float* src, size_t size, float* dst)
    {
        float slope = 1.0f;
        ::SimdSynetSigmoid32f(src, size, &slope, dst);
    }

    template<> SYNET_INLINE void CpuSoftplus<float>(const float* src, size_t size, float beta, float threshold, float* dst)
    {
        ::SimdSynetSoftplus32f(src, size, &beta, &threshold, dst);
    }

    template <> SYNET_INLINE void CpuSwish<float>(const float* src, size_t size, float* dst)
    {
        float slope = 1.0f;
        ::SimdSynetSwish32f(src, size, &slope, dst);
    }
#endif
}