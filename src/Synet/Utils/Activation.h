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

#include "Synet/Utils/Math.h"

namespace Synet
{
    template <typename T> SYNET_INLINE T CpuElu(T value, T alpha)
    {
        return value >= T(0) ? value : alpha * (exp(value) - T(1));
    }

    template <typename T> void CpuElu(const T * src, size_t size, const T & alpha, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuElu(src[i], alpha);
    }

    //-------------------------------------------------------------------------

    template<class T> SYNET_INLINE T CpuRestrictRange(T value, T lower, T upper)
    {
        return std::min(std::max(lower, value), upper);
    }

    template<class T> void CpuRestrictRange(const T * src, size_t size, T lower, T upper, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuRestrictRange(src[i], lower, upper);
    }

    //-------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuRelu(T value, T slope)
    {
        return std::max(value, T(0)) + slope * std::min(value, T(0));
    }

    template <typename T> void CpuRelu(const T * src, size_t size, const T & slope, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuRelu(src[i], slope);
    }    

    //-------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuSigmoid(T value)
    {
        return T(1) / (T(1) + ::exp(-value));
    }   

    template <typename T> void CpuSigmoid(const T * src, size_t size, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuSigmoid(src[i]);
    }
    
    //-------------------------------------------------------------------------

    template <typename T> SYNET_INLINE T CpuSoftplus(T value, T beta, T threshold = 50)
    {
        return value > threshold ? value : ::log(T(1) + ::exp(value * beta)) / beta;
    }

    template <typename T> void CpuSoftplus(const T * src, size_t size, const T & beta, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = CpuSoftplus(src[i], beta);
    }

    //-------------------------------------------------------------------------

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template <> SYNET_INLINE void CpuElu<float>(const float * src, size_t size, const float & alpha, float * dst)
    {
        ::SimdSynetElu32f(src, size, &alpha, dst);
    }    

    template <> SYNET_INLINE void CpuRelu<float>(const float * src, size_t size, const float & negativeSlope, float * dst)
    {
        ::SimdNeuralRelu(src, size, &negativeSlope, dst);
    }

    template<> SYNET_INLINE void CpuRestrictRange<float>(const float * src, size_t size, float lower, float upper, float * dst)
    {
        ::SimdSynetRestrictRange32f(src, size, &lower, &upper, dst);
    } 

    template <> SYNET_INLINE void CpuSigmoid<float>(const float * src, size_t size, float * dst)
    {
        float slope = 1.0f;
        ::SimdNeuralSigmoid(src, size, &slope, dst);
    }
#endif
}