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

#include "Synet/Utils/Math.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    SYNET_INLINE float CpuElu32f(float value, float alpha)
    {
        return value >= 0.0f ? value : alpha * (expf(value) - 1.0f);
    }

    void Elu32f(const float* src, size_t size, float alpha, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetElu32f(src, size, &alpha, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Elu32f(src[i], alpha);
#endif
    }

    //-------------------------------------------------------------------------------------------------

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void CpuElu<float>(const float * src, size_t size, float alpha, float * dst)
    {
        ::SimdSynetElu32f(src, size, &alpha, dst);
    }  

    template <> void CpuGelu<float>(const float* src, size_t size, float* dst)
    {
        ::SimdSynetGelu32f(src, size, dst);
    }

    template <> void CpuHardSigmoid(const float* src, size_t size, float scale, float shift, float* dst)
    {
        ::SimdSynetHardSigmoid32f(src, size, &scale, &shift, dst);
    }

    template <> void CpuHswish<float>(const float * src, size_t size, float shift, float scale, float * dst)
    {
        ::SimdSynetHswish32f(src, size, &shift, &scale, dst);
    }

    template<> void CpuMish<float>(const float* src, size_t size, float threshold, float* dst)
    {
        ::SimdSynetMish32f(src, size, &threshold, dst);
    }

    template <> void CpuRelu<float>(const float * src, size_t size, float negativeSlope, float * dst)
    {
        ::SimdSynetRelu32f(src, size, &negativeSlope, dst);
    }

    template<> void CpuRestrictRange<float>(const float * src, size_t size, float lower, float upper, float * dst)
    {
        ::SimdSynetRestrictRange32f(src, size, &lower, &upper, dst);
    } 

    template <> void CpuSigmoid<float>(const float * src, size_t size, float * dst)
    {
        float slope = 1.0f;
        ::SimdSynetSigmoid32f(src, size, &slope, dst);
    }

    template<> void CpuSoftplus<float>(const float* src, size_t size, float beta, float threshold, float* dst)
    {
        ::SimdSynetSoftplus32f(src, size, &beta, &threshold, dst);
    }

    template <> void CpuSwish<float>(const float * src, size_t size, float * dst)
    {
        float slope = 1.0f;
        ::SimdSynetSwish32f(src, size, &slope, dst);
    }
#endif
}