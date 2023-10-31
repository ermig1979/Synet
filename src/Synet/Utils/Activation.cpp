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

//#define SYNET_SIMD_SYNET_DISABLE

namespace Synet
{
    SYNET_INLINE float Elu32f(float value, float alpha)
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

    SYNET_INLINE float Gelu32f(float value)
    {
        return value * (::erff(value * float(M_SQRT1_2)) + 1.0f) * 0.5f;
    }

    void Gelu32f(const float* src, size_t size, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetGelu32f(src, size, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Gelu32f(src[i]);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float HardSigmoid32f(float value, float scale, float shift)
    {
        return Max(0.0f, Min(value * scale + shift, 1.0f));
    }

    void HardSigmoid32f(const float* src, size_t size, float scale, float shift, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetHardSigmoid32f(src, size, &scale, &shift, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = HardSigmoid32f(src[i], scale, shift);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Hswish32f(float value, float shift, float scale)
    {
        return Max(Min(value, shift) + shift, 0.0f) * scale * value;
    }

    void Hswish32f(const float* src, size_t size, float shift, float scale, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetHswish32f(src, size, &shift, &scale, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Hswish32f(src[i], shift, scale);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Mish32f(float value, float threshold)
    {
        return value > threshold ? value : value * (1.0f - 2.0f / (Square(::expf(value) + 1.0f) + 1.0f));
    }

    void Mish32f(const float* src, size_t size, float threshold, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetMish32f(src, size, &threshold, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Mish32f(src[i], threshold);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Relu32f(float value, float slope)
    {
        return Max(value, 0.0f) + slope * Min(value, 0.0f);
    }

    void Relu32f(const float* src, size_t size, float slope, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetRelu32f(src, size, &slope, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Relu32f(src[i], slope);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float RestrictRange32f(float value, float lower, float upper)
    {
        return Min(Max(lower, value), upper);
    }

    void RestrictRange32f(const float* src, size_t size, float lower, float upper, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetRestrictRange32f(src, size, &lower, &upper, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = RestrictRange32f(src[i], lower, upper);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Sigmoid32f(float value)
    {
        return 1.0f / (1.0f + ::expf(-value));
    }

    void Sigmoid32f(const float* src, size_t size, float* dst)
    {
        const float slope = 1.0f;
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetSigmoid32f(src, size, &slope, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Sigmoid32f(src[i]);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Softplus32f(float value, float beta, float threshold)
    {
        return value > threshold ? value : ::logf(1.0f + ::expf(value * beta)) / beta;
    }

    void Softplus32f(const float* src, size_t size, float beta, float threshold, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetSoftplus32f(src, size, &beta, &threshold, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Softplus32f(src[i], beta, threshold);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE float Swish32f(float value)
    {
        return value / (1.0f + ::expf(-value));
    }

    void Swish32f(const float* src, size_t size, float* dst)
    {
        const float slope = 1.0f;
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetSwish32f(src, size, &slope, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = Swish32f(src[i]);
#endif
    }
}