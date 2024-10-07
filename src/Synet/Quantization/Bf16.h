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

#include "Synet/Common.h"

namespace Synet
{
    namespace Bf16
    {
        union Bits
        {
            float f32;
            uint32_t u32;

            SYNET_INLINE Bits(float val) : f32(val) { }
            SYNET_INLINE Bits(uint32_t val) : u32(val) { }
        };

        const int SHIFT = 16;
        const uint32_t MASK = 0xFFFF0000;
        const uint32_t ROUND = 0x00007FFF;
    }

    SYNET_INLINE float RoundToBFloat16(float value)
    {
        uint32_t u32 = Bf16::Bits(value).u32;
        uint32_t round = Bf16::ROUND + ((u32 >> Bf16::SHIFT) & 1);
        return Bf16::Bits((u32 + round) & Bf16::MASK).f32;
    }

    SYNET_INLINE uint16_t Float32ToBFloat16(float value)
    {
        uint32_t u32 = Bf16::Bits(value).u32;
        uint32_t round = Bf16::ROUND + ((u32 >> Bf16::SHIFT) & 1);
        return uint16_t((u32 + round) >> Bf16::SHIFT);
    }

    SYNET_INLINE float BFloat16ToFloat32(uint16_t value)
    {
        return Bf16::Bits(uint32_t(value) << Bf16::SHIFT).f32;
    }

    inline void RoundToBFloat16(const float * src, size_t size, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = RoundToBFloat16(src[i]);
    }

    SYNET_INLINE bool BFloat16HardwareSupport()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        return SimdCpuInfo(SimdCpuInfoAmxBf16) != 0;
#else
        return false;
#endif
    }

    //-------------------------------------------------------------------------------------------------

    template <class S, class D> SYNET_INLINE D Convert(const S& src)
    {
        return (D)src;
    }

    template <> SYNET_INLINE float Convert(const uint16_t& src)
    {
        return BFloat16ToFloat32(src);
    }

    template <> SYNET_INLINE uint16_t Convert(const float& src)
    {
        return Float32ToBFloat16(src);
    }
}
