/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
    namespace Detail
    {
        union F32
        {
            SYNET_INLINE F32(float val) : f32{ val } {   }
            SYNET_INLINE F32(uint32_t val) : u32{ val } {  }

            float f32;
            uint32_t u32;
        };
    }

    SYNET_INLINE float RoundToBf16(float value)
    {
        return Detail::F32((Detail::F32(value).u32 + 0x8000) & 0xFFFF0000).f32;
    }

    inline void RoundAsTo16(const float * src, size_t size, float * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = RoundToBf16(src[i]);
    }

    SYNET_INLINE bool BFloat16HardwareSupport()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        return SimdCpuInfo(SimdCpuInfoAmxBf16) != 0;
#else
        return false;
#endif
    }
}
