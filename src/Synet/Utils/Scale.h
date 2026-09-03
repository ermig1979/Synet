/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Params.h"

namespace Synet
{
    class Scale16b
    {
    public:
        Scale16b()
            : _context(NULL)
            , _spatial(0)
        {
        }

        virtual ~Scale16b()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t channels, size_t spatial, TensorType srcType, TensorType dstType, TensorFormat format, bool norm, bool bias)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_spatial != spatial)
            {
                _spatial = spatial;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                _context = ::SimdSynetScale16bInit(channels, spatial, (SimdTensorDataType)srcType, (SimdTensorDataType)dstType, 
                    (SimdTensorFormatType)format, norm ? SimdTrue : SimdFalse, bias ? SimdTrue : SimdFalse);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE void Forward(const uint8_t* src, const float * norm, const float * bias, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetScale16bForward(_context, src, norm, bias, dst);
#endif
        }

    private:
        void* _context;
        size_t _spatial;
    };
}