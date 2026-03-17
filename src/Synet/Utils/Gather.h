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
    class GatherElements
    {
    public:
        GatherElements()
            : _context(NULL)
        {
        }

        virtual ~GatherElements()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(TensorType dataType, TensorType indexType, bool indexConst, size_t srcOuter, size_t srcCount, size_t srcInner, size_t idxCount)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                _context = ::SimdSynetGatherElementsInit((SimdTensorDataType)dataType, (SimdTensorDataType)indexType, indexConst ? SimdTrue : SimdFalse, srcOuter, srcCount, srcInner, idxCount);
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetGatherElementsInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        SYNET_INLINE void SetIndex(const uint8_t* idx)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetGatherElementsSetIndex(_context, idx);
#endif
        }

        SYNET_INLINE void Forward(const uint8_t* src, const uint8_t* idx, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetGatherElementsForward(_context, src, idx, dst);
#endif
        }

    private:
        void* _context;
    };
}