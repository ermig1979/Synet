/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
    class InnerProduct32f
    {
    public:
        InnerProduct32f()
            : _context(NULL)
            , _batch(0)
        {
        }

        virtual ~InnerProduct32f()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t batch, size_t input, size_t output, int transpose)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_batch != batch)
            {
                _batch = batch;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                _context = ::SimdSynetInnerProduct32fInit(batch, input, output, transpose ? SimdTrue : SimdFalse, SimdConvolutionActivationIdentity);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetInnerProduct32fInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        SYNET_INLINE void SetParams(const float* weight, int* internal, const float* bias, const float* params)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetInnerProduct32fSetParams(_context, weight, (::SimdBool*)internal, bias, params);
#endif
        }

        SYNET_INLINE void Forward(const float* src, float* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        if (_context)
            ::SimdSynetInnerProduct32fForward(_context, src, dst);
#endif
        }

    private:
        void * _context;
        size_t _batch;
    };
}