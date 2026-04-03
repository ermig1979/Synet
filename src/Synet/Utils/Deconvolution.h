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

#include "Synet/Utils/ConvParam.h"

namespace Synet
{
    class Deconvolution32f
    {
    public:
        Deconvolution32f()
            : _context(NULL)
            , _batch(0)
        {
        }

        virtual ~Deconvolution32f()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context);
#endif
        }

        void Init(size_t batch, const ConvParam * conv)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_batch != batch || _srcH != conv->srcH || _srcW != conv->srcW)
            {
                _batch = batch, _srcH = conv->srcH, _srcW = conv->srcW;
                if (_context)
                    ::SimdRelease(_context);
                _context = ::SimdSynetDeconvolution32fInit(batch, (const SimdConvolutionParameters*)conv, SimdSynetCompatibilityDefault);
            }
#endif
        }

        bool Enable() const
        {
            return _context != NULL;
        }

        size_t ExternalBufferSize() const 
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution32fExternalBufferSize(_context) : 1;
#else
            return 1;
#endif
        }

        size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution32fInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        String Info() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution32fInfo(_context) : String();
#else
            return String();
#endif
        }

        void SetParams(const float* weight, int * internal, const float * bias, const float * params)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetDeconvolution32fSetParams(_context, weight, (::SimdBool*)internal, bias, params);
#endif
        }

        void Forward(const float* src, float* buf, float* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetDeconvolution32fForward(_context, src, buf, dst);
#endif
        }

    private:
        void * _context;
        size_t _batch, _srcH, _srcW;
    };

    //-------------------------------------------------------------------------------------------------

    class Deconvolution16b
    {
    public:
        Deconvolution16b()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~Deconvolution16b()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t batch, const ConvParam* conv)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_batch != batch || _srcH != conv->srcH || _srcW != conv->srcW)
            {
                _batch = batch, _srcH = conv->srcH, _srcW = conv->srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault;
                _context = ::SimdSynetDeconvolution16bInit(batch, (const SimdConvolutionParameters*)conv, compatibility);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t ExternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution16bExternalBufferSize(_context) : 1;
#else
            return 1;
#endif
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution16bInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        String Info() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetDeconvolution16bInfo(_context) : String();
#else
            return String();
#endif
        }

        SYNET_INLINE void SetParams(const float* weight, const float* bias, const float* params)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetDeconvolution16bSetParams(_context, weight, bias, params);
#endif
        }

        SYNET_INLINE void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetDeconvolution16bForward(_context, src, buf, dst);
#endif
        }

    private:
        void* _context;
        size_t _batch, _srcH, _srcW;
    };
}