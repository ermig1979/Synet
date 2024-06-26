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

#include "Synet/Utils/ConvParam.h"

namespace Synet
{
    class MergedConvolution32f
    {
    public:
        MergedConvolution32f()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~MergedConvolution32f()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        void Init(size_t batch, const ConvParam * convs, size_t count, int add, bool bf16, bool bf16test)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_batch != batch || _srcH != convs[0].srcH || _srcW != convs[0].srcW)
            {
                _batch = batch, _srcH = convs[0].srcH, _srcW = convs[0].srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault;
                if (bf16)
                {
                    if(bf16test)
                        compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility16bfSoft | SimdSynetCompatibilityFmaUse);
                    else
                        compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility16bfHard | SimdSynetCompatibilityFmaUse);
                }
                if (convs[1].dstH > 1 && convs[1].dstW > 1)
                    _context = ::SimdSynetMergedConvolution32fInit(batch, (const SimdConvolutionParameters*)convs, count, (SimdBool)add, compatibility);
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
            return _context ? ::SimdSynetMergedConvolution32fExternalBufferSize(_context) : 1;
#else
            return 1;
#endif
        }

        size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution32fInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        String Info() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution32fInfo(_context) : String();
#else
            return String();
#endif
        }

        void SetParams(const float * const * weight, int * internal, const float* const * bias, const float* const * params)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution32fSetParams(_context, weight, (::SimdBool*)internal, bias, params);
#endif
        }

        void Forward(const float* src, float* buf, float* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution32fForward(_context, src, buf, dst);
#endif
        }

    private:
        void * _context;
        size_t _batch, _srcH, _srcW;
    };

    //-------------------------------------------------------------------------------------------------

    class MergedConvolution16b
    {
    public:
        MergedConvolution16b()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~MergedConvolution16b()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        void Init(size_t batch, const ConvParam* convs, size_t count)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_batch != batch || _srcH != convs[0].srcH || _srcW != convs[0].srcW)
            {
                _batch = batch, _srcH = convs[0].srcH, _srcW = convs[0].srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility = SimdSynetCompatibilityDefault;
                if (convs[1].dstH > 1 && convs[1].dstW > 1)
                    _context = ::SimdSynetMergedConvolution16bInit(batch, (const SimdConvolutionParameters*)convs, count, compatibility);
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
            return _context ? ::SimdSynetMergedConvolution16bExternalBufferSize(_context) : 1;
#else
            return 1;
#endif
        }

        size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution16bInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        String Info() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution16bInfo(_context) : String();
#else
            return String();
#endif
        }

        void SetParams(const float* const* weight, int* internal, const float* const* bias, const float* const* params)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution16bSetParams(_context, weight, (::SimdBool*)internal, bias, params);
#endif
        }

        void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution16bForward(_context, src, buf, dst);
#endif
        }

    private:
        void* _context;
        size_t _batch, _srcH, _srcW;
    };

    //-------------------------------------------------------------------------------------------------

    class MergedConvolution8i
    {
    public:
        MergedConvolution8i()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~MergedConvolution8i()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t batch, const ConvParam* convs, size_t count, QuantizationMethod method)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_batch != batch || _srcH != convs[0].srcH || _srcW != convs[0].srcW)
            {
                _batch = batch, _srcH = convs[0].srcH, _srcW = convs[0].srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility;
                if (method == QuantizationMethodIECompatible)
                {
                    compatibility = SimdCpuInfo(SimdCpuInfoAvx512vnni) ? SimdSynetCompatibility8iPrecise : SimdSynetCompatibility8iOverflow;
                    compatibility = (SimdSynetCompatibilityType)(compatibility | SimdSynetCompatibilityFmaNoTail);
                }
                else if (method == QuantizationMethodSymmetricNarrowed || method == QuantizationMethodUnifiedNarrowed)
                {
                    compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
                }
                else
                    return;
                _context = ::SimdSynetMergedConvolution8iInit(batch, (const SimdConvolutionParameters*)convs, count, compatibility);
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
            return _context ? ::SimdSynetMergedConvolution8iExternalBufferSize(_context) : 1;
#else
            return 1;
#endif
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution8iInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        String Info() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetMergedConvolution8iInfo(_context) : String();
#else
            return String();
#endif
        }

        SYNET_INLINE void SetParams(const float* const* weight, int* internal, const float* const* bias, const float* const* params, const float* const* stats)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution8iSetParams(_context, weight, (::SimdBool*)internal, bias, params, stats);
#endif
        }

        SYNET_INLINE void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetMergedConvolution8iForward(_context, src, buf, dst);
#endif
        }

    private:
        void* _context;
        size_t _batch, _srcH, _srcW;
    };
}