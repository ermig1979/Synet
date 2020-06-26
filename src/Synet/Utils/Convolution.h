/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
    class Convolution32f
    {
    public:
        Convolution32f()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~Convolution32f()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        typedef void(*Gemm32fNNPtr)(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);

        SYNET_INLINE void Init(size_t batch, const ConvParam * conv, Gemm32fNNPtr gemm)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_batch != batch || _srcH != conv->srcH || _srcW != conv->srcW)
            {
                _batch = batch, _srcH = conv->srcH, _srcW = conv->srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                _context = ::SimdSynetConvolution32fInit(batch, (const SimdConvolutionParameters*)conv, gemm);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t ExternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetConvolution32fExternalBufferSize(_context) : 0;
#else
            return 1;
#endif
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetConvolution32fInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
            return 0;
        }

        SYNET_INLINE void SetParams(const float* weight, int* internal, const float* bias, const float* params)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetConvolution32fSetParams(_context, weight, (::SimdBool*)internal, bias, params);
#endif
        }

        SYNET_INLINE void Forward(const float* src, float* buf, float* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        if (_context)
            ::SimdSynetConvolution32fForward(_context, src, buf, dst);
#endif
        }

    private:
        void * _context;
        size_t _batch, _srcH, _srcW;
    };

    //-------------------------------------------------------------------------

    class Convolution8i
    {
    public:
        Convolution8i()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~Convolution8i()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t batch, const ConvParam* conv, QuantizationMethod method)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_batch != batch || _srcH != conv->srcH || _srcW != conv->srcW)
            {
                _batch = batch, _srcH = conv->srcH, _srcW = conv->srcW;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility;
                if (method == QuantizationMethodIECompatible)
                {
                    compatibility = SimdCpuInfo(SimdCpuInfoAvx512vnni) ? SimdSynetCompatibility8iPrecise : SimdSynetCompatibility8iOverflow;
                    compatibility = (SimdSynetCompatibilityType)(compatibility | SimdSynetCompatibilityFmaNoTail);
                }
                else if (method == QuantizationMethodSymmetricNarrowed)
                {
                    compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
                }
                else
                    return;
                _context = ::SimdSynetConvolution8iInit(batch, (const SimdConvolutionParameters*)conv, compatibility);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t ExternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetConvolution8iExternalBufferSize(_context) : 0;
#else
            return 1;
#endif
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetConvolution8iInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        SYNET_INLINE void SetParams(const float* weight, const float* bias, const float* params, const float* const *stats)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetConvolution8iSetParams(_context, weight, bias, params, stats);
#endif
        }

        SYNET_INLINE void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetConvolution8iForward(_context, src, buf, dst);
#endif
        }

    private:
        void * _context;
        size_t _batch, _srcH, _srcW;
    };
}