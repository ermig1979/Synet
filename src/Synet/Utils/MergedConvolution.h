/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
    template <class T>  class MergedConvolution
    {
    public:
        MergedConvolution()
            : _context(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~MergedConvolution()
        {
        }

        typedef void(*Gemm32fNNPtr)(size_t M, size_t N, size_t K, const T * alpha, const T * A, size_t lda, const T * B, size_t ldb, const T * beta, T * C, size_t ldc);

        void Init(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, 
            size_t padY, size_t padX, size_t padH, size_t padW, ActivationFunctionType activation0, ActivationFunctionType activation1, Gemm32fNNPtr gemm)
        {
        }

        bool Enable()
        {
            return _context != NULL;
        }

        size_t ExternalBufferSize() const 
        {
            return 1;
        }

        size_t InternalBufferSize() const
        {
            return 0;
        }

        void SetParams(const T * weight0, const T * weight1, int * internal, const T * bias0, const T * bias1, const T * params0, const T * params1)
        {
        }

        void Forward(const T * src, T * buf, T * dst)
        {
        }

    private:
        void * _context;
        size_t _batch, _srcH, _srcW;
    };

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template<> SYNET_INLINE MergedConvolution<float>::~MergedConvolution()
    {
        if (_context)
            ::SimdRelease(_context);
    }

    template<> SYNET_INLINE void MergedConvolution<float>::Init(
        size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t strideY, size_t strideX,
        size_t padY, size_t padX, size_t padH, size_t padW, ActivationFunctionType activation0, ActivationFunctionType activation1, Gemm32fNNPtr gemm)
    {
        if (_batch != batch || _srcH != srcH || _srcW != srcW)
        {
            _batch = batch, _srcH = srcH, _srcW = srcW;
            if (_context)
                ::SimdRelease(_context);
            _context = ::SimdMergedConvolutionInit(batch, srcC, srcH, srcW, dstC, kernelY, kernelX, strideY, strideX, 
                padY, padX, padH, padW, (::SimdConvolutionActivationType)activation0, (::SimdConvolutionActivationType)activation1, gemm);
        }
    }

    template<> SYNET_INLINE size_t MergedConvolution<float>::ExternalBufferSize() const
    {
        return ::SimdMergedConvolutionExternalBufferSize(_context);
    }

    template<> SYNET_INLINE size_t MergedConvolution<float>::InternalBufferSize() const
    {
        return ::SimdMergedConvolutionInternalBufferSize(_context);
    }

    template<> SYNET_INLINE void MergedConvolution<float>::SetParams(const float * weight0, const float * weight1, int * internal, 
        const float * bias0, const float * bias1, const float * params0, const float * params1)
    {
        ::SimdMergedConvolutionSetParams(_context, weight0, weight1, (::SimdBool*)internal, bias0, bias1, params0, params1);
    }

    template<> SYNET_INLINE void MergedConvolution<float>::Forward(const float * src, float * buf, float * dst)
    {
        ::SimdMergedConvolutionForward(_context, src, buf, dst);
    }
#endif
}