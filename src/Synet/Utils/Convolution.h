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
    template <class T>  class Convolution
    {
    public:
        Convolution()
            : _convolution(NULL)
            , _batch(0)
            , _srcH(0)
            , _srcW(0)
        {
        }

        virtual ~Convolution()
        {
        }

        typedef void(*Gemm32fNNPtr)(size_t M, size_t N, size_t K, const T * alpha, const T * A, size_t lda, const T * B, size_t ldb, const T * beta, T * C, size_t ldc);

        void Init(int trans, size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, 
            size_t padY, size_t padX, size_t padH, size_t padW, size_t group, ActivationFunctionType activation, Gemm32fNNPtr gemm)
        {
        }

        bool Enable()
        {
            return _convolution != NULL;
        }

        size_t ExternalBufferSize() const 
        {
            return 1;
        }

        size_t InternalBufferSize() const
        {
            return 0;
        }

        void SetParams(const T * weight, int trans, int * internal, const T * bias, const T * params)
        {
        }

        void Forward(const T * src, T * buf, T * dst)
        {
        }

    private:
        void * _convolution;
        size_t _batch, _srcH, _srcW;
    };

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template<> SYNET_INLINE Convolution<float>::~Convolution()
    {
        if (_convolution)
            ::SimdRelease(_convolution);
    }

    template<> SYNET_INLINE void Convolution<float>::Init(int trans, size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
        size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX,
        size_t padY, size_t padX, size_t padH, size_t padW, size_t group, ActivationFunctionType activation, ::SimdGemm32fNNPtr gemm)
    {
        if (_batch != batch || _srcH != srcH || _srcW != srcW)
        {
            _batch = batch, _srcH = srcH, _srcW = srcW;
            if (_convolution)
                ::SimdRelease(_convolution);
            _convolution = ::SimdConvolutionInit((::SimdBool)trans, batch, srcC, srcH, srcW, dstC, kernelY, kernelX,
                    dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, (::SimdConvolutionActivationType)activation, gemm);
        }
    }

    template<> SYNET_INLINE size_t Convolution<float>::ExternalBufferSize() const
    {
        return ::SimdConvolutionExternalBufferSize(_convolution);
    }

    template<> SYNET_INLINE size_t Convolution<float>::InternalBufferSize() const
    {
        return ::SimdConvolutionInternalBufferSize(_convolution);
    }

    template<> SYNET_INLINE void Convolution<float>::SetParams(const float * weight, int trans, int * internal, const float * bias, const float * params)
    {
        ::SimdConvolutionSetParams(_convolution, weight, (::SimdBool)trans, (::SimdBool*)internal, bias, params);
    }

    template<> SYNET_INLINE void Convolution<float>::Forward(const float * src, float * buf, float * dst)
    {
        ::SimdConvolutionForward(_convolution, src, buf, dst);
    }
#endif
}