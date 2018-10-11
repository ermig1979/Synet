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
        {
        }

        virtual ~Convolution()
        {
        }

        void Init(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
        }

        bool Enable()
        {
            return _convolution != NULL;
        }

        size_t BufferSize()
        {
            return 1;
        }

        void SetWeight(const T * weight, const T * bias)
        {
        }

        void SetActivation(ActivationFunctionType type, const T * params)
        {
        }

        void Forward(const T * src, T * buf, T * dst)
        {
        }

    private:
        void * _convolution;
    };

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template<> SYNET_INLINE Convolution<float>::~Convolution()
    {
        if (_convolution)
            ::SimdRelease(_convolution);
    }

    template<> SYNET_INLINE void Convolution<float>::Init(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
    {
        _convolution = ::SimdConvolutionInit(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
    }

    template<> SYNET_INLINE size_t Convolution<float>::BufferSize()
    {
        return ::SimdConvolutionBufferSize(_convolution);
    }

    template<> SYNET_INLINE void Convolution<float>::SetWeight(const float * weight, const float * bias)
    {
        ::SimdConvolutionSetWeight(_convolution, weight, bias);
    }

    template<> SYNET_INLINE void Convolution<float>::SetActivation(ActivationFunctionType type, const float * params)
    {
        ::SimdConvolutionSetActivation(_convolution, (::SimdConvolutionActivationType)type, params);
    }

    template<> SYNET_INLINE void Convolution<float>::Forward(const float * src, float * buf, float * dst)
    {
        ::SimdConvolutionForward(_convolution, src, buf, dst);
    }
#endif
}