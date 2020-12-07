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

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/MergedConvolutionLayer.h"

namespace Synet
{
    namespace Detail
    {
        template<class T, int update> struct Update
        {
            static void Func(T * ptr, T val);
        };

        template<class T> struct Update<T, 0>
        {
            static SYNET_INLINE void Func(T * ptr, T val)
            {
                *ptr = val;
            }
        };

        template<class T> struct Update<T, 1>
        {
            static SYNET_INLINE void Func(T * ptr, T val)
            {
                *ptr += val;
            }
        };

        template<class T, ActivationFunctionType activation, int update> void MergedConvolutionLayerDirect(
            const T * src, const ConvParam & conv, const T * weight, const T * bias, const T * params, T * dst)
        {
            Tensor<T> buffer({ conv.dstC });
            T * buf = buffer.CpuData();
            for (size_t dy = 0; dy < conv.dstH; ++dy)
            {
                for (size_t dx = 0; dx < conv.dstW; ++dx)
                {
                    if (bias)
                        memcpy(buf, bias, conv.dstC * sizeof(T));
                    else
                        memset(buf, 0, conv.dstC * sizeof(T));
                    for (size_t ky = 0; ky < conv.kernelY; ++ky)
                    {
                        size_t sy = dy * conv.strideY + ky - conv.padY;
                        if (sy < conv.srcH)
                        {
                            for (size_t kx = 0; kx < conv.kernelX; ++kx)
                            {
                                size_t sx = dx * conv.strideX + kx - conv.padX;
                                if (sx < conv.srcW)
                                {
                                    const float * pw = weight + (ky*conv.kernelX + kx)*conv.srcC*conv.dstC;
                                    const float * ps = src + (sy*conv.srcW + sx)*conv.srcC;
                                    for (size_t sc = 0; sc < conv.srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < conv.dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += conv.dstC;
                                    }
                                }
                            }
                        }
                    }
                    for (size_t dc = 0; dc < conv.dstC; ++dc)
                        Update<T, update>::Func(dst + dc, Activation<T, activation>::Func(buf[dc], params, dc));
                    dst += conv.dstC;
                }
            }
        }
    }

    template <class T> class MergedConvolution32fLayer : public MergedConvolutionLayer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;       
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolution32fLayer(const LayerParam & param)
            : MergedConvolutionLayer<T>(param)
        {
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _mergedConvolution32f.InternalBufferSize() * sizeof(Type);
        }

    protected:
        typedef typename MergedConvolutionLayer<T>::AlgParam AlgParam;

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
        {
            AlgParam& a = this->_alg;
            size_t directIdx, depthwiseIdx;
            if (a.conv[0].group == 1 && a.conv[1].IsDepthwise())
                directIdx = 0, depthwiseIdx = 1;
            else if (a.count == 2 && a.conv[0].IsDepthwise() && a.conv[1].Is1x1())
                directIdx = 1, depthwiseIdx = 0;
            else
                assert(0);

            switch (a.conv[directIdx].activation)
            {
            case ActivationFunctionTypeIdentity: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 0>; break;
            case ActivationFunctionTypeRelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 0>; break;
            case ActivationFunctionTypeLeakyRelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 0>; break;
            case ActivationFunctionTypeRestrictRange: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 0>; break;
            case ActivationFunctionTypePrelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 0>; break;
            case ActivationFunctionTypeElu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 0>; break;
            case ActivationFunctionTypeHswish: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 0>; break;
            case ActivationFunctionTypeMish: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeMish, 0>; break;
            default: assert(0);
            }

            switch (a.conv[depthwiseIdx].activation)
            {
            case ActivationFunctionTypeIdentity: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeIdentity>; break;
            case ActivationFunctionTypeRelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeRelu>; break;
            case ActivationFunctionTypeLeakyRelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeLeakyRelu>; break;
            case ActivationFunctionTypeRestrictRange: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeRestrictRange>; break;
            case ActivationFunctionTypePrelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypePrelu>; break;
            case ActivationFunctionTypeElu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeElu>; break;
            case ActivationFunctionTypeHswish: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeHswish>; break;
            case ActivationFunctionTypeMish: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeMish>; break;
            default: assert(0);
            }

            if (a.count > 2)
            {
                assert(a.conv[2].Is1x1());
                if (a.add)
                {
                    switch (a.conv[2].activation)
                    {
                    case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 1>; break;
                    case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 1>; break;
                    case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 1>; break;
                    case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 1>; break;
                    case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 1>; break;
                    case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 1>; break;
                    case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 1>; break;
                    case ActivationFunctionTypeMish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeMish, 1>; break;
                    default: assert(0);
                    }
                }
                else
                {
                    switch (a.conv[2].activation)
                    {
                    case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 0>; break;
                    case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 0>; break;
                    case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 0>; break;
                    case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 0>; break;
                    case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 0>; break;
                    case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 0>; break;
                    case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 0>; break;
                    case ActivationFunctionTypeMish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeMish, 0>; break;
                    default: assert(0);
                    }
                }
            }

            const ConvParam& back = a.conv[a.count - 1];
            dst->Reshape(Shp(a.batch, back.dstH, back.dstW, back.dstC), src->Format());

            _mergedConvolution32f.Init(a.batch, a.conv, a.count, a.add);
            if (_mergedConvolution32f.Enable())
            {
                Base::Extend32f(buf, 0, Shp(_mergedConvolution32f.ExternalBufferSize()));
                _mergedConvolution32f.SetParams(a.weight, a.internal, a.bias, a.params);
            }
            else
            {
                Base::Extend32f(buf, 0, a.conv[0].DstShape(1));
                if (a.count > 2)
                    Base::Extend32f(buf, 1, a.conv[1].DstShape(1));
            }
        }

        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
             ForwardCpu(src[0]->CpuData(), Base::Buf32f(buf, 0), Base::Buf32f(buf, 1), dst[0]->CpuData());
        }

        void ForwardCpu(const float * src, float* buf0, float* buf1, float* dst)
        {
            if (_mergedConvolution32f.Enable())
                _mergedConvolution32f.Forward(src, buf0, dst);
            else
            {
                const AlgParam& a = this->_alg;
                for (size_t b = 0; b < a.batch; ++b)
                {
                    _convolution[0](src, a.conv[0], a.weight[0], a.bias[0], a.params[0], buf0);
                    if (a.count > 2)
                    {
                        _convolution[1](buf0, a.conv[1], a.weight[1], a.bias[1], a.params[1], buf1);
                        if (a.add)
                            memcpy(dst, src, sizeof(T) * a.dSize);
                        _convolution[2](buf1, a.conv[2], a.weight[2], a.bias[2], a.params[2], dst);
                    }
                    else
                        _convolution[1](buf0, a.conv[1], a.weight[1], a.bias[1], a.params[1], dst);
                    src += a.sSize;
                    dst += a.dSize;
                }
            }
        }

    private:
        typedef void(*ConvolutionBiasActivationPtr)(const T * src, const ConvParam & conv, const T * weight, const T * bias, const T * params, T * dst);
        ConvolutionBiasActivationPtr _convolution[Detail::MCC_MAX];

        MergedConvolution32f _mergedConvolution32f;
    };
}