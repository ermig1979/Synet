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

#include "Synet/Layers/ConvolutionLayer.h"
#include "Synet/Quantization/Bf16.h" 

namespace Synet
{
    template <class T> class Convolution32fLayer : public Synet::ConvolutionLayer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;

        Convolution32fLayer(const LayerParam & param, Context* context)
            : ConvolutionLayer<T>(param, context)
        {
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _convolution32f.InternalBufferSize() * sizeof(Type);
        }

    protected:
        typedef typename ConvolutionLayer<T>::AlgParam AlgParam;

        virtual String InternalInfo() const
        {
            return  String(" fp32") + (_convolution32f.Enable() ? String(" ") + _convolution32f.Info() : String());
        }

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
        {
            const Tensors& weight = this->Weight();
            const ConvParam& conv = this->_conv;
            AlgParam & alg = this->_alg;
            dst->Reshape(conv.DstShape(alg.batch), src->Format());
            _convolution32f.Init(alg.batch, &conv, this->Param().convolution().bf16(), this->Options().bf16RoundTest);
            if (_convolution32f.Enable())
            {
                Base::Extend32f(buf, 0, Shp(_convolution32f.ExternalBufferSize()), src->Format());
                _convolution32f.SetParams(weight[0].CpuData(), &alg.internal, alg.bias ? weight[1].CpuData() : NULL,
                    conv.activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : alg.params);
            }
            else
                Base::Extend32f(buf, 0, Shp(conv.ImgSize()), src->Format());
        }

        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            ForwardCpu(src[0]->CpuData(), Base::Buf32f(buf, 0), dst[0]->CpuData());
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
            if (_convolution32f.Enable())
                _convolution32f.Forward(src, buf, dst);
            else
            {
                const Type * weight = this->Weight()[0].CpuData();
                const ConvParam& conv = this->_conv;
                const AlgParam& alg = this->_alg;
                for (size_t b = 0; b < alg.batch; ++b)
                {
                    const Type * tmp = src;
                    if (!alg.is1x1)
                    {
                        if (alg.trans)
                            Synet::ImgToRow(tmp, conv.srcH, conv.srcW, conv.srcC, conv.kernelY, conv.kernelX, 
                                conv.padY, conv.padX, conv.padH, conv.padW, conv.strideY, conv.strideX, 
                                conv.dilationY, conv.dilationX, conv.group, (const Type*)NULL, buf);
                        else
                            Synet::ImgToCol(tmp, conv.srcC, conv.srcH, conv.srcW, conv.kernelY, conv.kernelX, 
                                conv.padY, conv.padX, conv.padH, conv.padW, conv.strideY, conv.strideX, 
                                conv.dilationY, conv.dilationX, (const Type*)NULL, buf);
                        tmp = buf;
                    }
                    if (alg.trans)
                    {
                        assert(conv.group == 1 || conv.group == conv.srcC);
                        for (size_t g = 0; g < conv.group; ++g)
                            CpuGemm(CblasNoTrans, CblasNoTrans, alg.siS, alg.siD, alg.siW, Type(1), tmp + alg.grS * g, alg.ldS,
                                weight + alg.grW * g, alg.ldW, Type(0), dst + alg.grD * g, alg.ldD);
                    }
                    else
                    {
                        for (size_t g = 0; g < conv.group; ++g)
                            CpuGemm(CblasNoTrans, CblasNoTrans, alg.siD, alg.siS, alg.siW, Type(1), weight + alg.grW * g, alg.ldW,
                                tmp + alg.grS * g, alg.ldS, Type(0), dst + alg.grD * g, alg.ldD);
                    }
                    if (alg.bias)
                        CpuAddBias(this->Weight()[1].CpuData(), conv.dstC, conv.dstH*conv.dstW, dst, alg.trans);
                    switch (conv.activation)
                    {
                    case ActivationFunctionTypeIdentity:
                        break;
                    case ActivationFunctionTypeRelu:
                        CpuRelu(dst, alg.dSize, 0.0f, dst);
                        break;
                    case ActivationFunctionTypeLeakyRelu:
                        CpuRelu(dst, alg.dSize, alg.params[0], dst);
                        break;
                    case ActivationFunctionTypeRestrictRange:
                        CpuRestrictRange(dst, alg.dSize, alg.params[0], alg.params[1], dst);
                        break;
                    case ActivationFunctionTypePrelu:
                        PreluLayerForward(dst, this->Weight().back().CpuData(), conv.dstC, conv.dstH * conv.dstW, dst, alg.trans ? TensorFormatNhwc : TensorFormatNchw);
                        break;
                    case ActivationFunctionTypeElu:
                        CpuElu(dst, alg.dSize, alg.params[0], dst);
                        break;
                    case ActivationFunctionTypeHswish:
                        CpuHswish(dst, alg.dSize, alg.params[0], alg.params[1], dst);
                        break;
                    case ActivationFunctionTypeMish:
                        CpuMish(dst, alg.dSize, alg.params[0], dst);
                        break;
                    case ActivationFunctionTypeHardSigmoid:
                        CpuHardSigmoid(dst, alg.dSize, alg.params[0], alg.params[1], dst);
                        break;
                    case ActivationFunctionTypeSwish:
                        CpuSwish(dst, alg.dSize, dst);
                        break;
                    case ActivationFunctionTypeGelu:
                        CpuGelu(dst, alg.dSize, dst);
                        break;
                    default:
                        assert(0);
                    }
                    src += alg.sSize;
                    dst += alg.dSize;
                }
            }
        }

    private:
        Convolution32f _convolution32f;
    };
}