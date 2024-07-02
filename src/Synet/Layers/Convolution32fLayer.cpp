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
#include "Synet/Layers/Convolution32fLayer.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Layers/PreluLayer.h"

namespace Synet
{
    Convolution32fLayer::Convolution32fLayer(const LayerParam & param, Context* context)
        : ConvolutionLayer(param, context)
    {
    }

    size_t Convolution32fLayer::MemoryUsage() const
    {
        return ConvolutionLayer::MemoryUsage() + _convolution32f.InternalBufferSize() * sizeof(float);
    }

    String Convolution32fLayer::InternalInfo() const
    {
        return  String(" fp32") + (_convolution32f.Enable() ? String(" ") + _convolution32f.Info() : String());
    }

    bool Convolution32fLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        const Tensors& weight = this->Weight();
        const ConvParam& conv = this->_conv;
        AlgParam & alg = this->_alg;
        if(src->GetType() != TensorType32f || dst->GetType() != TensorType32f)
            SYNET_ERROR("Convolution32fLayer supports only FP32 input and output!");
        dst->Reshape(TensorType32f, conv.DstShape(alg.batch), src->Format());
        _convolution32f.Init(alg.batch, &conv, this->Param().convolution().quantizationLevel() == TensorType16b, this->Options().bf16RoundTest);
        if (_convolution32f.Enable())
        {
            Base::Extend32f(buf, 0, Shp(_convolution32f.ExternalBufferSize()), src->Format());
            _convolution32f.SetParams(weight[0].CpuData(), &alg.internal, alg.bias ? weight[1].CpuData() : NULL,
                conv.activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : alg.params);
        }
        else
            Base::Extend32f(buf, 0, Shp(conv.ImgSize()), src->Format());
        return true;
    }

    void Convolution32fLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        ForwardCpu(src[0]->Data<float>(), Base::Buf32f(buf, 0), dst[0]->Data<float>());
    }

    void Convolution32fLayer::ForwardCpu(const float * src, float* buf, float* dst)
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
}