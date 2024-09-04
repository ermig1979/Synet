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

#include "Synet/Layers/Convolution8iLayer.h"
#include "Synet/Quantization/Const.h"
#include "Synet/Quantization/Gemm.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Utils/ImgToCol.h"

namespace Synet
{
    Convolution8iLayer::Convolution8iLayer(const LayerParam& param, Context* context, QuantizationMethod method)
        : ConvolutionLayer(param, context)
        , _method(method)
    {
        const ConvolutionParam& p = this->Param().convolution();
        assert(p.quantizationLevel() == TensorType8i);
        _src8u = false;
        _dst8u = false;
    }

    size_t Convolution8iLayer::MemoryUsage() const
    {
        return Base::MemoryUsage() + _convolution8i.InternalBufferSize()
            + _weight8i.MemoryUsage() + _norm32f.MemoryUsage() + _bias32f.MemoryUsage();
    }

    bool Convolution8iLayer::Can8i() const
    {
        return true;
    }

    bool Convolution8iLayer::Is8i() const
    {
        return true;
    }

    void Convolution8iLayer::DebugPrint(std::ostream& os, int flag, int first, int last, int precision)
    {
        Synet::DebugPrint(os, _srcCvt.scale, _srcCvt.channels, "_srcCvt.scale", first, last, precision);
        Synet::DebugPrint(os, _srcCvt.shift, _srcCvt.channels, "_srcCvt.shift", first, last, precision);
        _weight8i.DebugPrint(os, "_weight8i", true, first, last, precision);
        _norm32f.DebugPrint(os, "_norm32f", false, first, last, precision);
        _bias32f.DebugPrint(os, "_bias32f", false, first, last, precision);
        Synet::DebugPrint(os, _dstCvt.scale, _dstCvt.channels, "_dstCvt.scale", first, last, precision);
        Synet::DebugPrint(os, _dstCvt.shift, _dstCvt.channels, "_dstCvt.shift", first, last, precision);
    }

    String Convolution8iLayer::InternalInfo() const
    {
        std::stringstream info;
        info << " int8-" << (_src8u ? "u" : "f") << (_dst8u ? "u" : "f");
        if (_convolution8i.Enable())
            info << " " << _convolution8i.Info();
        return info.str();
    }

    bool Convolution8iLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        const Tensors& weight = this->Weight();
        const ConvParam& conv = this->_conv;
        AlgParam& alg = this->_alg;

        if ((src->GetType() != TensorType32f && src->GetType() != TensorType8u) || 
            (dst->GetType() != TensorType32f && dst->GetType() != TensorType8u))
            SYNET_ERROR("Convolution8iLayer supports only FP32 or INT8 input and output!");

        _src8u = src->GetType() == TensorType8u;
        _dst8u = dst->GetType() == TensorType8u;
        Shape shape = conv.DstShape(alg.batch);
        if (_dst8u)
            dst->Reshape(TensorType8u, shape, src->Format());
        else
            dst->Reshape(TensorType32f, shape, src->Format());
        _convolution8i.Init(alg.batch, &conv, _method);
        if (_convolution8i.Enable())
        {
            Base::Extend8u(buf, 0, Shp(_convolution8i.ExternalBufferSize()));
            const float* bias = alg.bias ? weight[1].Data<float>() : NULL;
            const float* params = conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : alg.params;
            const float* stats[4] = {
                this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data() };
            _convolution8i.SetParams(weight[0].Data<float>(), bias, params, stats);
        }
        else
        {
            if (!_src8u)
                Base::Extend8u(buf, 0, conv.SrcShape(1));
            if(!conv.Is1x1())
                Base::Extend8u(buf, 1, Shp(conv.ImgSize()));
            Base::Extend32i(buf, 0, conv.DstShape(1));
            if(_dst8u)
                Base::Extend32f(buf, 0, conv.DstShape(1));
            Quantize();
        }
        alg.internal = 1;
        return true;
    }

    void Convolution8iLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (_convolution8i.Enable())
            _convolution8i.Forward(src[0]->RawData(), Base::Buf8u(buf, 0), dst[0]->RawData());
        else
        {
            const AlgParam& alg = this->_alg;
            const float* src32f = _src8u ? NULL : src[0]->Data<float>();
            uint8_t* src8u = _src8u ? src[0]->Data<uint8_t>() : Base::Buf8u(buf, 0);
            uint8_t* buf8u = Base::Buf8u(buf, 1);
            int32_t* sum32i = Base::Buf32i(buf, 0);
            float* dst32f = _dst8u ? Base::Buf32f(buf, 0) : dst[0]->Data<float>();
            uint8_t* dst8u = _dst8u ? dst[0]->Data<uint8_t>() : NULL;
            for (size_t b = 0; b < alg.batch; ++b)
            {
                if(!_src8u)
                {
                    _srcCvt.Convert(src32f, src8u);
                    src32f += alg.sSize;
                }
                ForwardCpu(src8u, buf8u, sum32i, dst32f);
                if (_src8u)
                    src8u += alg.sSize;
                if (_dst8u)
                {
                    _dstCvt.Convert(dst32f, dst8u);
                    dst8u += alg.dSize;
                }
                else
                    dst32f += alg.dSize;
            }
        }
    }

    void Convolution8iLayer::SetNormMinMax(const float* weight, const float* scale, float* norm, float& min, float& max)
    {
        const ConvParam& conv = this->_conv;
        const AlgParam& alg = this->_alg;
        size_t G = conv.group, GD = conv.dstC, C = conv.srcC / G, K = conv.kernelY * conv.kernelX, CK = C * K;
        if (alg.trans)
        {
            for (size_t k = 0, kc = 0; k < K; ++k)
            {
                for (size_t c = 0; c < C; ++c, ++kc)
                {
                    norm[kc] = weight[kc * GD] / scale[c];
                    min = Min(min, norm[kc]);
                    max = Max(max, norm[kc]);
                }
            }
        }
        else
        {
            for (size_t c = 0, ck = 0; c < C; ++c)
            {
                for (size_t k = 0; k < K; ++k, ++ck)
                {
                    norm[ck] = weight[ck] / scale[c];
                    min = Min(min, norm[ck]);
                    max = Max(max, norm[ck]);
                }
            }
        }
    }

    void Convolution8iLayer::Quantize()
    {
        const ConvParam& conv = this->_conv;
        const AlgParam& alg = this->_alg;
        Stat& statS = *this->Stats(0)[0];
        Stat& statD = *this->Stats(2)[0];
        statS.Init8u(_method);
        statD.Init8u(_method);
        _weight8i.Reshape(TensorType8i, this->Weight()[0].Shape(), alg.trans ? TensorFormatNhwc : TensorFormatNchw);
        _norm32f.Reshape(TensorType32f, Shp(conv.dstC));
        _bias32f.Reshape(TensorType32f, Shp(conv.dstC));
        size_t G = conv.group, D = conv.dstC / G, C = conv.srcC / G, K = conv.kernelY * conv.kernelX, CK = C * K, GD = G * D;
        Floats normW(CK);
        const float* pSrcW = this->Weight()[0].Data<float>();
        const float* pSrcB = alg.bias ? this->Weight()[1].Data<float>() : NULL;
        const float* pScale = statS.scale32fTo8u.data();
        const float* pShift = statS.shift32fTo8u.data();
        float* pNormW = normW.data();
        int8_t* pDstW = _weight8i.Data<int8_t>();
        float* pNorm = _norm32f.Data<float>();
        float* pBias = _bias32f.Data<float>();
        bool avoidOverflow16i = statS.negative && _method == QuantizationMethodIECompatible;
        _srcCvt.Init(1, conv.srcC, conv.srcH, conv.srcW, (TensorFormat)alg.trans, statS.scale32fTo8u.data(), statS.shift32fTo8u.data(), _method);
        _dstCvt.Init(1, conv.dstC, conv.dstH, conv.dstW, (TensorFormat)alg.trans, statD.scale32fTo8u.data(), statD.shift32fTo8u.data(), _method);
        for (size_t g = 0; g < G; ++g)
        {
            for (size_t d = 0; d < D; ++d)
            {
                float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
                if (alg.trans)
                {
                    SetNormMinMax(pSrcW + d, pScale, pNormW, minW, maxW);
                    scale = statS.iMax / Max(Abs(maxW), Abs(minW));
                    for (size_t k = 0, kc = 0; k < K; ++k)
                    {
                        for (size_t c = 0; c < C; ++c, ++kc)
                        {
                            if (avoidOverflow16i)
                            {
                                int w = ConvertTo8i(pNormW[kc], scale, 0, statS.iMin, statS.iMax);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[kc * GD + d] = w / 2;
                                normB -= w * pShift[c];
                            }
                            else
                            {
                                pDstW[kc * GD + d] = ConvertTo8i(pNormW[kc], scale, 0, statS.iMin, statS.iMax);
                                normB -= pDstW[kc * GD + d] * pShift[c];
                            }
                        }
                    }
                }
                else
                {
                    SetNormMinMax(pSrcW + d * CK, pScale, pNormW, minW, maxW);
                    scale = statS.iMax / Max(Abs(maxW), Abs(minW));
                    for (size_t c = 0, ck = 0; c < C; ++c)
                    {
                        for (size_t k = 0; k < K; ++k, ++ck)
                        {
                            if (avoidOverflow16i)
                            {
                                int w = ConvertTo8i(pNormW[ck], scale, 0, statS.iMin, statS.iMax);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[d * CK + ck] = w / 2;
                                normB -= w * pShift[c];
                            }
                            else
                            {
                                pDstW[d * CK + ck] = ConvertTo8i(pNormW[ck], scale, 0, statS.iMin, statS.iMax);
                                normB -= pDstW[d * CK + ck] * pShift[c];
                            }
                        }
                    }
                }
                pNorm[d] = (avoidOverflow16i ? 2.0f : 1.0f) / scale;
                pBias[d] = (pSrcB ? pSrcB[d] : 0.0f) + normB / scale;
            }
            if (alg.trans)
            {
                pSrcW += D;
                pDstW += D;
            }
            else
            {
                pSrcW += CK * D;
                pDstW += CK * D;
            }
            if (pSrcB)
                pSrcB += D;
            pScale += C;
            pShift += C;
            pNorm += D;
            pBias += D;
        }
    }

    void Convolution8iLayer::ForwardCpu(const uint8_t* src, uint8_t* buf, int32_t* sum, float* dst)
    {
        const bool overflow16i = true;
        const ConvParam& conv = this->_conv;
        const AlgParam& alg = this->_alg;
        const uint8_t* zero = this->Stats(0)[0]->zero8u.data();
        const int8_t* weight = _weight8i.Data<int8_t>();
        const float* norm = _norm32f.Data<float>();
        const float* bias = _bias32f.Data<float>();
        const uint8_t* tmp = src;
        if (!alg.is1x1)
        {
            if (alg.trans)
                Synet::ImgToRow(tmp, conv.srcH, conv.srcW, conv.srcC, conv.kernelY, conv.kernelX,
                    conv.padY, conv.padX, conv.padH, conv.padW, conv.strideY, conv.strideX, conv.dilationY, conv.dilationX, conv.group, zero, buf);
            else
                Synet::ImgToCol(tmp, conv.srcC, conv.srcH, conv.srcW, conv.kernelY, conv.kernelX,
                    conv.padY, conv.padX, conv.padH, conv.padW, conv.strideY, conv.strideX, conv.dilationY, conv.dilationX, zero, buf);
            tmp = buf;
        }
        if (alg.trans)
        {
            assert(conv.group == 1 || conv.group == conv.srcC);
            if (conv.group == 1)
                Synet::CpuGemm8iNN(alg.siS, alg.siD, conv.kernelY * conv.kernelX, conv.srcC, tmp, alg.ldS, weight, alg.ldW, sum, alg.ldD, overflow16i);
            else
                for (size_t g = 0; g < conv.group; ++g)
                    Synet::CpuGemmNN(alg.siS, alg.siD, alg.siW, tmp + alg.grS * g, alg.ldS, weight + alg.grW * g, alg.ldW, sum + alg.grD * g, alg.ldD);
        }
        else
        {
            if (conv.group == 1)
                Synet::CpuGemm8iNN(alg.siD, alg.siS, conv.srcC, conv.kernelY * conv.kernelX, weight, alg.ldW, tmp, alg.ldS, sum, alg.ldD, overflow16i);
            else
                for (size_t g = 0; g < conv.group; ++g)
                    Synet::CpuGemmNN(alg.siD, alg.siS, alg.siW, weight + alg.grW * g, alg.ldW, tmp + alg.grS * g, alg.ldS, sum + alg.grD * g, alg.ldD);
        }
        Detail::Convert<int32_t, float, float>(sum, 1, conv.dstC, conv.dstH, conv.dstW, conv.dstF, norm, bias, 0, 0, dst);
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
            PreluLayerForward(dst, this->Weight().back().Data<float>(), conv.dstC, conv.dstH * conv.dstW, dst, alg.trans ? TensorFormatNhwc : TensorFormatNchw);
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
    }
}