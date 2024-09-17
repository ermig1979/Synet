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

#include "Synet/Layers/MergedConvolution8iLayer.h"
#include "Synet/Utils/DebugPrint.h"
#include "Synet/Quantization/Gemm.h"

namespace Synet
{
    MergedConvolution8iLayer::MergedConvolution8iLayer(const LayerParam & param, Context* context, QuantizationMethod method)
        : MergedConvolutionLayer(param, context)
        , _method(method)
    {
    }

    LowPrecisionType MergedConvolution8iLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    size_t MergedConvolution8iLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _mergedConvolution8i.InternalBufferSize() +
            + _weight8i[0].MemoryUsage() + _norm32f[0].MemoryUsage() + _bias32f[0].MemoryUsage()
            + _weight8i[1].MemoryUsage() + _norm32f[1].MemoryUsage() + _bias32f[1].MemoryUsage();
    }

    void MergedConvolution8iLayer::DebugPrint(std::ostream& os, int flag, int first, int last, int precision)
    {
        Synet::DebugPrint(os, _srcCvt.scale, _srcCvt.channels, "_srcCvt.scale", first, last, precision);
        Synet::DebugPrint(os, _srcCvt.shift, _srcCvt.channels, "_srcCvt.shift", first, last, precision);
        _weight8i[0].DebugPrint(os, "_weight8i[0]", true, first, last, precision);
        _norm32f[0].DebugPrint(os, "_norm32f[0]", false, first, last, precision);
        _bias32f[0].DebugPrint(os, "_bias32f[0]", false, first, last, precision);
        Synet::DebugPrint(os, _intCvt.scale, _intCvt.channels, "_intCvt.scale", first, last, precision);
        Synet::DebugPrint(os, _intCvt.shift, _intCvt.channels, "_intCvt.shift", first, last, precision);
        _weight8i[1].DebugPrint(os, "_weight8i[1]", true, first, last, precision);
        _norm32f[1].DebugPrint(os, "_norm32f[1]", false, first, last, precision);
        _bias32f[1].DebugPrint(os, "_bias32f[1]", false, first, last, precision);
        Synet::DebugPrint(os, _dstCvt.scale, _dstCvt.channels, "_dstCvt.scale", first, last, precision);
        Synet::DebugPrint(os, _dstCvt.shift, _dstCvt.channels, "_dstCvt.shift", first, last, precision);
    }

    String MergedConvolution8iLayer::InternalInfo() const
    {
        std::stringstream info;
        info << " int8-" << (_src8u ? "u" : "f") << (_dst8u ? "u" : "f");
        if (_mergedConvolution8i.Enable())
            info << " " << _mergedConvolution8i.Info();
        return info.str();
    }

    bool MergedConvolution8iLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        if ((src->GetType() != TensorType32f && src->GetType() != TensorType8u) ||
            (dst->GetType() != TensorType32f && dst->GetType() != TensorType8u))
            SYNET_ERROR("MergedConvolution8iLayer supports only FP32 or INT8 input and output!");
        AlgParam& a = this->_alg;
        if(a.add != 0)
            SYNET_ERROR("MergedConvolution8iLayer support only add=0 case!");
        const ConvParam& back = a.conv[a.count - 1];
        _src8u = src->GetType() == TensorType8u;
        _dst8u = dst->GetType() == TensorType8u;
        _dw0 = a.conv[0].IsDepthwise();
        Shape shape = back.DstShape(a.batch);
        dst->Reshape(dst->GetType(), shape, src->Format());

        _mergedConvolution8i.Init(a.batch, a.conv, a.count, _method);
        if (_mergedConvolution8i.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_mergedConvolution8i.ExternalBufferSize()));
            if (!this->Stats(1).empty() && (_method == QuantizationMethodIECompatible || 
                _method == QuantizationMethodUnifiedNarrowed))
                this->Stats(1).back()->Unify();
            const float* stats[6] = { 
                this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                this->Stats(1).empty() ? NULL : this->Stats(1).back()->min.data(),
                this->Stats(1).empty() ? NULL : this->Stats(1).back()->max.data(),
                this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data()};
            _mergedConvolution8i.SetParams(a.weight, a.internal, a.bias, a.params, stats);
        }
        else
        {
            if (_dw0)
            {
                if (_src8u)
                    Layer::Extend32f(buf, 0, a.conv[0].SrcShape(1));
                Layer::Extend32f(buf, 1, a.conv[0].DstShape(1));
                Layer::Extend8u(buf, 0, a.conv[1].SrcShape(1));
                Layer::Extend32i(buf, 0, a.conv[1].DstShape(1));
                a.internal[1] = 1;
            }
            else
            {
                if (!_src8u)
                    Layer::Extend8u(buf, 0, a.conv[0].SrcShape(1));
                if(!a.conv[0].Is1x1())
                    Layer::Extend8u(buf, 1, Shp(a.conv[0].ImgSize()));
                Layer::Extend32i(buf, 0, a.conv[0].DstShape(1));
                Layer::Extend32f(buf, 0, a.conv[0].DstShape(1));
                if (a.count == 3)
                {
                    Layer::Extend32f(buf, 1, a.conv[1].DstShape(1));
                    Layer::Extend8u(buf, 1, a.conv[1].DstShape(1));
                    Layer::Extend32i(buf, 0, a.conv[2].DstShape(1));
                    a.internal[2] = 1;
                }
                a.internal[0] = 1;
            }
            if(_dst8u)
                Layer::Extend32f(buf, 1, back.DstShape(1));
            Init();
        }
        return true;
    }

    void MergedConvolution8iLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_mergedConvolution8i.Enable())
            _mergedConvolution8i.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
        else
        {
            float* buf0 = Layer::Buf32f(buf, 0);
            float* buf1 = Layer::Buf32f(buf, 1);
            uint8_t* buf2 = Layer::Buf8u(buf, 0);
            uint8_t* buf3 = Layer::Buf8u(buf, 1);
            int32_t* buf4 = Layer::Buf32i(buf, 0);

            const AlgParam& a = this->_alg;
            float* src32f = _src8u ? (_dw0 ? buf0 : NULL) : src[0]->Data<float>();
            uint8_t* src8u = _src8u ? src[0]->Data<uint8_t>() : (_dw0 ? NULL : buf2);
            float* dst32f = _dst8u ? buf1 : dst[0]->Data<float>();
            uint8_t* dst8u = _dst8u ? dst[0]->Data<uint8_t>() : NULL;
            for (size_t b = 0; b < a.batch; ++b)
            {
                if (_dw0)
                {
                    if (_src8u)
                    {
                        _srcCvt.Convert(src8u, src32f);
                        src8u += a.sSize;
                    }  
                    _depthwise(src32f, a.conv[0], a.weight[0], a.bias[0], a.params[0], buf1);
                    if (!_src8u)
                        src32f += a.sSize;
                    _intCvt.Convert(buf1, buf3);
                    DirectConvolution8i(buf3, 1, 0, NULL, NULL, buf4, dst32f);
                }
                else
                {
                    if (!_src8u)
                    {
                        _srcCvt.Convert(src32f, src8u);
                        src32f += a.sSize;
                    }
                    DirectConvolution8i(src8u, 0, 0, this->Stats(0)[0]->zero8u.data(), buf3, buf4, buf0);
                    if (_src8u)
                        src8u += a.sSize;
                    _depthwise(buf0, a.conv[1], a.weight[1], a.bias[1], a.params[1], a.IsCdc() ? buf1 : dst32f);
                    if (a.IsCdc())
                    {
                        _intCvt.Convert(buf1, buf3);
                        DirectConvolution8i(buf3, 2, 1, NULL, NULL, buf4, dst32f);
                    }                    
                }
                if (_dst8u)
                {
                    _dstCvt.Convert(dst32f, dst8u);
                    dst8u += a.dSize;
                }
                else
                    dst32f += a.dSize;
            }
        }
    }

    void MergedConvolution8iLayer::DirectConvolution8i(const uint8_t* src, size_t cIdx, size_t wIdx, const uint8_t* zero, uint8_t* buf, int32_t* sum, float* dst)
    {
        const AlgParam& a = this->_alg;
        const bool overflow16i = true;
        const ConvParam & conv = a.conv[cIdx];
        const int8_t* weight = _weight8i[wIdx].Data<int8_t>();
        const float* norm = _norm32f[wIdx].Data<float>();
        const float* bias = _bias32f[wIdx].Data<float>();
        const float* params = a.params[wIdx];
        const uint8_t* tmp = src;
        if (!conv.Is1x1())
        {
            Synet::ImgToRow(tmp, conv.srcH, conv.srcW, conv.srcC, conv.kernelY, conv.kernelX,
                conv.padY, conv.padX, conv.padH, conv.padW, conv.strideY, conv.strideX, conv.dilationY, conv.dilationX, conv.group, zero, buf);
            tmp = buf;
        }
        size_t K = conv.srcC * conv.kernelY * conv.kernelX, N = conv.dstH * conv.dstW, M = conv.dstC;
        Synet::CpuGemm8iNN(N, M, conv.kernelY * conv.kernelX, conv.srcC, tmp, K, weight, M, sum, M, overflow16i);
        Detail::Convert<int32_t, float, float>(sum, 1, conv.dstC, conv.dstH, conv.dstW, conv.dstF, norm, bias, 0, 0, dst);
        size_t dSize = conv.dstC * conv.dstH * conv.dstW;
        switch (conv.activation)
        {
        case ActivationFunctionTypeIdentity:
            break;
        case ActivationFunctionTypeRelu:
            CpuRelu(dst, dSize, 0.0f, dst);
            break;
        case ActivationFunctionTypeLeakyRelu:
            CpuRelu(dst, dSize, params[0], dst);
            break;
        case ActivationFunctionTypeRestrictRange:
            CpuRestrictRange(dst, dSize, params[0], params[1], dst);
            break;
        case ActivationFunctionTypePrelu:
            PreluLayerForward(dst, params, conv.dstC, conv.dstH * conv.dstW, dst, TensorFormatNhwc);
            break;
        case ActivationFunctionTypeElu:
            CpuElu(dst, dSize, params[0], dst);
            break;
        case ActivationFunctionTypeHswish:
            CpuHswish(dst, dSize, params[0], params[1], dst);
            break;
        case ActivationFunctionTypeMish:
            CpuMish(dst, dSize, params[0], dst);
            break;
        case ActivationFunctionTypeHardSigmoid:
            CpuHardSigmoid(dst, dSize, params[0], params[1], dst);
            break;
        case ActivationFunctionTypeSwish:
            CpuSwish(dst, dSize, dst);
            break;
        case ActivationFunctionTypeGelu:
            CpuGelu(dst, dSize, dst);
            break;
        default:
            assert(0);
        }
    }

    void MergedConvolution8iLayer::Init()
    {
        const AlgParam& a = this->_alg;
        const ConvParam* c = a.conv;
        const ConvParam& b = c[a.count - 1];

        Stat * statS = this->Stats(0)[0];
        statS->Init8u(_method);
        _srcCvt.Init(1, c[0].srcC, c[0].srcH, c[0].srcW, TensorFormatNhwc, statS->scale32fTo8u.data(), statS->shift32fTo8u.data(), _method);

        Stat * statI = this->Stats(1).empty() ? NULL : this->Stats(1).back();
        if (statI)
        {
            if (_method == QuantizationMethodIECompatible || _method == QuantizationMethodUnifiedNarrowed)
                statI->Unify();
            statI->Init8u(_method);
            _intCvt.Init(1, b.srcC, b.srcH, b.srcW, TensorFormatNhwc, statI->scale32fTo8u.data(), statI->shift32fTo8u.data(), _method);
        }

        Stat * statD = this->Stats(2)[0];
        statD->Init8u(_method);
        _dstCvt.Init(1, b.dstC, b.dstH, b.dstW, TensorFormatNhwc, statD->scale32fTo8u.data(), statD->shift32fTo8u.data(), _method);

        if (a.IsDc())
            Quantize(1, *statI, 0);
        else
        {
            Quantize(0, *statS, 0);
            if (a.IsCdc())
                Quantize(2, *statI, 1);
        }

        switch (a.conv[_dw0 ? 0 : 1].activation)
        {
        case ActivationFunctionTypeIdentity: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeIdentity>; break;
        case ActivationFunctionTypeRelu: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeRelu>; break;
        case ActivationFunctionTypeLeakyRelu: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeLeakyRelu>; break;
        case ActivationFunctionTypeRestrictRange: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeRestrictRange>; break;
        case ActivationFunctionTypePrelu: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypePrelu>; break;
        case ActivationFunctionTypeElu: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeElu>; break;
        case ActivationFunctionTypeHswish: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeHswish>; break;
        case ActivationFunctionTypeMish: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeMish>; break;
        case ActivationFunctionTypeHardSigmoid: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeHardSigmoid>; break;
        case ActivationFunctionTypeSwish: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeSwish>; break;
        case ActivationFunctionTypeGelu: _depthwise = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeGelu>; break;
        default: assert(0);
        }
    }

    void MergedConvolution8iLayer::Quantize(size_t srcIdx, const Stat & stat, size_t dstIdx)
    {
        const AlgParam& a = this->_alg;
        const ConvParam & conv = a.conv[srcIdx];
        assert(conv.group == 1);
        const Tensor* weight = this->Weight().data() + a.index[srcIdx];
        _weight8i[dstIdx].Reshape(TensorType8i, weight[0].Shape(), TensorFormatNhwc);
        _norm32f[dstIdx].Reshape(TensorType32f, Shp(conv.dstC));
        _bias32f[dstIdx].Reshape(TensorType32f, Shp(conv.dstC));
        size_t D = conv.dstC, C = conv.srcC, K = conv.kernelY * conv.kernelX, CK = C * K;
        Floats normW(CK);
        const float* pSrcW = weight[0].Data<float>();
        const float* pSrcB = a.biasTerm[srcIdx] ? weight[1].Data<float>() : NULL;
        const float* pScale = stat.scale32fTo8u.data();
        const float* pShift = stat.shift32fTo8u.data();
        float* pNormW = normW.data();
        int8_t* pDstW = _weight8i[dstIdx].Data<int8_t>();
        float* pNorm = _norm32f[dstIdx].Data<float>();
        float* pBias = _bias32f[dstIdx].Data<float>();
        bool avoidOverflow16i = stat.negative && _method == QuantizationMethodIECompatible;
        for (size_t d = 0; d < conv.dstC; ++d)
        {
            float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
            for (size_t k = 0, kc = 0; k < K; ++k)
            {
                for (size_t c = 0; c < C; ++c, ++kc)
                {
                    pNormW[kc] = pSrcW[kc * D + d] / pScale[c];
                    minW = Min(minW, pNormW[kc]);
                    maxW = Max(maxW, pNormW[kc]);
                }
            }
            scale = stat.iMax / Max(Abs(maxW), Abs(minW));
            for (size_t k = 0, kc = 0; k < K; ++k)
            {
                for (size_t c = 0; c < C; ++c, ++kc)
                {
                    int w = ConvertTo8i(pNormW[kc], scale, 0, stat.iMin, stat.iMax);
                    if (avoidOverflow16i)
                    {
                        if (w & 1)
                            w = Round(w * 0.25f) * 4;
                        pDstW[kc * D + d] = w / 2;
                    }
                    else
                        pDstW[kc * D + d] = w;
                    normB -= w * pShift[c];
                }
            }
            pNorm[d] = (avoidOverflow16i ? 2.0f : 1.0f) / scale;
            pBias[d] = (pSrcB ? pSrcB[d] : 0.0f) + normB / scale;
        }
    }
}