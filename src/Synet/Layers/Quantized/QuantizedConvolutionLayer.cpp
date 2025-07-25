/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/QuantizedConvolutionLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Quantization/Gemm.h"
#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    QuantizedConvolutionLayer::QuantizedConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    size_t QuantizedConvolutionLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _norm32f.MemoryUsage() + _bias32i.MemoryUsage() + 
            _quantizedConvolution.InternalBufferSize();
    }

    int64_t QuantizedConvolutionLayer::Flop() const
    {
        return _alg.batch * _conv.dstC * _conv.dstH * _conv.dstW *
            (_conv.kernelY * _conv.kernelX * _conv.srcC / _conv.group * 2 + _alg.bias + _conv.ActivalionFlop());
    }

    void QuantizedConvolutionLayer::CompactWeight()
    {
        if (_quantizedConvolution.Enable())
            ((Tensor&)this->Weight()[0]).Clear();
    }

    LowPrecisionType QuantizedConvolutionLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedConvolutionLayer supports only 1 input and 1 output!");

        const LayerParam& param = this->Param();
        const ConvolutionParam& conv = param.convolution();

        _conv.Set(conv);
        _conv.Set(*src[0], *dst[0], true, conv.autoPad());
        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = param.qDst().size() && param.qDst()[0].type() == TensorType8u;
        _conv.dstT = _dst8u ? TensorType8u : TensorType32f;
        if (src[0]->Count() != 4)
            SYNET_ERROR("QuantizedConvolutionLayer supports only 4D tensors!");

        _alg.params[0] = conv.activationParam0();
        _alg.params[1] = conv.activationParam1();

        _alg.is1x1 = _conv.Is1x1() ? 1 : 0;
        _alg.bias = conv.biasTerm() ? 1 : 0;
        _alg.batch = src[0]->Axis(0);
        _alg.trans = _conv.Trans();
        if (_alg.trans)
        {
            _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
            _alg.ldW = _conv.dstC;
            _alg.grW = _conv.dstC / _conv.group;

            _alg.siS = _conv.dstH * _conv.dstW;
            _alg.ldS = _alg.siW * (_alg.is1x1 ? _conv.group : 1);
            _alg.grS = _alg.siW * (_alg.is1x1 ? 1 : _alg.siS);

            _alg.siD = _conv.dstC / _conv.group;
            _alg.ldD = _conv.dstC;
            _alg.grD = _alg.siD;
        }
        else
        {
            _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
            _alg.ldW = _alg.siW;
            _alg.grW = _conv.dstC * _alg.siW / _conv.group;

            _alg.siS = _conv.dstH * _conv.dstW;
            _alg.ldS = _alg.siS;
            _alg.grS = _alg.siS * _alg.siW;

            _alg.siD = _conv.dstC / _conv.group;
            _alg.ldD = _conv.dstH * _conv.dstW;
            _alg.grD = _alg.siD * _alg.siS;
        }

        dst[0]->Reshape(_dst8u ? TensorType8u: TensorType32f, _conv.DstShape(_alg.batch), src[0]->Format());

        _alg.sSize = src[0]->Size(1);
        _alg.dSize = dst[0]->Size(1);

        _quantizedConvolution.Init(_alg.batch, &_conv);
        if (_quantizedConvolution.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_quantizedConvolution.ExternalBufferSize()));
            const Tensors& weight = this->Weight();
            int bias = param.qSrc()[1].weights();
            const float* params = _conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _alg.params;
            float srcScale = (float)param.qSrc()[0].scale(), dstScale = (float)param.qDst()[0].scale();
            uint8_t srcZero = (uint8_t)param.qSrc()[0].zero(), dstZero = (uint8_t)param.qDst()[0].zero();
            _quantizedConvolution.SetParams(&srcScale, &srcZero, weight[0].Data<int8_t>(), weight[1].Data<float>(), 
                _alg.bias ? weight[bias + 0].Data<int32_t>() : NULL, params, &dstScale, &dstZero);
        }
        else
        {
            if (!(Compartible() && InitParams()))
                return false;
            if (!_src8u)
                Layer::Extend8u(buf, 0, _conv.SrcShape(1));
            if (!_conv.Is1x1())
                Layer::Extend8u(buf, 1, Shp(_conv.ImgSize()));
            Layer::Extend32i(buf, 0, _conv.DstShape(1));
            if (_dst8u)
                Layer::Extend32f(buf, 0, _conv.DstShape(1));
        }

        std::stringstream desc;
        desc << _alg.batch << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
        desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
        desc << "-" << Max(_conv.dilationY, _conv.dilationX) << "-" << Max(_conv.strideY, _conv.strideX);
        desc << "-" << _conv.group;
        if(_quantizedConvolution.Enable())
            desc << " " << _quantizedConvolution.Info();
        this->UsePerfStat(desc.str(), Flop()); 

        return true;
    }

    bool QuantizedConvolutionLayer::Compartible() const
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();

        if(param.qSrc().size() < 2)
            SYNET_ERROR("QuantizedConvolutionLayer must have at least 2 input dequantizers!");
        if (param.qSrc()[0].weights() != 0)
            SYNET_ERROR("QuantizedConvolutionLayer supports only uniform input quantization!");
        if (param.qSrc()[1].weights() < 2 || weight.size() < param.qSrc()[1].weights())
            SYNET_ERROR("QuantizedConvolutionLayer: check weight or dequantizers!");
        if (weight[0].GetType() != TensorType8i)
            SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
        bool weightZeroZero = true;
        if (param.qSrc()[1].weights() == 2)
        {
            if(param.qSrc()[1].type() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
            weightZeroZero = param.qSrc()[1].zero() == 0;
        }
        else
        {
            if (weight[2].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
            for (size_t i = 0, n = weight[2].Size(); i < n && weightZeroZero; ++i)
                weightZeroZero = weight[2].Data<int8_t>()[i] == 0;
        }
        if(!weightZeroZero)
            SYNET_ERROR("QuantizedConvolutionLayer supports only weight 'zero' == 0!");

        if (_alg.bias)
        {
            if (param.qSrc().size() != 3)
                SYNET_ERROR("QuantizedConvolutionLayer must have 3 input dequantizers for when uses bias!");
            int biasStart = param.qSrc()[1].weights();
            bool biasZeroZero = true;
            if (param.qSrc()[2].weights() < 3)
            {
                if (param.qSrc()[2].type() != TensorType32i)
                    SYNET_ERROR("QuantizedConvolutionLayer supports only INT32 bias!");
                biasZeroZero = param.qSrc()[2].zero() == 0;
            }
            else
            {
                if (weight[biasStart + 2].GetType() != TensorType32i)
                    SYNET_ERROR("QuantizedConvolutionLayer supports only INT32 bias!");
                for (size_t i = 0, n = weight[biasStart + 2].Size(); i < n && biasZeroZero; ++i)
                    biasZeroZero = weight[biasStart + 2].Data<int32_t>()[i] == 0;
            }
            if (!biasZeroZero)
                SYNET_ERROR("QuantizedConvolutionLayer supports only bias 'zero' == 0!");

            if (weight[0].Count() != 4 || weight[0].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer: weight[0] must be 4D int8 tensor!");
            if (param.qSrc()[2].weights() > 1)
            {
                bool equalScale = true;
                if (weight[1].Count() != 1 || weight[biasStart + 1].Count() != 1 || weight[1].Axis(0) != weight[biasStart + 1].Axis(0))
                    SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) must the same size as bias scale (weight[" << biasStart + 1 << "]) !");
                float srcScale = (float)param.qSrc()[0].scale();
                for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
                {
                    if (::fabs(weight[1].Data<float>()[i] * srcScale - weight[biasStart + 1].Data<float>()[i]) > 0.000001)
                        SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) and bias scale (weight[" << biasStart + 1 << "]) are not compartible!");
                }
            }
        }

        if (param.qDst().size())
        {
            if (param.qDst()[0].weights() != 0)
                SYNET_ERROR("QuantizedConvolutionLayer supports only uniform output quantization!");
        }

        return true;
    }

    bool QuantizedConvolutionLayer::InitParams()
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();
        int srcZero = param.qSrc()[0].zero();
        _srcZero8u.Reshape(TensorType8u, Shp(_conv.srcC), TensorFormatNchw, uint8_t(srcZero));
        _bias32i.Reshape(TensorType32i, Shp(weight[1].Size()), TensorFormatNchw, int32_t(0));
        if (weight[0].Format() == TensorFormatNhwc)
        {
            size_t K = weight[0].Size(0, 3), M = weight[0].Size(3, 4);
            const int8_t* pw = weight[0].Data<int8_t>();
            int32_t* pb = _bias32i.Data<int32_t>();
            for (size_t i = 0; i < M; ++i)
            {
                pb[i] = 0;
                for (size_t k = 0; k < K; ++k)
                    pb[i] -= pw[k * M + i] * srcZero;
            }
            if (_alg.bias)
            {
                int biasStart = param.qSrc()[1].weights();
                const int32_t* pw = weight[biasStart + 0].Data<int32_t>();
                int32_t* pb = _bias32i.Data<int32_t>();
                for (size_t i = 0; i < M; ++i)
                    pb[i] += pw[i];
            }
        }
        else
        {
            size_t M = weight[0].Size(0, 1), K = weight[0].Size(1, 4);
            const int8_t* pw = weight[0].Data<int8_t>();
            int32_t* pb = _bias32i.Data<int32_t>();
            for (size_t i = 0; i < M; ++i)
            {
                pb[i] = 0;
                for (size_t k = 0; k < K; ++k)
                    pb[i] -= pw[i * K + k] * srcZero;
            }
            if(_alg.bias)
            {
                int biasStart = param.qSrc()[1].weights();
                const int32_t* pw = weight[biasStart + 0].Data<int32_t>();
                int32_t* pb = _bias32i.Data<int32_t>();
                for (size_t i = 0; i < M; ++i)
                    pb[i] += pw[i];
            }
        }
        _norm32f.Reshape(TensorType32f, Shp(weight[1].Size()), TensorFormatNchw, float(0));
        if (param.qDst().size())
        {
            int dstZero = param.qDst()[0].zero();
            float dstScale = (float)param.qDst()[0].scale();
            _dstZero8u.Reshape(TensorType8u, Shp(_conv.dstC), TensorFormatNchw, uint8_t(dstZero));
            if (_alg.bias && param.qSrc()[2].weights() > 1)
            {
                int biasStart = param.qSrc()[1].weights();
                for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i] / dstScale;
            }
            else
            {
                float srcScale = (float)param.qSrc()[0].scale();
                for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale / dstScale;
            }
        }
        else
        {
            if (_alg.bias && param.qSrc()[2].weights() > 1)
            {
                int biasStart = param.qSrc()[1].weights();
                for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i];
            }
            else
            {
                float srcScale = (float)param.qSrc()[0].scale();
                for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale;
            }
        }
        return true;
    }

    void QuantizedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_quantizedConvolution.Enable())
        {
            _quantizedConvolution.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
            return;
        }
        const AlgParam& alg = this->_alg;
        const float* src32f = _src8u ? NULL : src[0]->Data<float>();
        uint8_t* src8u = _src8u ? src[0]->Data<uint8_t>() : Layer::Buf8u(buf, 0);
        uint8_t* buf8u = Layer::Buf8u(buf, 1);
        int32_t* sum32i = Layer::Buf32i(buf, 0);
        float* dst32f = _dst8u ? Layer::Buf32f(buf, 0) : dst[0]->Data<float>();
        uint8_t* dst8u = _dst8u ? dst[0]->Data<uint8_t>() : NULL;
        for (size_t b = 0; b < alg.batch; ++b)
        {
            if (!_src8u)
            {
                //_srcCvt.Convert(src32f, src8u);
                src32f += alg.sSize;
            }
            Convolution(src8u, buf8u, sum32i);
            if (_src8u)
                src8u += alg.sSize;
            if (_dst8u)
            {
                PostProcess(sum32i, dst32f, dst8u);
                dst8u += alg.dSize;
            }
            else
            {
                PostProcess(sum32i, dst32f);
                dst32f += alg.dSize;
            }
        }
    }

    void QuantizedConvolutionLayer::Convolution(const uint8_t* src, uint8_t* buf, int32_t* sum)
    {
        const bool overflow16i = true;
        const int8_t* weight = Weight()[0].Data<int8_t>();
        const uint8_t* tmp = src;
        if (!_alg.is1x1)
        {
            const uint8_t* zero = _srcZero8u.Data<uint8_t>();
            if (_alg.trans)
                Synet::ImgToRow(tmp, _conv.srcH, _conv.srcW, _conv.srcC, _conv.kernelY, _conv.kernelX,
                    _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, _conv.group, zero, buf);
            else
                Synet::ImgToCol(tmp, _conv.srcC, _conv.srcH, _conv.srcW, _conv.kernelY, _conv.kernelX,
                    _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, zero, buf);
            tmp = buf;
        }
        if (_alg.trans)
        {
            assert(_conv.group == 1 || _conv.group == _conv.srcC);
            if (_conv.group == 1)
                Synet::CpuGemm8iNN(_alg.siS, _alg.siD, _conv.kernelY * _conv.kernelX, _conv.srcC, tmp, _alg.ldS, weight, _alg.ldW, sum, _alg.ldD, overflow16i);
            else
                for (size_t g = 0; g < _conv.group; ++g)
                    Synet::CpuGemmNN(_alg.siS, _alg.siD, _alg.siW, tmp + _alg.grS * g, _alg.ldS, weight + _alg.grW * g, _alg.ldW, sum + _alg.grD * g, _alg.ldD);
        }
        else
        {
            if (_conv.group == 1)
                Synet::CpuGemm8iNN(_alg.siD, _alg.siS, _conv.srcC, _conv.kernelY * _conv.kernelX, weight, _alg.ldW, tmp, _alg.ldS, sum, _alg.ldD, overflow16i);
            else
                for (size_t g = 0; g < _conv.group; ++g)
                    Synet::CpuGemmNN(_alg.siD, _alg.siS, _alg.siW, weight + _alg.grW * g, _alg.ldW, tmp + _alg.grS * g, _alg.ldS, sum + _alg.grD * g, _alg.ldD);
        }
    }

    void QuantizedConvolutionLayer::PostProcess(const int32_t* sum, float* dst)
    {
        const float* norm = _norm32f.Data<float>();
        const int32_t* bias = _bias32i.Data<int32_t>();
        DequantizeLinear(sum, 1, _conv.dstC, _conv.dstH, _conv.dstW, _conv.dstF, bias, norm, dst);
        switch (_conv.activation)
        {
        case ActivationFunctionTypeIdentity:
            break;
        case ActivationFunctionTypeRelu:
            CpuRelu(dst, _alg.dSize, 0.0f, dst);
            break;
        case ActivationFunctionTypeLeakyRelu:
            CpuRelu(dst, _alg.dSize, _alg.params[0], dst);
            break;
        case ActivationFunctionTypeRestrictRange:
            CpuRestrictRange(dst, _alg.dSize, _alg.params[0], _alg.params[1], dst);
            break;
        //case ActivationFunctionTypePrelu:
        //    PreluLayerForward(dst, this->Weight().back().Data<float>(), conv.dstC, conv.dstH * conv.dstW, dst, alg.trans ? TensorFormatNhwc : TensorFormatNchw);
        //    break;
        case ActivationFunctionTypeElu:
            CpuElu(dst, _alg.dSize, _alg.params[0], dst);
            break;
        case ActivationFunctionTypeHswish:
            CpuHswish(dst, _alg.dSize, _alg.params[0], _alg.params[1], dst);
            break;
        case ActivationFunctionTypeMish:
            CpuMish(dst, _alg.dSize, _alg.params[0], dst);
            break;
        case ActivationFunctionTypeHardSigmoid:
            CpuHardSigmoid(dst, _alg.dSize, _alg.params[0], _alg.params[1], dst);
            break;
        case ActivationFunctionTypeSwish:
            CpuSwish(dst, _alg.dSize, dst);
            break;
        case ActivationFunctionTypeGelu:
            CpuGelu(dst, _alg.dSize, dst);
            break;
        default:
            assert(0);
        }
    }

    void QuantizedConvolutionLayer::PostProcess(const int32_t* sum, float* buf, uint8_t* dst)
    {
        const float* norm = _norm32f.Data<float>();
        const int32_t* bias = _bias32i.Data<int32_t>();
        const uint8_t* zero = _dstZero8u.Data<uint8_t>();
        if (_conv.activation == ActivationFunctionTypeIdentity || _conv.activation == ActivationFunctionTypeRelu)
        {
            QuantizeSumLinear(sum, 1, _conv.dstC, _conv.dstH, _conv.dstW, _conv.dstF, bias, norm, zero, dst);
        }
        else
        {
            assert(0);
        }
    }
}