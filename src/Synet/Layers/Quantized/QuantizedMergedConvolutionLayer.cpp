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

#include "Synet/Layers/Quantized/QuantizedMergedConvolutionLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Quantization/Gemm.h"
#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    QuantizedMergedConvolutionLayer::QuantizedMergedConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    size_t QuantizedMergedConvolutionLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage();
    }

    int64_t QuantizedMergedConvolutionLayer::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _count; ++i)
            flop += _batch * _conv[i].dstC * _conv[i].dstH * _conv[i].dstW *
            (_conv[i].kernelY * _conv[i].kernelX * _conv[i].srcC / _conv[i].group * 2 + (_bias[i] ? 1 : 0) + _conv[i].ActivalionFlop());
        return flop;
    }

    void QuantizedMergedConvolutionLayer::CompactWeight()
    {
        //if (_quantizedMergedConvolution.Enable())
        //    ((Tensor&)this->Weight()[0]).Clear();
    }

    LowPrecisionType QuantizedMergedConvolutionLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedMergedConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedConvolutionLayer supports only 1 input and 1 output!");

        const LayerParam& param = this->Param();
        //const ConvolutionParam& conv = param.convolution();

        //_conv.Set(conv);
        //_conv.Set(*src[0], *dst[0], true, conv.autoPad());
        //_src8u = src[0]->GetType() == TensorType8u;
        //_dst8u = param.qDst().size() && param.qDst()[0].type() == TensorType8u;
        //_conv.dstT = _dst8u ? TensorType8u : TensorType32f;
        //if (src[0]->Count() != 4)
        //    SYNET_ERROR("QuantizedConvolutionLayer supports only 4D tensors!");

        //_alg.params[0] = conv.activationParam0();
        //_alg.params[1] = conv.activationParam1();

        //_alg.is1x1 = _conv.Is1x1() ? 1 : 0;
        //_alg.bias = conv.biasTerm() ? 1 : 0;
        //_alg.batch = src[0]->Axis(0);
        //_alg.trans = _conv.Trans();
        //if (_alg.trans)
        //{
        //    _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
        //    _alg.ldW = _conv.dstC;
        //    _alg.grW = _conv.dstC / _conv.group;

        //    _alg.siS = _conv.dstH * _conv.dstW;
        //    _alg.ldS = _alg.siW * (_alg.is1x1 ? _conv.group : 1);
        //    _alg.grS = _alg.siW * (_alg.is1x1 ? 1 : _alg.siS);

        //    _alg.siD = _conv.dstC / _conv.group;
        //    _alg.ldD = _conv.dstC;
        //    _alg.grD = _alg.siD;
        //}
        //else
        //{
        //    _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
        //    _alg.ldW = _alg.siW;
        //    _alg.grW = _conv.dstC * _alg.siW / _conv.group;

        //    _alg.siS = _conv.dstH * _conv.dstW;
        //    _alg.ldS = _alg.siS;
        //    _alg.grS = _alg.siS * _alg.siW;

        //    _alg.siD = _conv.dstC / _conv.group;
        //    _alg.ldD = _conv.dstH * _conv.dstW;
        //    _alg.grD = _alg.siD * _alg.siS;
        //}

        //dst[0]->Reshape(_dst8u ? TensorType8u: TensorType32f, _conv.DstShape(_alg.batch), src[0]->Format());

        //_alg.sSize = src[0]->Size(1);
        //_alg.dSize = dst[0]->Size(1);

        //_quantizedMergedConvolution.Init(_alg.batch, &_conv);
        //if (_quantizedMergedConvolution.Enable())
        //{
        //    Layer::Extend8u(buf, 0, Shp(_quantizedMergedConvolution.ExternalBufferSize()));
        //    const Tensors& weight = this->Weight();
        //    int bias = param.qSrc()[1].weights();
        //    const float* params = _conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _alg.params;
        //    float srcScale = (float)param.qSrc()[0].scale(), dstScale = (float)param.qDst()[0].scale();
        //    uint8_t srcZero = (uint8_t)param.qSrc()[0].zero(), dstZero = (uint8_t)param.qDst()[0].zero();
        //    _quantizedConvolution.SetParams(&srcScale, &srcZero, weight[0].Data<int8_t>(), weight[1].Data<float>(), 
        //        _alg.bias ? weight[bias + 0].Data<int32_t>() : NULL, params, &dstScale, &dstZero);
        //}
        //else
        {
            if (!(Compartible() && InitParams()))
                return false;
            //if (!_src8u)
            //    Layer::Extend8u(buf, 0, _conv.SrcShape(1));
            //if (!_conv.Is1x1())
            //    Layer::Extend8u(buf, 1, Shp(_conv.ImgSize()));
            //Layer::Extend32i(buf, 0, _conv.DstShape(1));
            //if (_dst8u)
            //    Layer::Extend32f(buf, 0, _conv.DstShape(1));
        }

        std::stringstream desc;
        desc << _count << ": " << _batch << "x" << _conv[0].srcC << "x" << _conv[0].srcH << "x" << _conv[0].srcW;
        for (size_t i = 0; i < _count; ++i)
            desc << "-" << (_conv[i].IsDepthwise() ? String("") : Cpl::ToStr(_conv[i].dstC) + "x") << _conv[i].kernelY << "x" << _conv[i].strideY;

        //if(_quantizedMergedConvolution.Enable())
        //    desc << " " << _quantizedMergedConvolution.Info();
        this->UsePerfStat(desc.str(), Flop()); 

        return true;
    }

    bool QuantizedMergedConvolutionLayer::Compartible() const
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();

        //if(param.qSrc().size() < 2)
        //    SYNET_ERROR("QuantizedConvolutionLayer must have at least 2 input dequantizers!");
        //if (param.qSrc()[0].weights() != 0)
        //    SYNET_ERROR("QuantizedConvolutionLayer supports only uniform input quantization!");
        //if (param.qSrc()[1].weights() < 2 || weight.size() < param.qSrc()[1].weights())
        //    SYNET_ERROR("QuantizedConvolutionLayer: check weight or dequantizers!");
        //if (weight[0].GetType() != TensorType8i)
        //    SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
        //bool weightZeroZero = true;
        //if (param.qSrc()[1].weights() == 2)
        //{
        //    if(param.qSrc()[1].type() != TensorType8i)
        //        SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
        //    weightZeroZero = param.qSrc()[1].zero() == 0;
        //}
        //else
        //{
        //    if (weight[2].GetType() != TensorType8i)
        //        SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
        //    for (size_t i = 0, n = weight[2].Size(); i < n && weightZeroZero; ++i)
        //        weightZeroZero = weight[2].Data<int8_t>()[i] == 0;
        //}
        //if(!weightZeroZero)
        //    SYNET_ERROR("QuantizedConvolutionLayer supports only weight 'zero' == 0!");

        //if (_alg.bias)
        //{
        //    if (param.qSrc().size() != 3)
        //        SYNET_ERROR("QuantizedConvolutionLayer must have 3 input dequantizers for when uses bias!");
        //    int biasStart = param.qSrc()[1].weights();
        //    bool biasZeroZero = true;
        //    if (param.qSrc()[2].weights() < 3)
        //    {
        //        if (param.qSrc()[2].type() != TensorType32i)
        //            SYNET_ERROR("QuantizedConvolutionLayer supports only INT32 bias!");
        //        biasZeroZero = param.qSrc()[2].zero() == 0;
        //    }
        //    else
        //    {
        //        if (weight[biasStart + 2].GetType() != TensorType32i)
        //            SYNET_ERROR("QuantizedConvolutionLayer supports only INT32 bias!");
        //        for (size_t i = 0, n = weight[biasStart + 2].Size(); i < n && biasZeroZero; ++i)
        //            biasZeroZero = weight[biasStart + 2].Data<int32_t>()[i] == 0;
        //    }
        //    if (!biasZeroZero)
        //        SYNET_ERROR("QuantizedConvolutionLayer supports only bias 'zero' == 0!");

        //    if (weight[0].Count() != 4 || weight[0].GetType() != TensorType8i)
        //        SYNET_ERROR("QuantizedConvolutionLayer: weight[0] must be 4D int8 tensor!");
        //    if (param.qSrc()[2].weights() > 1)
        //    {
        //        bool equalScale = true;
        //        if (weight[1].Count() != 1 || weight[biasStart + 1].Count() != 1 || weight[1].Axis(0) != weight[biasStart + 1].Axis(0))
        //            SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) must the same size as bias scale (weight[" << biasStart + 1 << "]) !");
        //        float srcScale = (float)param.qSrc()[0].scale();
        //        for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
        //        {
        //            if (::fabs(weight[1].Data<float>()[i] * srcScale - weight[biasStart + 1].Data<float>()[i]) > 0.000001)
        //                SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) and bias scale (weight[" << biasStart + 1 << "]) are not compartible!");
        //        }
        //    }
        //}

        if (param.qDst().size())
        {
            if (param.qDst()[0].weights() != 0)
                SYNET_ERROR("QuantizedMergedConvolutionLayer supports only uniform output quantization!");
        }

        return true;
    }

    bool QuantizedMergedConvolutionLayer::InitParams()
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();
        //int srcZero = param.qSrc()[0].zero();
        //_srcZero8u.Reshape(TensorType8u, Shp(_conv.srcC), TensorFormatNchw, uint8_t(srcZero));
        //_bias32i.Reshape(TensorType32i, Shp(weight[1].Size()), TensorFormatNchw, int32_t(0));
        //if (weight[0].Format() == TensorFormatNhwc)
        //{
        //    size_t K = weight[0].Size(0, 3), M = weight[0].Size(3, 4);
        //    const int8_t* pw = weight[0].Data<int8_t>();
        //    int32_t* pb = _bias32i.Data<int32_t>();
        //    for (size_t i = 0; i < M; ++i)
        //    {
        //        pb[i] = 0;
        //        for (size_t k = 0; k < K; ++k)
        //            pb[i] -= pw[k * M + i] * srcZero;
        //    }
        //    if (_alg.bias)
        //    {
        //        int biasStart = param.qSrc()[1].weights();
        //        const int32_t* pw = weight[biasStart + 0].Data<int32_t>();
        //        int32_t* pb = _bias32i.Data<int32_t>();
        //        for (size_t i = 0; i < M; ++i)
        //            pb[i] += pw[i];
        //    }
        //}
        //else
        //{
        //    size_t M = weight[0].Size(0, 1), K = weight[0].Size(1, 4);
        //    const int8_t* pw = weight[0].Data<int8_t>();
        //    int32_t* pb = _bias32i.Data<int32_t>();
        //    for (size_t i = 0; i < M; ++i)
        //    {
        //        pb[i] = 0;
        //        for (size_t k = 0; k < K; ++k)
        //            pb[i] -= pw[i * K + k] * srcZero;
        //    }
        //    if(_alg.bias)
        //    {
        //        int biasStart = param.qSrc()[1].weights();
        //        const int32_t* pw = weight[biasStart + 0].Data<int32_t>();
        //        int32_t* pb = _bias32i.Data<int32_t>();
        //        for (size_t i = 0; i < M; ++i)
        //            pb[i] += pw[i];
        //    }
        //}
        //_norm32f.Reshape(TensorType32f, Shp(weight[1].Size()), TensorFormatNchw, float(0));
        //if (param.qDst().size())
        //{
        //    int dstZero = param.qDst()[0].zero();
        //    float dstScale = (float)param.qDst()[0].scale();
        //    _dstZero8u.Reshape(TensorType8u, Shp(_conv.dstC), TensorFormatNchw, uint8_t(dstZero));
        //    if (_alg.bias && param.qSrc()[2].weights() > 1)
        //    {
        //        int biasStart = param.qSrc()[1].weights();
        //        for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
        //            _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i] / dstScale;
        //    }
        //    else
        //    {
        //        float srcScale = (float)param.qSrc()[0].scale();
        //        for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
        //            _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale / dstScale;
        //    }
        //}
        //else
        //{
        //    if (_alg.bias && param.qSrc()[2].weights() > 1)
        //    {
        //        int biasStart = param.qSrc()[1].weights();
        //        for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
        //            _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i];
        //    }
        //    else
        //    {
        //        float srcScale = (float)param.qSrc()[0].scale();
        //        for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
        //            _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale;
        //    }
        //}
        return true;
    }

    void QuantizedMergedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {

    }
}