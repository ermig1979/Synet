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

namespace Synet
{
    QuantizedConvolutionLayer::QuantizedConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    size_t QuantizedConvolutionLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _weight8i.MemoryUsage() + _norm32f.MemoryUsage() + _bias32i.MemoryUsage();
    }

    int64_t QuantizedConvolutionLayer::Flop() const
    {
        return _alg.batch * _conv.dstC * _conv.dstH * _conv.dstW *
            (_conv.kernelY * _conv.kernelX * _conv.srcC / _conv.group * 2 + _alg.bias + _conv.ActivalionFlop());
    }

    bool QuantizedConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedConvolutionLayer supports only 1 input and 1 output!");

        const ConvolutionParam& param = this->Param().convolution();

        _conv.Set(param);
        _conv.Set(*src[0], *dst[0], true, param.autoPad());

        if (src[0]->Count() != 4)
            SYNET_ERROR("QuantizedConvolutionLayer supports only 4D tensors!");

        _alg.params[0] = param.activationParam0();
        _alg.params[1] = param.activationParam1();

        _alg.is1x1 = _conv.Is1x1() ? 1 : 0;
        _alg.bias = param.biasTerm() ? 1 : 0;
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

        if (!(Compartible() && InitParams()))
            return false;

        dst[0]->Reshape(TensorType32f, _conv.DstShape(_alg.batch), src[0]->Format());

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
            if (param.qSrc()[2].weights() == 2)
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

            bool equalScale = true;
            if (weight[0].Count() != 4 || weight[0].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer: weight[0] must be 4D int8 tensor!");
            if (weight[1].Count() != 1 || weight[biasStart + 1].Count() != 1 || weight[1].Axis(0) != weight[biasStart + 1].Axis(0))
                SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) must the same size as bias scale (weight[" << biasStart + 1 << "]) !");
            float srcScale = param.qSrc()[0].scale();
            for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
            {
                if(::fabs(weight[1].Data<float>()[i] * srcScale - weight[biasStart + 1].Data<float>()[i]) > 0.000001)
                    SYNET_ERROR("QuantizedConvolutionLayer: weight scale (weight[1]) and bias scale (weight[" << biasStart + 1 << "]) are not compartible!");
            }
        }

        return true;
    }

    bool QuantizedConvolutionLayer::InitParams()
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();
        _bias32i.Reshape(TensorType32i, Shp(weight[1].Size()), TensorFormatNchw, int32_t(0));
        int srcZero = param.qSrc()[0].zero();
        if (weight[0].Format() == TensorFormatNhwc)
        {
            SYNET_ERROR("QuantizedConvolutionLayer: unsupported weight[0] format: " << weight[0].Format() << " !");
        }
        else
        {
            size_t M = weight[0].Size(0, 1), K = weight[0].Size(1, 4);
            const int8_t* pw = weight[0].Data<int8_t>();
            int32_t* pdb = _bias32i.Data<int32_t>();
            for (size_t i = 0; i < M; ++i)
            {
                pdb[i] = 0;
                for (size_t k = 0; k < K; ++k)
                    pdb[i] -= pw[i * K + k] * srcZero;
            }
            if(_alg.bias)
            {
                int biasStart = param.qSrc()[1].weights();
                const int32_t* psb = weight[biasStart].Data<int32_t>();
                for (size_t i = 0; i < M; ++i)
                    pdb[i] += psb[i];
            }
        }
        _norm32f.Reshape(TensorType32f, Shp(weight[1].Size()), TensorFormatNchw, float(0));
        float srcScale = param.qSrc()[0].scale();
        for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
            _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale;
        return true;
    }

    void QuantizedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}