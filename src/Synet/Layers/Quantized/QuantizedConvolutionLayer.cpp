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
        bool zeroZero = true;
        if (param.qSrc()[1].weights() == 2)
        {
            if(param.qSrc()[1].type() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
            zeroZero = param.qSrc()[1].zero() == 0;
        }
        else
        {
            if (weight[2].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedConvolutionLayer supports only INT8 weight!");
            for (size_t i = 0, n = weight[2].Size(); i < n && zeroZero; ++i)
                zeroZero = weight[2].Data<int8_t>()[i] == 0;
        }
        if(!zeroZero)
            SYNET_ERROR("QuantizedConvolutionLayer supports only weight 'zero' == 0!");

        if (_alg.bias)
        {
            if (param.qSrc().size() != 3)
                SYNET_ERROR("QuantizedConvolutionLayer must have 3 input dequantizers for when uses bias!");
            int biasStart = param.qSrc()[1].weights();

            bool equalScale = true;
        }

        return true;
    }

    bool QuantizedConvolutionLayer::InitParams()
    {

        return true;
    }

    void QuantizedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}