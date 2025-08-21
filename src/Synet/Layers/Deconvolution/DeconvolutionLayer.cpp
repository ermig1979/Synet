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

#include "Synet/Layers/Deconvolution/DeconvolutionLayer.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    DeconvolutionLayer::DeconvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        _internal = 0;
    }

    int64_t DeconvolutionLayer::Flop() const
    {
        return _num * _conv.kernelY * _conv.kernelX * _conv.srcC * _conv.srcH * _conv.srcW * _conv.dstC / _conv.group * 2 +
            _num * _conv.dstC * _conv.dstH * _conv.dstW * ((_biasTerm ? 1 : 0) + _conv.ActivalionFlop());
    }

    void DeconvolutionLayer::CompactWeight()
    {
        if (_internal)
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool DeconvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("DeconvolutionLayer supports only 1 input and 1 output!");

        const ConvolutionParam & param = this->Param().convolution();
        const Tensors & weight = this->Weight();

        if(!_conv.Set(param))
            SYNET_ERROR("DeconvolutionLayer has invalid parameters!");
        if (!_conv.Set(*src[0], *dst[0], false, false))
            return false;

        _is1x1 = _conv.Is1x1();
        _biasTerm = param.biasTerm();
        if (_biasTerm)
        {
            if(weight.size() < 2 || weight[1].Size() != _conv.dstC)
                SYNET_ERROR("DeconvolutionLayer has invalid weight[1] parameter!");
        }

        _params[0] = param.activationParam0();
        _params[1] = param.activationParam1();

        if(weight.size() != 1 + _biasTerm + (_conv.activation == ActivationFunctionTypePrelu))
            SYNET_ERROR("DeconvolutionLayer has invalid weight number!");
        if (_conv.activation == ActivationFunctionTypePrelu)
        {
            if (weight.back().Size() == 1)
            {
                _conv.activation = ActivationFunctionTypeLeakyRelu;
                _params[0] = weight.back().Data<float>()[0];
            }
            else
            {
                if(weight.back().Size() != _conv.dstC)
                    SYNET_ERROR("DeconvolutionLayer has invalid last weight size!");
            }
        }

        _axis = src[0]->Index(param.axis());
        if(src[0]->Count() != _axis + 3)
            SYNET_ERROR("DeconvolutionLayer has invalid input shape for given axis " << param.axis() << " parameter!");

        _num = src[0]->Size(0, _axis);
        _trans = src[0]->Format() == TensorFormatNhwc;
        if(weight[0].Shape() != _conv.WeightShape(_trans != 0, false) || weight[0].Format() != src[0]->Format())
            SYNET_ERROR("DeconvolutionLayer has invalid input weigth[0] size " << ToStr(weight[0].Shape()) << " instead of " << ToStr(_conv.WeightShape(_trans != 0, false)) << " !");

        if (!Reshape(src[0], buf, dst[0]))
            return false;

        _srcSize = src[0]->Size(_axis);
        _dstSize = dst[0]->Size(_axis);

        std::stringstream desc;
        desc << _num << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
        desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
        desc << "-" << Max(_conv.strideY, _conv.strideX) << "-" << _conv.group;
        desc << InternalInfo();
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }
}