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

#include "Synet/Layers/ConvolutionLayer.h"

namespace Synet
{
    ConvolutionLayer::ConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        _alg.internal = 0;
    }

    int64_t ConvolutionLayer::Flop() const
    {
        return _alg.batch * _conv.dstC * _conv.dstH * _conv.dstW *
            (_conv.kernelY * _conv.kernelX * _conv.srcC / _conv.group * 2 + _alg.bias + _conv.ActivalionFlop());
    }

    void ConvolutionLayer::CompactWeight()
    {
        if (_alg.internal)
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool ConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const ConvolutionParam & param = this->Param().convolution();
        Tensors & weight = (Tensors&)this->Weight();

        bool is1d = src[0]->Count() == 3;
        if (is1d)
            To2D(*src[0]), To2D(weight[0]);
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ConvolutionLayer supports only 1 input and 1 output!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("ConvolutionLayer supports only 4D input tensor!");

        _conv.Set(param);
        _conv.Set(*src[0], *dst[0], true, param.autoPad());

        _alg.is1x1 = _conv.Is1x1() ? 1 : 0;
        _alg.bias = param.biasTerm() ? 1 : 0;
        if (_alg.bias)
        {
            if(weight[1].Size() != _conv.dstC)
                SYNET_ERROR("ConvolutionLayer has wrong bias size!");
        }

        _alg.params[0] = param.activationParam0();
        _alg.params[1] = param.activationParam1();

        if(weight.size() != 1 + _alg.bias + (_conv.activation == ActivationFunctionTypePrelu))
            SYNET_ERROR("ConvolutionLayer has wrong weight number!");

        if (_conv.activation == ActivationFunctionTypePrelu)
        {
            if (weight.back().Size() == 1)
            {
                _conv.activation = ActivationFunctionTypeLeakyRelu;
                _alg.params[0] = weight.back().Data<float>()[0];
            }
            else
            {
                if (weight.back().Size() != _conv.dstC)
                    SYNET_ERROR("ConvolutionLayer: check weight[" << weight.size() - 1 << "] size!");
            }
        }

        _alg.batch = src[0]->Axis(0);
        _alg.trans = _conv.Trans();
        if(weight[0].Shape() != _conv.WeightShape(_alg.trans != 0, true) || weight[0].Format() != src[0]->Format())
            SYNET_ERROR("ConvolutionLayer: check weight[0] size or format!");

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

        if (!Reshape(src[0], buf, dst[0]))
            return false;
        if (is1d)
            To1D(*dst[0]);

        _alg.sSize = src[0]->Size(1);
        _alg.dSize = dst[0]->Size(1);
        std::stringstream desc;
        desc << _alg.batch << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
        desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
        desc << "-" << Max(_conv.dilationY, _conv.dilationX) << "-" << Max(_conv.strideY, _conv.strideX);
        desc << "-" << _conv.group << InternalInfo();
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }

    void ConvolutionLayer::To2D(const Tensor& tensor)
    {
        Shape shape = tensor.Shape();
        if (shape.size() != 3)
            return;
        TensorFormat format = tensor.Format();
        switch (format)
        {
        case TensorFormatUnknown:
        case TensorFormatNchw:
            format = TensorFormatNchw;
            shape = Shp(shape[0], shape[1], shape[2], 1);
            break;
        case TensorFormatNhwc:
            shape = Shp(shape[0], shape[1], 1, shape[2]);
            break;
        default:
            assert(0);
        }
        ((Tensor&)tensor).ShareAs(tensor, shape, format);
    }

    void ConvolutionLayer::To1D(const Tensor& tensor)
    {
        Shape shape = tensor.Shape();
        if (shape.size() != 4)
            return;
        TensorFormat format = tensor.Format();
        switch (format)
        {
        case TensorFormatNchw:
            assert(shape[3] == 1);
            shape = Shp(shape[0], shape[1], shape[2]);
            break;
        case TensorFormatNhwc:
            assert(shape[2] == 1);
            shape = Shp(shape[0], shape[1], shape[3]);
            break;
        default:
            assert(0);
        }
        ((Tensor&)tensor).ShareAs(tensor, shape, format);
    }
}