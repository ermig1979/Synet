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

#include "Synet/Layers/Quantized/QuantizedInnerProductLayer.h"

namespace Synet
{
    QuantizedInnerProductLayer::QuantizedInnerProductLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool QuantizedInnerProductLayer::Resizable() const
    {
        return false;
    }

    int64_t QuantizedInnerProductLayer::Flop() const
    {
        return _batch * _M * _N * (_K * 2 + (_biasTerm ? 1 : 0));
    }

    bool QuantizedInnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("QuantizedInnerProductLayer supports only 1 or 2 inputs and 1 output!");

        const InnerProductParam& param = this->Param().innerProduct();

        _biasTerm = param.biasTerm();
        _transA = param.transposeA();
        _transB = param.transposeB();
        _axis = (int)src[0]->Index(param.axis());
        _batch = 1;
        _K = src[0]->Size(_axis);

        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = dst[0]->GetType() == TensorType8u;

        if (src.size() == 2)
        {
            SYNET_ERROR("QuantizedInnerProductLayer 2 inputs support is not implemented!");
        }
        else
        {
            _N = this->Param().innerProduct().outputNum();
            _M = src[0]->Size(0, _axis);
            _batch = 1;
        }

        if (!(Compartible() && InitParams()))
            return false;

        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(TensorType32f, dstShape, TensorFormatNchw);

        std::stringstream desc;
        desc << _batch << "x" << _M << "x" << _K << "-" << _N << " ";
        desc << ToChar(src[0]->GetType()) << (src.size() > 1 ? ToChar(src[1]->GetType()) : "0") << ToChar(dst[0]->GetType()) << "-";
        desc << (_transB ? "n" : "t") << (_biasTerm ? "b" : "o");
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    bool QuantizedInnerProductLayer::Compartible() const
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();

        if (param.qSrc().size() < 2)
            SYNET_ERROR("QuantizedInnerProductLayer must have at least 2 input dequantizers!");
        if (param.qSrc()[0].weights() != 0)
            SYNET_ERROR("QuantizedInnerProductLayer supports only uniform input quantization!");
        if (param.qSrc()[1].weights() < 2 || weight.size() < param.qSrc()[1].weights())
            SYNET_ERROR("QuantizedInnerProductLayer: check weight or dequantizers!");
        if (weight[0].GetType() != TensorType8i)
            SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
        bool weightZeroZero = true;
        if (param.qSrc()[1].weights() == 2)
        {
            if (param.qSrc()[1].type() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
            weightZeroZero = param.qSrc()[1].zero() == 0;
        }
        else
        {
            if (weight[2].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
            for (size_t i = 0, n = weight[2].Size(); i < n && weightZeroZero; ++i)
                weightZeroZero = weight[2].Data<int8_t>()[i] == 0;
        }
        if (!weightZeroZero)
            SYNET_ERROR("QuantizedInnerProductLayer supports only weight 'zero' == 0!");

        if (_biasTerm)
        {
            if (param.qSrc().size() != 3)
                SYNET_ERROR("QuantizedInnerProductLayer must have 3 input dequantizers for when uses bias!");
            int biasStart = param.qSrc()[1].weights();
            bool biasZeroZero = true;
            if (param.qSrc()[2].weights() == 2)
            {
                if (param.qSrc()[2].type() != TensorType32i)
                    SYNET_ERROR("QuantizedInnerProductLayer supports only INT32 bias!");
                biasZeroZero = param.qSrc()[2].zero() == 0;
            }
            else
            {
                if (weight[biasStart + 2].GetType() != TensorType32i)
                    SYNET_ERROR("QuantizedInnerProductLayer supports only INT32 bias!");
                for (size_t i = 0, n = weight[biasStart + 2].Size(); i < n && biasZeroZero; ++i)
                    biasZeroZero = weight[biasStart + 2].Data<int32_t>()[i] == 0;
            }
            if (!biasZeroZero)
                SYNET_ERROR("QuantizedInnerProductLayer supports only bias 'zero' == 0!");

            bool equalScale = true;
            if (weight[0].Count() != 2 || weight[0].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer: weight[0] must be 2D int8 tensor!");
            if (weight[1].Count() != 1 || weight[biasStart + 1].Count() != 1 || weight[1].Axis(0) != weight[biasStart + 1].Axis(0))
                SYNET_ERROR("QuantizedInnerProductLayer: weight scale (weight[1]) must the same size as bias scale (weight[" << biasStart + 1 << "]) !");
            float srcScale = param.qSrc()[0].scale();
            for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
            {
                if (::fabs(weight[1].Data<float>()[i] * srcScale - weight[biasStart + 1].Data<float>()[i]) > 0.000001)
                    SYNET_ERROR("QuantizedInnerProductLayer: weight scale (weight[1]) and bias scale (weight[" << biasStart + 1 << "]) are not compartible!");
            }
        }

        return true;
    }

    bool QuantizedInnerProductLayer::InitParams()
    {
        return true;
    }

    void QuantizedInnerProductLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}