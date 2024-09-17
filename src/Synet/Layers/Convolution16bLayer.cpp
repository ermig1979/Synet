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
#include "Synet/Layers/Convolution16bLayer.h"

namespace Synet
{
    Convolution16bLayer::Convolution16bLayer(const LayerParam & param, Context* context)
        : ConvolutionLayer(param, context)
    {
    }

    LowPrecisionType Convolution16bLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    size_t Convolution16bLayer::MemoryUsage() const
    {
        return ConvolutionLayer::MemoryUsage() + _convolution16b.InternalBufferSize();
    }

    String Convolution16bLayer::InternalInfo() const
    {
        std::stringstream info;
        info << " bf16-" << (_src16b ? "b" : "f") << (_dst16b ? "b" : "f");
        if (_convolution16b.Enable())
            info << " " << _convolution16b.Info();
        return info.str();
    }

    bool Convolution16bLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        const Tensors& weight = this->Weight();
        const ConvParam& conv = this->_conv;
        AlgParam & alg = this->_alg;
        if ((src->GetType() != TensorType32f && src->GetType() != TensorType16b) ||
            (dst->GetType() != TensorType32f && dst->GetType() != TensorType16b))
            SYNET_ERROR("Convolution16bLayer supports only FP32 or BF16 input and output!");

        _src16b = src->GetType() == TensorType16b;
        _dst16b = dst->GetType() == TensorType16b;

        Shape shape = conv.DstShape(alg.batch);
        if (_dst16b)
            dst->Reshape(TensorType16b, shape, src->Format());
        else
            dst->Reshape(TensorType32f, shape, src->Format());
        _convolution16b.Init(alg.batch, &conv);
        if (_convolution16b.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_convolution16b.ExternalBufferSize()), src->Format());
            _convolution16b.SetParams(weight[0].Data<float>(), alg.bias ? weight[1].Data<float>() : NULL,
                conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : alg.params);
            alg.internal = 1;
        }
        else
            SYNET_ERROR("Convolution16bLayer can't create SimdSynetConvolution16b backend!");
        return true;
    }

    void Convolution16bLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_convolution16b.Enable())
            _convolution16b.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
    }
}