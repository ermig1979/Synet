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
#include "Synet/Layers/Deconvolution/Deconvolution16bLayer.h"

namespace Synet
{
    Deconvolution16bLayer::Deconvolution16bLayer(const LayerParam & param, Context* context)
        : DeconvolutionLayer(param, context)
    {
    }

    LowPrecisionType Deconvolution16bLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    size_t Deconvolution16bLayer::MemoryUsage() const
    {
        return DeconvolutionLayer::MemoryUsage() + _deconvolution16b.InternalBufferSize();
    }

    String Deconvolution16bLayer::InternalInfo() const
    {
        std::stringstream info;
        info << " bf16-" << (_src16b ? "b" : "f") << (_dst16b ? "b" : "f");
        if (_deconvolution16b.Enable())
            info << " " << _deconvolution16b.Info();
        return info.str();
    }

    bool Deconvolution16bLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        const Tensors& weight = this->Weight();
        const ConvParam& conv = this->_conv;
        if ((src->GetType() != TensorType32f && src->GetType() != TensorType16b) ||
            (dst->GetType() != TensorType32f && dst->GetType() != TensorType16b))
            SYNET_ERROR("Deconvolution16bLayer supports only FP32 or BF16 input and output!");

        _src16b = src->GetType() == TensorType16b;
        _dst16b = dst->GetType() == TensorType16b;

        Shape shape = conv.DstShape(_num);
        if (_dst16b)
            dst->Reshape(TensorType16b, shape, src->Format());
        else
            dst->Reshape(TensorType32f, shape, src->Format());
        _deconvolution16b.Init(_num, &conv);
        if (_deconvolution16b.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_deconvolution16b.ExternalBufferSize()), src->Format());
            _deconvolution16b.SetParams(weight[0].Data<float>(), _biasTerm ? weight[1].Data<float>() : NULL,
                conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _params);
            _internal = 1;
        }
        else
            SYNET_ERROR("Deconvolution16bLayer can't create SimdSynetDeconvolution16b backend!");
        return true;
    }

    void Deconvolution16bLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_deconvolution16b.Enable())
            _deconvolution16b.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
    }
}