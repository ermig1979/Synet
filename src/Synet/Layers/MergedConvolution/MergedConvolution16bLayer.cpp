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

#include "Synet/Layers/MergedConvolution/MergedConvolution16bLayer.h"

namespace Synet
{
    MergedConvolution16bLayer::MergedConvolution16bLayer(const LayerParam & param, Context* context)
        : MergedConvolutionLayer(param, context)
    {
    }

    size_t MergedConvolution16bLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _mergedConvolution16b.InternalBufferSize();
    }

    LowPrecisionType MergedConvolution16bLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    String MergedConvolution16bLayer::InternalInfo() const
    {
        std::stringstream info;
        info << " bf16-" << (_src16b ? "b" : "f") << (_dst16b ? "b" : "f");
        if (_alg.count == 3)
            info << _alg.add;
        if (_mergedConvolution16b.Enable())
            info << " " << _mergedConvolution16b.Info();
        return info.str();
    }

    bool MergedConvolution16bLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        if ((src->GetType() != TensorType32f && src->GetType() != TensorType16b) ||
            (dst->GetType() != TensorType32f && dst->GetType() != TensorType16b))
            SYNET_ERROR("MergedConvolution16bLayer supports only FP32 or BF16 input and output!");
        _src16b = src->GetType() == TensorType16b;
        _dst16b = dst->GetType() == TensorType16b;

        AlgParam& a = this->_alg;
        if (a.conv[1].group != 1)
        {
            a.conv[0].dstT = TensorType32f;
            a.conv[1].srcT = TensorType32f;
        }

        const ConvParam& back = a.conv[a.count - 1];
        dst->Reshape(dst->GetType(), Shp(a.batch, back.dstH, back.dstW, back.dstC), src->Format());

        _mergedConvolution16b.Init(a.batch, a.conv, a.count, a.add);
        if (_mergedConvolution16b.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_mergedConvolution16b.ExternalBufferSize()));
            _mergedConvolution16b.SetParams(a.weight, a.internal, a.bias, a.params);
        }
        else
            SYNET_ERROR("MergedConvolution16bLayer can't create SimdSynetMergedConvolution16b backend!");

        return true;
    }

    void MergedConvolution16bLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if(_mergedConvolution16b.Enable())
            _mergedConvolution16b.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
    }
}