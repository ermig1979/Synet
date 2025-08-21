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

#include "Synet/Layers/InnerProduct/InnerProduct16bLayer.h"

namespace Synet
{
    InnerProduct16bLayer::InnerProduct16bLayer(const LayerParam & param, Context* context)
        : InnerProductLayer(param, context)
    {
    }

    LowPrecisionType InnerProduct16bLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    size_t InnerProduct16bLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _innerProduct16b.InternalBufferSize();
    }

    void InnerProduct16bLayer::CompactWeight()
    {
        if(this->Weight().size())
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool InnerProduct16bLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (!InnerProductLayer::Reshape(src, buf, dst))
            return false;
        if ((src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType16b) ||
            (src.size() > 1 && src[1]->GetType() != TensorType32f && src[1]->GetType() != TensorType16b) ||
            (dst[0]->GetType() != TensorType32f && dst[0]->GetType() != TensorType16b))
            SYNET_ERROR("InnerProduct16bLayer supports only FP32 or BF16 input and output!");
        if(_transA)
            SYNET_ERROR("InnerProduct16bLayer does not support transposed A matrix!");
        _sizeA = _M * _K * GetTensorTypeSize(src[0]->GetType());
        _sizeB = src.size() > 1 ? _K * _N * GetTensorTypeSize(src[1]->GetType()) : 0;
        _sizeC = _M * _N * GetTensorTypeSize(dst[0]->GetType());
        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(dst[0]->GetType(), dstShape, TensorFormatNchw);
        _innerProduct16b.Init(_M, _N, _K, src[0]->GetType(), src.size() > 1 ? src[1]->GetType() : TensorType32f,
            dst[0]->GetType(), _transB ? 0 : 1, src.size() == 1 ? 1 : 0, _biasTerm ? 1 : 0);
        if(!_innerProduct16b.Enable())
            SYNET_ERROR("InnerProduct16bLayer can't create SimdSynetInnerProduct16b backend!");
        if (src.size() == 1)
        {
            const float* weight = this->Weight()[0].Data<float>();
            const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
            _innerProduct16b.SetParams(weight, bias);
        }
        Layer::Extend8u(buf, 0, Shp(_innerProduct16b.ExternalBufferSize()), src[0]->Format());
        this->UsePerfStat(_desc + " " + _innerProduct16b.Info(), Flop());
        return true;
    }

    void InnerProduct16bLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_innerProduct16b.Enable())
        {
            const uint8_t* A = src[0]->RawData();
            const uint8_t* B = src.size() > 1 ? src[1]->RawData() : NULL;
            uint8_t* C = dst[0]->RawData();
            for (size_t b = 0; b < _batch; ++b)
            {
                _innerProduct16b.Forward(A, B, Layer::Buf8u(buf, 0), C);
                A += _sizeA;
                B += _sizeB;
                C += _sizeC;
            }
        }
    }
}
