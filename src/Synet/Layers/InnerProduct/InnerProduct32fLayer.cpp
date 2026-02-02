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

#include "Synet/Utils/Gemm.h"
#include "Synet/Layers/InnerProduct/InnerProduct32fLayer.h"

namespace Synet
{
    InnerProduct32fLayer::InnerProduct32fLayer(const LayerParam & param, Context* context)
        : InnerProductLayer(param, context)
        , _internal(0)
    {

    }

    size_t InnerProduct32fLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _innerProduct32f.InternalBufferSize() * sizeof(float);
    }

    void InnerProduct32fLayer::CompactWeight()
    {
        if (_internal)
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool InnerProduct32fLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (!InnerProductLayer::Reshape(src, buf, dst))
            return false;
        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(TensorType32f, dstShape, TensorFormatNchw);
        if (!_transA && src.size() == 1)
        {
            _innerProduct32f.Init(_M, _K, _N, _transB ? 0 : 1);
            if (_innerProduct32f.Enable())
            {
                const float* weight = this->Weight()[0].Data<float>();
                const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
                _innerProduct32f.SetParams(weight, &_internal, bias, NULL);
            }
        }
        return true;
    }

    void InnerProduct32fLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        const float* src0 = src[0]->Data<float>();
        float* dst0 = dst[0]->Data<float>();
        if (_innerProduct32f.Enable())
            _innerProduct32f.Forward(src0, dst0);
        else if (src.size() > 1)
        {
            const float* src1 = src[1]->Data<float>();
            for(size_t b = 0; b < _batch; ++b)
            {
                Forward(src0, src1, dst0);
                src0 += _M * _K;
                src1 += _K * _N;
                dst0 += _M * _N;
            }
        }
        else
        {
            const float* wgt = this->Weight()[0].Data<float>();
            Forward(src0, wgt, dst0);
        }
    }

    void InnerProduct32fLayer::Forward(const float * src, const float* wgt, float* dst)
    {
        const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
        if (!_transB && _M == 1)
            Detail::InnerProductLayerForwardCpu(src, wgt, bias, _N, _K, dst);
        else
        {
            size_t lds = _transA ? _M : _K;
            size_t ldw = _transB ? _N : _K;
            CpuGemm(_transA ? CblasTrans : CblasNoTrans, _transB ? CblasNoTrans : CblasTrans, _M, _N, _K, 1.0f, src, lds, wgt, ldw, 0.0f, dst, _N);
            if (_biasTerm)
            {
                for (size_t i = 0; i < _M; ++i)
                    CpuAddBias(bias, _N, 1, dst + i * _N);
            }
        }
    }
}
