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

#include "Synet/Layers/ScaledDotProductAttentionLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Utils/Permute.h"
#include "Synet/Utils/Gemm.h"

namespace Synet
{
    ScaledDotProductAttentionLayer::ScaledDotProductAttentionLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
        _simdPermute = std::make_shared<SimdPermute>();
    }

    bool ScaledDotProductAttentionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("ScaledDotProductAttentionLayer supports only 3 inputs and 1 output!");
        if(src[0]->Shape() != src[1]->Shape() || src[0]->Shape() != src[2]->Shape())
            SYNET_ERROR("ScaledDotProductAttentionLayer inputs must have the same shape!");
        if (src[0]->GetType() != TensorType32f || src[1]->GetType() != TensorType32f || src[2]->GetType() != TensorType32f)
            SYNET_ERROR("ScaledDotProductAttentionLayer inputs must have FP32 type!");
        if (src[0]->Format() != TensorFormatNchw)
            SYNET_ERROR("ScaledDotProductAttentionLayer supports only NCHW!");

        _batch = src[0]->Size(0, -2);
        _prev = src[0]->Axis(-2);
        _last = src[0]->Axis(-1);
        _scale = 1.0f / sqrt(sqrt((float)_last));
        _fast = false;

        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), dst[0]->Format());

        size_t size = _prev * _last * 2 + _prev * _prev;
        Base::Extend32f(buf, 0, Shp(size), src[0]->Format());

        _simdPermute->Init(Shp(_prev, _last), Shp(1, 0), TensorType32f);

        std::stringstream desc;
        desc << _batch << "-" << _prev << "-" << _last << (_fast ? "-f" : "-p");
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    int64_t ScaledDotProductAttentionLayer::Flop() const
    {
        return _batch * _prev * _last * (_last + _prev) * 2;
    }

    void ScaledDotProductAttentionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const float* query = src[0]->Data<float>();
        const float* key = src[1]->Data<float>();
        const float* value = src[2]->Data<float>();
        float *buf0 = Base::Buf32f(buf, 0);
        float *dst0 = dst[0]->Data<float>();
        for (size_t b = 0, o = 0; b < _batch; ++b, o += _prev * _last)
            Attention(query + o, key + o, value + o, buf0, dst0 + o);
    }

    void ScaledDotProductAttentionLayer::Attention(const float* query, const float* key, const float* value, float* buf, float* dst)
    {
        float* kbuf = buf, * abuf = kbuf + _prev * _last, * qbuf = _fast ? (float*)query : abuf + _prev * _prev;
        _simdPermute->Forward(key, kbuf);
        if (_fast)
            CpuGemm(CblasNoTrans, CblasNoTrans, _prev, _prev, _last, _scale*_scale, kbuf, _last, qbuf, _prev, 0.0f, abuf, _prev);
        else
        {
            ScaleForward32f(kbuf, &_scale, NULL, 1, _prev, _last, kbuf, TensorFormatNchw, 1);
            ScaleForward32f(query, &_scale, NULL, 1, _prev, _last, qbuf, TensorFormatNchw, 1);
            CpuGemm(CblasNoTrans, CblasNoTrans, _prev, _prev, _last, 1.0f, qbuf, _last, kbuf, _prev, 0.0f, abuf, _prev);
        }
        SoftmaxLayerForward(abuf, _prev, _prev, 1, abuf);
        CpuGemm(CblasNoTrans, CblasNoTrans, _prev, _last, _prev, 1.0f, abuf, _prev, value, _last, 0.0f, dst, _last);
    }
}