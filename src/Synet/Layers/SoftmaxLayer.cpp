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

#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    void SoftmaxLayerForward32f(const float * src, size_t outer, size_t count, size_t inner, float * dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetSoftmax32f(src, outer, count, inner, dst);
#else
        Tensor<float> _buffer(TensorType32f, Shp(inner), TensorFormatUnknown);
        float * buffer = _buffer.Data<float>();
        for (size_t o = 0; o < outer; ++o)
        {
            Synet::CpuCopy(src, inner, buffer);
            const float* s = src + inner;
            for (size_t i = 1; i < count; ++i)
            {
                Synet::CpuMax(s, buffer, inner, buffer);
                s += inner;
            }

            s = src;
            float * d = dst;
            for (size_t i = 0; i < count; ++i)
            {
                Synet::CpuSub(s, buffer, inner, d);
                s += inner;
                d += inner;
            }

            Synet::CpuExp(dst, count*inner, dst);

            Synet::CpuCopy(dst, inner, buffer);
            d = dst + inner;
            for (size_t i = 1; i < count; ++i)
            {
                Synet::CpuAdd(d, buffer, inner, buffer);
                d += inner;
            }

            d = dst;
            for (size_t i = 0; i < count; ++i)
            {
                Synet::CpuDiv(d, buffer, inner, d);
                d += inner;
            }
            src += count*inner;
            dst += count*inner;
        }
#endif    
    }

    void SoftmaxLayerForward16b(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetSoftmax16b(src, outer, count, inner, dst);
#else
        assert(0);
#endif    
    }

    //-------------------------------------------------------------------------------------------------

    void LogSoftmaxLayerForward(const float* src, size_t outer, size_t count, size_t inner, float* dst)
    {
        Tensor<float> _buffer(TensorType32f, Shp(inner * 2), TensorFormatUnknown);
        float* max = _buffer.Data<float>(), * sum = max + inner;
        for (size_t o = 0; o < outer; ++o)
        {
            Synet::CpuCopy(src, inner, max);
            const float* s = src + inner;
            for (size_t i = 1; i < count; ++i)
            {
                Synet::CpuMax(s, max, inner, max);
                s += inner;
            }

            s = src;
            float* d = dst;
            for (size_t i = 0; i < count; ++i)
            {
                Synet::CpuSub(s, max, inner, d);
                s += inner;
                d += inner;
            }

            Synet::CpuExp(dst, count * inner, dst);

            Synet::CpuCopy(dst, inner, sum);
            d = dst + inner;
            for (size_t i = 1; i < count; ++i)
            {
                Synet::CpuAdd(d, sum, inner, sum);
                d += inner;
            }

            Synet::CpuLog(sum, inner, sum);
            Synet::CpuAdd(max, sum, inner, sum);

            s = src;
            d = dst;
            for (size_t i = 0; i < count; ++i)
            {
                Synet::CpuSub(s, sum, inner, d);
                s += inner;
                d += inner;
            }
            src += count * inner;
            dst += count * inner;
        }
    }

    //-------------------------------------------------------------------------------------------------

    SoftmaxLayer::SoftmaxLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType SoftmaxLayer::LowPrecision(TensorType type) const
    {
        const SoftmaxParam& param = this->Param().softmax();
        if (type == TensorType16b && param.log() == false)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    bool SoftmaxLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SoftmaxLayer supports only 1 input and 1 output!");
        _type = src[0]->GetType();
        if (_type != TensorType32f && _type != TensorType16b)
            SYNET_ERROR("SoftmaxLayer has unsupported input types!");

        const SoftmaxParam& softmax = this->Param().softmax();
        _axis = src[0]->Index(softmax.axis());
        _log = softmax.log();
        if(_axis >= src[0]->Count())
            SYNET_ERROR("SoftmaxLayer has wrong axis " << softmax.axis() << " parameter!");

        if (_type == TensorType16b && _log)
            SYNET_ERROR("SoftmaxLayer unsupports log softmax with BF16 input type!");

        _outer = src[0]->Size(0, _axis);
        _count = src[0]->Axis(_axis);
        _inner = src[0]->Size(_axis + 1);
        dst[0]->Reshape(_type, src[0]->Shape(), src[0]->Format());
        std::stringstream desc;
        desc << _log << ToChar(_type);
        this->UsePerfStat(desc.str());
        return true;
    }

    int64_t SoftmaxLayer::Flop() const
    {
        return _inner * _outer * _count * 24;
    }

    void SoftmaxLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if(_log)
            LogSoftmaxLayerForward(src[0]->Data<float>(), _outer, _count, _inner, dst[0]->Data<float>());
        else
        {
            if (_type == TensorType32f)
                SoftmaxLayerForward32f(src[0]->Data<float>(), _outer, _count, _inner, dst[0]->Data<float>());
            else if (_type == TensorType16b)
                SoftmaxLayerForward16b(src[0]->Data<uint16_t>(), _outer, _count, _inner, dst[0]->Data<uint16_t>());
            else
                assert(0);
        }
    }
}