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

#include "Synet/Layers/Quantized/QuantizedConcatLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    static void QuantizedConcatLayerForward(size_t count, const uint8_t ** src, size_t num, const size_t *size, const int32_t *bias, const float * norm, float scale, int32_t zero, uint8_t * dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetQuantizedConcatLayerForward(count, src, num, size, bias, norm, &scale, zero, dst);
#else
        for (size_t o = 0; o < num; ++o)
        {
            for (size_t s = 0; s < count; ++s)
            {
                size_t _size = size[s];
                const uint8_t* _src = src[s] + o * _size;
                for (size_t i = 0; i < _size; ++i)
                    dst[i] = DequantizeQuantizeLinear(_src[i], bias[s], norm[s], scale, zero, 0, 255);
                dst += _size;
            }
        }
#endif   
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedConcatLayer::QuantizedConcatLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t QuantizedConcatLayer::Flop() const
    {
        if (_const)
            return 0;
        return _outputSize * _dstSize * 4;
    }

    LowPrecisionType QuantizedConcatLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedConcatLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (dst.size() != 1)
            SYNET_ERROR("QuantizedConcatLayer supports only 1 output!");
        const LayerParam& param = this->Param();
        if (param.qSrc().size() != src.size())
            SYNET_ERROR("QuantizedConcatLayer must have equal number of inputs and quantization parameters!");
        if (param.qDst().size() != 1 || param.qDst()[0].weights() != 0)
            SYNET_ERROR("QuantizedConcatLayer: wrong output quantization parameters!");

        size_t num = src.size();
        _srcSize.resize(num);
        _bias.resize(num);
        _norm.resize(num);
        size_t axis = src[0]->Index((int32_t)this->Param().concat().axis());
        _outputSize = src[0]->Size(0, axis);
        Shape dstShape = src[0]->Shape();
        dstShape[axis] = 0;
        _dstSize = 0;
        _const = true;
        for (size_t i = 0; i < num; ++i)
        {
            if(src[i]->GetType() != TensorType8u)
                SYNET_ERROR("QuantizedConcatLayer supports only UINT8 inputs!");
            if (param.qSrc()[i].weights() != 0)
                SYNET_ERROR("QuantizedConcatLayer supports only uniform quantization!");
            for (size_t a = 0; a < dstShape.size(); ++a)
                if(a != axis && dstShape[a] != src[i]->Axis(a))
                    SYNET_ERROR("QuantizedConcatLayer unsupported input shapes!");
            _srcSize[i] = src[i]->Size(axis);
            _dstSize += src[i]->Size(axis);
            dstShape[axis] += src[i]->Axis(axis);
            _bias[i] = -param.qSrc()[i].zero();
            _norm[i] = float(param.qSrc()[i].scale());
            if (!src[i]->Const())
                _const = false;
        }
        _scale = 1.0f / float(param.qDst()[0].scale());
        _zero = param.qDst()[0].zero();

        dst[0]->Reshape(TensorType8u, dstShape, src[0]->Format());
        if (_const)
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
        }
        else
        {
            this->UsePerfStat();
        }

        return true;
    }

    void QuantizedConcatLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        ByteConstPtrs _src(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            _src[i] = src[i]->Data<uint8_t>();
        QuantizedConcatLayerForward(_src.size(), _src.data(), _outputSize, _srcSize.data(), _bias.data(), _norm.data(), _scale, _zero, dst[0]->Data<uint8_t>());
    }
}