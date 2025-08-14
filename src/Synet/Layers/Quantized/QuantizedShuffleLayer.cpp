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

#include "Synet/Layers/Quantized/QuantizedShuffleLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    static void QuantizedShuffleLayerForwardCpu(const uint8_t* src0, int bias0, float norm0, size_t srcC0, const uint8_t* src1, int bias1, float norm1, size_t srcC1, 
        size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero, TensorFormat format, int shuffleType)
    {
        size_t dstC = (srcC0 + srcC1) / 2;
        switch (shuffleType)
        {
        case 0:
            if (format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t cd = 0;
                    for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
                    {
                        dst0[cd] = DequantizeQuantizeLinear(src0[cs + 0], bias0, norm0, scale, zero, 0, 255);
                        dst1[cd] = DequantizeQuantizeLinear(src0[cs + 1], bias0, norm0, scale, zero, 0, 255);
                    }
                    for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
                    {
                        dst0[cd] = DequantizeQuantizeLinear(src1[cs + 0], bias1, norm1, scale, zero, 0, 255);
                        dst1[cd] = DequantizeQuantizeLinear(src1[cs + 1], bias1, norm1, scale, zero, 0, 255);
                    }
                    src0 += srcC0;
                    src1 += srcC1;
                    dst0 += dstC;
                    dst1 += dstC;
                }
            }
            else
            {
                size_t cd = 0;
                for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst0[s] = DequantizeQuantizeLinear(src0[s], bias0, norm0, scale, zero, 0, 255);
                    src0 += spatial;
                    dst0 += spatial;
                    for (size_t s = 0; s < spatial; ++s)
                        dst1[s] = DequantizeQuantizeLinear(src0[s], bias0, norm0, scale, zero, 0, 255);
                    src0 += spatial;
                    dst1 += spatial;
                }
                for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst0[s] = DequantizeQuantizeLinear(src1[s], bias1, norm1, scale, zero, 0, 255);
                    src1 += spatial;
                    dst0 += spatial;
                    for (size_t s = 0; s < spatial; ++s)
                        dst1[s] = DequantizeQuantizeLinear(src1[s], bias1, norm1, scale, zero, 0, 255);
                    src1 += spatial;
                    dst1 += spatial;
                }
            }
            break;
        case 1:
            if (format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t cs = 0;
                    for (size_t cd = 0; cd < srcC0; cd += 2, cs += 1)
                    {
                        dst0[cd + 0] = DequantizeQuantizeLinear(src0[cs], bias0, norm0, scale, zero, 0, 255);
                        dst0[cd + 1] = DequantizeQuantizeLinear(src1[cs], bias1, norm1, scale, zero, 0, 255);
                    }
                    for (size_t cd = 0; cd < srcC1; cd += 2, cs += 1)
                    {
                        dst1[cd + 0] = DequantizeQuantizeLinear(src0[cs], bias0, norm0, scale, zero, 0, 255);
                        dst1[cd + 1] = DequantizeQuantizeLinear(src1[cs], bias1, norm1, scale, zero, 0, 255);
                    }
                    src0 += dstC;
                    src1 += dstC;
                    dst0 += srcC0;
                    dst1 += srcC1;
                }
            }
            else
            {
                size_t cs = 0;
                for (size_t cd = 0; cd < srcC0; cs += 1, cd += 2)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst0[s] = DequantizeQuantizeLinear(src0[s], bias0, norm0, scale, zero, 0, 255);
                    src0 += spatial;
                    dst0 += spatial;
                    for (size_t s = 0; s < spatial; ++s)
                        dst0[s] = DequantizeQuantizeLinear(src1[s], bias1, norm1, scale, zero, 0, 255);
                    src1 += spatial;
                    dst0 += spatial;
                }
                for (size_t cd = 0; cd < srcC1; cs += 1, cd += 2)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst1[s] = DequantizeQuantizeLinear(src0[s], bias0, norm0, scale, zero, 0, 255);
                    src0 += spatial;
                    dst1 += spatial;
                    for (size_t s = 0; s < spatial; ++s)
                        dst1[s] = DequantizeQuantizeLinear(src1[s], bias1, norm1, scale, zero, 0, 255);
                    src1 += spatial;
                    dst1 += spatial;
                }
            }
            break;
        }
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedShuffleLayer::QuantizedShuffleLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t QuantizedShuffleLayer::Flop() const
    {
        if (_const)
            return 0;
        return _batch * _dstC * _spatial * 4;
    }

    bool QuantizedShuffleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 2)
            SYNET_ERROR("QuantizedShuffleLayer supports 2 inputs and 2 outputs!");
        if (src[0]->Count() != 4 || src[1]->Count() != 4)
            SYNET_ERROR("QuantizedShuffleLayer supports only 4D input tensors!");
        if (src[0]->Format() != src[1]->Format())
            SYNET_ERROR("QuantizedShuffleLayer inputs must have the same format!");
        if (src[0]->GetType() != TensorType8u || src[1]->GetType() != TensorType8u)
            SYNET_ERROR("ShuffleLayer supports only UINT8 inputs and outputs!");

        _format = src[0]->Format();
        const Shape& srcShape0 = src[0]->Shape();
        const Shape& srcShape1 = src[1]->Shape();
        _batch = srcShape0[0];
        if (srcShape0[0] != srcShape1[0])
            SYNET_ERROR("QuantizedShuffleLayer inputs must have the same shape[0]!");
        Shape dstShape = srcShape0;
        _shuffleType = this->Param().shuffle().type();
        if (_shuffleType != 0 && _shuffleType != 1)
            SYNET_ERROR("QuantizedShuffleLayer parameter shuffle.type() must be 1 or 2!");
        if (_format == TensorFormatNhwc)
        {
            _srcC0 = srcShape0[3];
            _srcC1 = srcShape1[3];
            _dstC = (_srcC0 + _srcC1) / 2;
            if (_srcC0 + _srcC1 != _dstC * 2)
                SYNET_ERROR("QuantizedShuffleLayer: check input channel dims!");
            dstShape[3] = _dstC;
            _spatial = srcShape0[1] * srcShape0[2];
            if (srcShape0[1] != srcShape1[1] || srcShape0[2] != srcShape1[2])
                SYNET_ERROR("QuantizedShuffleLayer: check input spatial dims!");
        }
        else
        {
            _srcC0 = srcShape0[1];
            _srcC1 = srcShape1[1];
            _dstC = (_srcC0 + _srcC1) / 2;
            if (_srcC0 + _srcC1 != _dstC * 2)
                SYNET_ERROR("QuantizedShuffleLayer: check input channel dims!");
            dstShape[1] = _dstC;
            _spatial = srcShape0[2] * srcShape0[3];
            if (srcShape0[2] != srcShape1[2] || srcShape0[3] != srcShape1[3])
                SYNET_ERROR("QuantizedShuffleLayer: check input spatial dims!");
        }

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 2 || param.qDst().size() != 1)
            SYNET_ERROR("QuantizedShuffleLayer must have 2 inputs and 1 output quantization parameters!");

        _bias0 = -param.qSrc()[0].zero();
        _norm0 = float(param.qSrc()[0].scale());
        _bias1 = -param.qSrc()[1].zero();
        _norm1 = float(param.qSrc()[1].scale());
        _zero = param.qDst()[0].zero();
        _scale = 1.0f / float(param.qDst()[0].scale());

        dst[0]->Reshape(src[0]->GetType(), dstShape, _format);
        dst[1]->Reshape(src[1]->GetType(), dstShape, _format);

        if (src[0]->Const() && src[1]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            dst[1]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }

        return true;
    }

    void QuantizedShuffleLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        QuantizedShuffleLayerForwardCpu(src[0]->Data<uint8_t>(), _bias0, _norm0, _srcC0, src[1]->Data<uint8_t>(), _bias1, _norm1, _srcC1,
            _spatial, src[0]->Data<uint8_t>(), src[1]->Data<uint8_t>(), _scale, _zero, _format, _shuffleType);
    }
}