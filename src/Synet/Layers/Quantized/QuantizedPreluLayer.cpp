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

#include "Synet/Layers/Quantized/QuantizedPreluLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    SYNET_INLINE void QuantizedPrelu(const uint8_t& src, int sBias, float sNorm, float slope, uint8_t& dst, float dNorm, int dZero)
    {
        float _src = DequantizeLinear(src, sBias, sNorm);
        float _dst = std::max(0.0f, _src) + slope * std::min(_src, 0.0f);
        dst = (uint8_t)QuantizeLinear(_dst, dNorm, dZero, 0, 255);
    }

    static void QuantizedPreluLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstScale, int dstZero, TensorFormat format)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetQuantizedPreluLayerForward(src, srcScale, srcZero, channels, spatial, slope, dst, dstScale, dstZero, (SimdTensorFormatType)format);
#else
        float sBias = -srcZero;
        float sNorm = srcScale[0], dNorm = 1.0f / dstScale[0];
        if (format == SimdTensorFormatNhwc)
        {
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < channels; ++c)
                    QuantizedPrelu(src[c], sBias, sNorm, slope[c], dst[c], dNorm, dstZero);
                src += channels;
                dst += channels;

            }
        }
        else
        {
            for (size_t c = 0; c < channels; ++c)
            {
                float _slope = slope[c];
                size_t s = 0;
                for (; s < spatial; ++s)
                    QuantizedPrelu(src[s], sBias, sNorm, _slope, dst[s], dNorm, dstZero);
                src += spatial;
                dst += spatial;
            }
        }
#endif
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedPreluLayer::QuantizedPreluLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t QuantizedPreluLayer::Flop() const
    {
        if (_const)
            return 0;
        return _batch * _channels * _spatial * 8;
    }

    LowPrecisionType QuantizedPreluLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedPreluLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedPreluLayer supports 1 input and 1 output!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("QuantizedPreluLayer supports only 4D input tensor!");
        if (src[0]->GetType() != TensorType8u)
            SYNET_ERROR("QuantizedPreluLayer supports only UINT8 input and output!");

        _format = src[0]->Format();
        const Shape& shape = src[0]->Shape();
        _batch = shape[0];
        if (_format == TensorFormatNhwc)
        {
            _channels = shape[3];
            _spatial = shape[1] * shape[2];
        }
        else
        {
            _channels = shape[1];
            _spatial = shape[2] * shape[3];
        }

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 1 || param.qDst().size() != 1)
            SYNET_ERROR("QuantizedPreluLayer must have 1 input and 1 output quantization parameters!");
        if (param.qSrc()[0].weights() != 0 || param.qDst()[0].weights() != 0)
            SYNET_ERROR("QuantizedPreluLayer supports only uniform quantization!");

        _srcScale = float(param.qSrc()[0].scale());
        _srcZero = param.qSrc()[0].zero();
        _dstScale = float(param.qDst()[0].scale());
        _dstZero = param.qDst()[0].zero();

        if (this->Weight().empty())
            SYNET_ERROR("QuantizedPreluLayer weights are absent!");
        _slope = this->Weight()[0].Data<float>();

        dst[0]->Reshape(src[0]->GetType(), shape, _format);

        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }

        return true;
    }

    void QuantizedPreluLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const uint8_t* src0 = src[0]->Data<uint8_t>();
        uint8_t* dst0 = dst[0]->Data<uint8_t>();
        for (size_t b = 0; b < _batch; ++b)
        {
            QuantizedPreluLayerForward(src0, &_srcScale, _srcZero, _channels, _spatial, _slope, dst0, &_dstScale, _dstZero, _format);
            src0 += _channels * _spatial;
            dst0 += _channels * _spatial;
        }
    }
}