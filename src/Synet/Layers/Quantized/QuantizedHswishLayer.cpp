/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/QuantizedHswishLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

#include "Synet/Utils/Activation.h"

namespace Synet
{
    SYNET_INLINE void QuantizedHswish(const uint8_t& src, int sBias, float sNorm, float shift, float scale, uint8_t& dst, float dNorm, int dZero)
    {
        float _src = DequantizeLinear(src, sBias, sNorm);
        float _dst = CpuHswish(_src, shift, scale);
        dst = (uint8_t)QuantizeLinear(_dst, dNorm, dZero, 0, 255);
    }

    static void QuantizedHswishLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t size, float shift, float scale, uint8_t* dst, const float* dstScale, int dstZero)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetQuantizedHswishLayerForward(src, srcScale, srcZero, size, &shift, &scale, dst, dstScale, dstZero);
#else
        float sBias = -srcZero;
        float sNorm = srcScale[0], dNorm = 1.0f / dstScale[0];
        for(size_t i = 0; i < size; ++i)
            QuantizedHswish(src[i], sBias, sNorm, shift, scale, dst[i], dNorm, dstZero);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedHswishLayer::QuantizedHswishLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t QuantizedHswishLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * 9;
    }

    LowPrecisionType QuantizedHswishLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedHswishLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedHswishLayer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType8u)
            SYNET_ERROR("QuantizedHswishLayer supports only UINT8 input and output!");

        _size = src[0]->Size();
        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 1 || param.qDst().size() != 1)
            SYNET_ERROR("QuantizedHswishLayer must have 1 input and 1 output quantization parameters!");
        if (param.qSrc()[0].weights() != 0 || param.qDst()[0].weights() != 0)
            SYNET_ERROR("QuantizedHswishLayer supports only uniform quantization!");

        _srcScale = float(param.qSrc()[0].scale());
        _srcZero = param.qSrc()[0].zero();
        _dstScale = float(param.qDst()[0].scale());
        _dstZero = param.qDst()[0].zero();

        HswishParam hswish = param.hswish();
        _shift = hswish.shift();
        _scale = hswish.scale();

        if (dst[0] != src[0])
        {
            if (TensorUsers(Param().src()[0]) == 1 && !src[0]->Const())
                dst[0]->Share(*src[0]);
            else
                dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }

        if (src[0]->Const())
        {
            Forward(src, buf, dst, 0);
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

    void QuantizedHswishLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        const uint8_t* src0 = src[0]->Data<uint8_t>();
        uint8_t* dst0 = dst[0]->Data<uint8_t>();
        QuantizedHswishLayerForward(src0, &_srcScale, _srcZero, _size, _shift, _scale, dst0, &_dstScale, _dstZero);
    }
}