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

#include "Synet/Layers/InnerProduct8iLayer.h"

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Quantization/Gemm.h"
#include "Synet/Quantization/Convert.h"
#include "Synet/Quantization/Const.h"
#include "Synet/Quantization/Bf16.h" 
#include "Synet/Utils/Math.h"
#include "Synet/Utils/InnerProduct.h"
#include "Synet/Utils/Gemm.h"

namespace Synet
{
    InnerProduct8iLayer::InnerProduct8iLayer(const LayerParam & param, Context* context, QuantizationMethod method)
        : InnerProductLayer(param, context)
        , _method(method)
    {
    }

    LowPrecisionType InnerProduct8iLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    size_t InnerProduct8iLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _weight8i.MemoryUsage() + _norm32i.MemoryUsage() + _norm32f.MemoryUsage();
    }

    void InnerProduct8iLayer::CompactWeight()
    {
        ((Tensor&)this->Weight()[0]).Clear();
    }

    bool InnerProduct8iLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (!InnerProductLayer::Reshape(src, buf, dst))
            return false;
        if ((src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType8u) ||
            (dst[0]->GetType() != TensorType32f && dst[0]->GetType() != TensorType8u))
            SYNET_ERROR("InnerProduct8iLayer supports only FP32 or INT8 input and output!");
        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = dst[0]->GetType() == TensorType8u;
        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(_dst8u ? TensorType8u : TensorType32f, dstShape, TensorFormatNchw);
        if (!_src8u)
            Layer::Extend8u(buf, 0, src[0]->Shape(), src[0]->Format());
        Layer::Extend32i(buf, 0, dstShape, TensorFormatNchw);
        Quantize();
        return true;
    }

    void InnerProduct8iLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        uint8_t* tmp = _src8u ? src[0]->Data<uint8_t>() : Layer::Buf8u(buf, 0);
        int32_t* sum = Layer::Buf32i(buf, 0);
        if (!_src8u)
            _srcCvt.Convert(src[0]->Data<float>(), tmp);
        ForwardCpu(tmp, sum);
        if (_dst8u)
            _dstCvt.Convert(sum, dst[0]->Data<uint8_t>());
        else
            _dstCvt.Convert(sum, dst[0]->Data<float>());
    }

    void InnerProduct8iLayer::Quantize()
    {
        Stat& statS = *this->Stats(0)[0];
        Stat& statD = *this->Stats(2)[0];
        statS.Init8u(_method);
        statD.Init8u(_method);
        _weight8i.Reshape(TensorType8i, this->Weight()[0].Shape(), TensorFormatNchw);
        _norm32i.Reshape(TensorType32i, Shp(2, _N));
        _norm32f.Reshape(TensorType32f, Shp(2, _N));
        Floats normW(_K);
        const float* pSrcW = this->Weight()[0].Data<float>();
        const float* pSrcB = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
        const float* pSrcScaleInv = statS.scale8uTo32f.data();
        const float* pSrcScale = statS.scale32fTo8u.data();
        const float* pSrcShift = statS.shift32fTo8u.data();
        const float* pDstScale = statD.scale8uTo32f.data();
        const float* pDstScaleInv = statD.scale32fTo8u.data();
        const float* pDstShift = statD.shift8uTo32f.data();
        float* pNormW = normW.data();
        int8_t* pDstW = _weight8i.Data<int8_t>();
        int32_t* pDstS = _norm32i.Data<int32_t>();
        int32_t* pDstB = pDstS + _N;
        float* pNormScale = _norm32f.Data<float>();
        float* pNormShift = pNormScale + _N;
        int wLo, wUp, sLo, sUp;
        bool avoidOverflow16i = statS.negative && _method == QuantizationMethodIECompatible;
        if (_method == QuantizationMethodIECompatible)
            wLo = QUANT_IE_COMP_WEIGHT_MIN, wUp = QUANT_IE_COMP_WEIGHT_MAX, sLo = QUANT_IE_COMP_SRC_U8_MIN, sUp = QUANT_IE_COMP_SRC_U8_MAX;
        else if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
            wLo = QUANT_SYMM_NARR_WEIGHT_MIN, wUp = QUANT_SYMM_NARR_WEIGHT_MAX, sLo = QUANT_SYMM_NARR_SRC_U8_MIN, sUp = QUANT_SYMM_NARR_SRC_U8_MAX;
        _srcCvt.Init(_M, _K, 1, 1, TensorFormatNhwc, pSrcScale, pSrcShift, _method);
        _dstCvt.Init(_M, _N, 1, 1, TensorFormatNhwc, pNormScale, pNormShift, _method);
        for (size_t i = 0; i < _N; ++i)
        {
            float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
            for (size_t k = 0; k < _K; ++k)
            {
                pNormW[k] = pSrcW[i * _K + k] * pSrcScaleInv[k];
                minW = Min(minW, pNormW[k]);
                maxW = Max(maxW, pNormW[k]);
            }
            float abs = Max(Abs(maxW), Abs(minW));
            if (pSrcB)
                abs = Max(abs, Abs(pSrcB[i]) / float(128 * 256 * 256));
            scale = wUp / abs;
            for (size_t k = 0; k < _K; ++k)
            {
                if (avoidOverflow16i)
                {
                    int w = ConvertTo8i(pNormW[k], scale, 0, wLo, wUp);
                    if (w & 1)
                        w = Round(w * 0.25f) * 4;
                    pDstW[i * _K + k] = w / 2;
                    normB -= w * pSrcShift[k];
                }
                else
                {
                    pDstW[i * _K + k] = ConvertTo8i(pNormW[k], scale, 0, wLo, wUp);
                    normB -= pDstW[i * _K + k] * pSrcShift[k];
                }
            }
            pDstS[i] = avoidOverflow16i ? 2 : 1;
            if (pSrcB)
                normB += pSrcB[i] * scale;
            pDstB[i] = Synet::Quantize(normB);
            if (_dst8u)
            {
                pNormScale[i] = (1.0f / scale) * pDstScaleInv[i];
                pNormShift[i] = -pDstShift[i] / pDstScale[i];
            }
            else
            {
                pNormScale[i] = 1.0f / scale;
                pNormShift[i] = 0;
            }
        }
    }

    void InnerProduct8iLayer::ForwardCpu(const uint8_t* src, int32_t* dst)
    {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        SimdSynetCompatibilityType compatibility = (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ?
            SimdSynetCompatibility8iNarrowed : SimdSynetCompatibility8iOverflow;
        SimdSynetInnerProduct8i(_M, _N, _K, src, _weight8i.Data<int8_t>(), dst, compatibility);
#else
        const bool overflow16i = true;
        Synet::CpuGemm8iNT(_M, _N, _K, src, _K, _weight8i.Data<int8_t>(), _K, dst, _N, overflow16i);
#endif           
        const int32_t* scale = _norm32i.Data<int32_t>();
        const int32_t* shift = scale + _N;
        for (size_t i = 0; i < _M; ++i, dst += _N)
            for (size_t j = 0; j < _N; ++j)
                dst[j] = dst[j] * scale[j] + shift[j];
    }
}
