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

#pragma once

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

#ifdef _N
#undef _N
#endif

namespace Synet
{
    namespace Detail
    {
        template <class T> void InnerProductLayerForwardCpu(const T * src, const T * weight, const T * bias, size_t count, size_t size, T * dst)
        {
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = CpuDotProduct(src, weight + size*i, size) + bias[i];
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = CpuDotProduct(src, weight + size*i, size);
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void InnerProductLayerForwardCpu<float>(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            ::SimdSynetInnerProductLayerForward(src, weight, bias, count, size, dst);
        }
#endif
    }

    template <class T> class InnerProductLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        InnerProductLayer(const LayerParam & param, Context* context, QuantizationMethod method)
            : Base(param, context)
            , _method(method)
            , _internal(0)
        {
            _is8i = param.innerProduct().quantizationLevel() == TensorType8i;
        }

        virtual bool Resizable() const
        {
            return false;
        }

        virtual bool Is8i() const
        {
            return _is8i;
        }

        virtual bool Can8i() const
        {
            return _is8i;
        }

        virtual int64_t Flop() const
        {
            return _batch * _M * _N * _K * 2;
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _innerProduct32f.InternalBufferSize() * sizeof(float) +
                _weight8i.MemoryUsage() + _norm32i.MemoryUsage() + _norm32f.MemoryUsage();
        }

        virtual void CompactWeight()
        {
            if (_is8i || _internal)
                ((Tensor&)this->Weight()[0]).Clear();
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const InnerProductParam& param = this->Param().innerProduct();
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
            _biasTerm = param.biasTerm();
            _transA = param.transposeA();
            _transB = param.transposeB();
            int axis = (int)src[0]->Index(param.axis());
            _batch = 1;
            _K = src[0]->Size(axis);
            if (src.size() == 2)
            {
                assert(_biasTerm == false);
                assert(_K = src[1]->Size(axis - 1, axis));
                _N = src[1]->Axis(axis);
            }
            else
            {
                _N = this->Param().innerProduct().outputNum();
                const typename Base::Tensors & weight = this->Weight();
                if (_biasTerm)
                    assert(weight.size() == 2);
                else
                    assert(weight.size() == 1);
                if (_transB)
                    assert(weight[0].Shape() == Shp(_K, _N));
                else
                {
                    if (weight[0].Shape()[1] != _K)
                    {
                        axis = (int)src[0]->Count() - 1;
                        _K = src[0]->Size(axis);
                    }
                    assert(weight[0].Shape() == Shp(_N, _K));
                }
                if (_biasTerm)
                    assert(weight[1].Shape() == Shp(_N));
#if defined(SYNET_BF16_ROUND_TEST)
                if (this->Param().convolution().bf16() && this->Options().bf16RoundTest)
                {
                    RoundAsTo16(weight[0].CpuData(), weight[0].Size(), (float*)weight[0].CpuData());
                    Base::Extend32f(buf, 1, src[0]->Shape(), src[0]->Format());
                }
#endif
            }
            if (src.size() > 1)
            {
                _M = src[0]->Size(Max(0, axis - 1), axis);
                _batch = axis > 0 ? src[0]->Size(0, axis - 1) : 1;
            }
            else
            {
                _M = src[0]->Size(0, axis);
                _batch = 1;
            }
            Shape dstShape = src[0]->Shape();
            dstShape.resize(axis + 1);
            dstShape[axis] = _N;
            if (_dst8u)
                dst[0]->As8u().Reshape(dstShape, TensorFormatNchw);
            else
                dst[0]->As32f().Reshape(dstShape, TensorFormatNchw);
            if (_is8i)
            {
                if (!_src8u)
                    Base::Extend8u(buf, 0, src[0]->Shape(), src[0]->Format());
                Base::Extend32i(buf, 0, dstShape, TensorFormatNchw);
                Quantize();
            }
            else if (!_transA && src.size() == 1)
            {
                _innerProduct32f.Init(_M, _K, _N, _transB ? 0 : 1);
                if (_innerProduct32f.Enable())
                {
                    const float* weight = this->Weight()[0].CpuData();
                    const float* bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
                    _innerProduct32f.SetParams(weight, &_internal, bias, NULL);
                }
            }
            std::stringstream desc;
            if (_batch > 1)
                desc << "B=" << _batch << " ";
            desc << "M=" << _M << " N=" << _N << " K=" << _K;
            this->UsePerfStat(desc.str(), Flop());
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_is8i)
            {
                uint8_t* tmp = _src8u ? src[0]->As8u().CpuData() : Base::Buf8u(buf, 0);
                int32_t* sum = Base::Buf32i(buf, 0);
                if (!_src8u)
                    _srcCvt.Convert(src[0]->As32f().CpuData(), tmp);
                ForwardCpu(tmp, sum);
                if (_dst8u)
                    _dstCvt.Convert(sum, dst[0]->As8u().CpuData());
                else
                    _dstCvt.Convert(sum, dst[0]->As32f().CpuData());
            }
            else if (_innerProduct32f.Enable())
                _innerProduct32f.Forward(src[0]->CpuData(), dst[0]->CpuData());
            else
            {
                if (src.size() > 1)
                {
                    const float* src0 = src[0]->CpuData();
                    const float* src1 = src[1]->CpuData();
                    float* dst0 = dst[0]->CpuData();
                    for(size_t b = 0; b < _batch; ++b)
                    {
                        ForwardCpu(src0, src1, dst0);
                        src0 += _M * _K;
                        src1 += _K * _N;
                        dst0 += _M * _N;
                    }
                }
                else
                {
#if defined(SYNET_BF16_ROUND_TEST)
                    if (this->Param().convolution().bf16() && this->Options().bf16RoundTest)
                    {
                        RoundAsTo16(src[0]->CpuData(), src[0]->Size(), Base::Buf32f(buf, 1));
                        ForwardCpu(Base::Buf32f(buf, 1), this->Weight()[0].CpuData(), dst[0]->CpuData());
                    }
                    else
#endif
                        ForwardCpu(src[0]->CpuData(), this->Weight()[0].CpuData(), dst[0]->CpuData());
                }
            }
        }

        void ForwardCpu(const float * src, const float* wgt, float* dst)
        {
            const float* bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
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

        void Quantize()
        {
            Stat& statS = *this->Stats(0)[0];
            Stat& statD = *this->Stats(2)[0];
            statS.Init8u(_method);
            statD.Init8u(_method);
            _weight8i.Reshape(this->Weight()[0].Shape(), TensorFormatNchw);
            _norm32i.Reshape(Shp(2, _N));
            _norm32f.Reshape(Shp(2, _N));
            Floats normW(_K);
            const float* pSrcW = this->Weight()[0].CpuData();
            const float* pSrcB = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            const float* pSrcScaleInv = statS.scale8uTo32f.data();
            const float* pSrcScale = statS.scale32fTo8u.data();
            const float* pSrcShift = statS.shift32fTo8u.data();
            const float* pDstScale = statD.scale8uTo32f.data();
            const float* pDstScaleInv = statD.scale32fTo8u.data();
            const float* pDstShift = statD.shift8uTo32f.data();
            float* pNormW = normW.data();
            int8_t* pDstW = _weight8i.CpuData();
            int32_t* pDstS = _norm32i.CpuData();
            int32_t* pDstB = pDstS + _N;
            float* pNormScale = _norm32f.CpuData();
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

        void ForwardCpu(const uint8_t* src, int32_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            SimdSynetCompatibilityType compatibility = (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ?
                SimdSynetCompatibility8iNarrowed : SimdSynetCompatibility8iOverflow;
            SimdSynetInnerProduct8i(_M, _N, _K, src, _weight8i.CpuData(), dst, compatibility);
#else
            const bool overflow16i = true;
            Synet::CpuGemm8iNT(_M, _N, _K, src, _K, _weight8i.CpuData(), _K, dst, _N, overflow16i);
#endif           
            const int32_t* scale = _norm32i.CpuData();
            const int32_t* shift = scale + _N;
            for (size_t i = 0; i < _M; ++i, dst += _N)
                for (size_t j = 0; j < _N; ++j)
                    dst[j] = dst[j] * scale[j] + shift[j];
        }

    private:
        QuantizationMethod _method;
        size_t _batch, _M, _N, _K;
        bool _biasTerm, _transA, _transB, _src8u, _dst8u, _is8i;
        int _internal;
        InnerProduct32f _innerProduct32f;
        Converter _srcCvt, _dstCvt;

        Tensor8i _weight8i;
        Tensor32i _norm32i;
        Tensor32f _norm32f;
    };
}
