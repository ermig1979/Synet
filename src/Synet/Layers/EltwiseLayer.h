/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar,
*               2019-2019 Artur Voronkov.
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
#include "Synet/Utils/Math.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/BiasLayer.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void EltwiseLayerForwardCpu(T const * const * src, const T * weight, size_t count, size_t size, EltwiseOperationType type, T * dst)
        {
            assert(count >= 2);
            switch(type)
            {
            case EltwiseOperationTypeProduct:
                CpuMul(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMul(dst, src[i], size, dst);
                break;
            case EltwiseOperationTypeSum:
                CpuScale(src[0], size, weight[0], dst);
                for (size_t i = 1; i < count; ++i)
                    CpuAxpy(src[i], size, weight[i], dst);
                break;
            case EltwiseOperationTypeMax:
                CpuMax(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMax(dst, src[i], size, dst);
                break;
            case EltwiseOperationTypeMin:
                CpuMin(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMin(dst, src[i], size, dst);
                break;
            default:
                assert(0);
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void EltwiseLayerForwardCpu<float>(float const * const * src, const float * weight, size_t count, size_t size, EltwiseOperationType type, float * dst)
        {
            ::SimdSynetEltwiseLayerForward(src, weight, count, size, (::SimdSynetEltwiseOperationType)type, dst);
        }
#endif
    }

    template <class T> class EltwiseLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        EltwiseLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const EltwiseParam & param = this->Param().eltwise();
            assert(param.coefficients().size() == 0 || param.coefficients().size() == src.size());
            assert(!(param.operation() == EltwiseOperationTypeProduct && param.coefficients().size()));
            _operation = param.operation();
            _coefficients.resize(src.size(), Type(1));
            if (param.coefficients().size())
            {
                for (size_t i = 0; i < src.size(); ++i)
                    _coefficients[i] = param.coefficients()[i];
            } 
            _special = SpecialNone;
            _index[0] = 0;
            _index[1] = 1;
            bool resized = false;
            if (src.size() == 2 && src[0]->Shape() != src[1]->Shape() && src[0]->Size() != src[1]->Size())
            {
                _index[0] = src[0]->Size() > src[1]->Size() ? 0 : 1;
                _index[1] = src[0]->Size() > src[1]->Size() ? 1 : 0;
                size_t signDims1 = SignificantDimsCount(src[_index[1]]->Shape());
                if (src[0]->Count() > 1 && src[0]->Count() == src[1]->Count() && src[0]->Size(1) == src[1]->Size(1))
                {
                    _special = SpecialBatch;
                    _batch = Max(src[_index[0]]->Axis(0), src[_index[1]]->Axis(0));
                    _channels = 1, _spatial = src[_index[0]]->Size(1);
                    Shape shape = src[_index[0]]->Shape();
                    shape[0] = _batch;
                    if (dst[0] != src[_index[0]] && dst[0] != src[_index[1]])
                    {
                        dst[0]->Reshape(shape, src[_index[0]]->Format());
                        resized = true;
                    }
                    assert(shape == dst[0]->Shape());
                }
                else if (_operation == EltwiseOperationTypeProduct && src[_index[0]]->Count() == 3 && signDims1 == 1)
                {
                    size_t size1 = src[_index[1]]->Size();
                    if (src[_index[0]]->Axis(2) == size1)
                    {
                        _batch = 1;
                        _channelsOuter = 1;
                        _spatial = src[_index[0]]->Axis(0) * src[_index[0]]->Axis(1);
                        _channelsInner = src[_index[0]]->Axis(2);
                    }
                    else if (src[_index[0]]->Axis(1) == size1)
                    {
                        _batch = src[_index[0]]->Axis(0);
                        _channelsOuter = src[_index[0]]->Axis(1);
                        _spatial = src[_index[0]]->Axis(2);
                        _channelsInner = 1;
                    }
                    else
                        assert(0);
                    _special = SpecialScaleComplex;
                    _channels = _channelsOuter * _channelsInner;
                }
                else if (_operation == EltwiseOperationTypeProduct && src[_index[0]]->Count() == 4)
                {
                    _trans = src[_index[0]]->Format() == TensorFormatNhwc;
                    _batch = src[_index[0]]->Axis(0);
                    _channels = src[_index[0]]->Axis(_trans ? 3 : 1);
                    _spatial = src[_index[0]]->Size() / _batch / _channels;
                    size_t size = src[_index[1]]->Count() == 4 ? src[_index[1]]->Size(1) : src[_index[1]]->Size(0);
                    if (size == _channels)
                        _special = SpecialScaleChannel;
                    else if (size == _spatial)
                        _special = SpecialScaleSpatial;
                    else
                        assert(0);
                }
                else if (_operation == EltwiseOperationTypeProduct && src[0]->Count() == 5)
                {
                    _batch = src[_index[0]]->Axis(0);
                    _channelsOuter = src[_index[0]]->Axis(1);
                    _spatial = src[_index[0]]->Size(2, 4);
                    _channelsInner = src[_index[0]]->Axis(4);
                    assert(src[_index[1]]->Size(2, 4) == 1);
                    assert(src[_index[0]]->Size(0, 2) == src[_index[1]]->Size(0, 2));
                    assert(src[_index[0]]->Size(4, 5) == src[_index[1]]->Size(4, 5));
                    _special = SpecialScaleComplex;
                    _channels = _channelsOuter*_channelsInner;
                }
                else if (_operation == EltwiseOperationTypeProduct && src[0]->Count() == 6)
                {
                    _batch = src[_index[0]]->Axis(0);
                    _channelsOuter = src[_index[0]]->Axis(1);
                    _spatial = src[_index[0]]->Size(2, 5);
                    _channelsInner = src[_index[0]]->Axis(5);
                    assert(src[_index[1]]->Size(2, 5) == 1);
                    assert(src[_index[0]]->Size(0, 2) == src[_index[1]]->Size(0, 2));
                    assert(src[_index[0]]->Size(5, 6) == src[_index[1]]->Size(5, 6));
                    _special = SpecialScaleComplex;
                    _channels = _channelsOuter * _channelsInner;
                }
                else if (_operation == EltwiseOperationTypeSum && src[0]->Count() == src[1]->Count())
                {
                    _special = SpecialBiasChannel;
                    _trans = 1;
                    _batch = 1;
                    _channels = 1;
                    _spatial = 1;
                    for (size_t i = 0, already = 0; i < src[_index[0]]->Count(); ++i)
                    {
                        if (src[_index[0]]->Axis(i) == src[_index[1]]->Axis(i))
                        {
                            if (already)
                                _channels *= src[_index[0]]->Axis(i);
                            else
                                _batch *= src[_index[0]]->Axis(i);
                        }
                        else
                        {
                            assert(src[_index[1]]->Axis(i) == 1);
                            already = 1;
                            _spatial *= src[_index[0]]->Axis(i);
                        }
                    }
                }
                else if (_operation == EltwiseOperationTypeSum && src[_index[1]]->Size() == 1)
                {
                    _special = SpecialBiasChannel;
                    _trans = 1;
                    _batch = 1;
                    _channels = 1;
                    _spatial = src[_index[0]]->Size();
                }
                else if (_operation == EltwiseOperationTypeSum && src[_index[1]]->Count() == 2)
                {
                    _special = SpecialBiasChannelV2;
                    _trans = 1;
                    _batch = src[_index[1]]->Axis(0);
                    _channels = 1;
                    _spatial = src[_index[0]]->Size();
                    if (dst[0] != src[_index[0]] && dst[0] != src[_index[1]])
                    {
                        dst[0]->Reshape(Shp(_batch, _spatial), src[_index[1]]->Format());
                        resized = true;
                    }
                }
                else if (_operation == EltwiseOperationTypeSum && src[_index[1]]->Count() == 3 && src[_index[0]]->Size(1) == src[_index[1]]->Size(0))
                {
                    _special = SpecialBiasChannel;
                    _trans = 1;
                    _batch = 1;
                    _spatial = src[_index[0]]->Axis(0);
                    _channels = src[_index[0]]->Size(1);
                }
                else if (_operation == EltwiseOperationTypeSum && src[_index[0]]->Count() == 3 && src[_index[1]]->Count() == 1 && src[_index[0]]->Axis(2) == src[_index[1]]->Axis(0))
                {
                    _special = SpecialBiasChannel;
                    _trans = 1;
                    _batch = 1;
                    _spatial = src[_index[0]]->Axis(0) * src[_index[0]]->Axis(1);
                    _channels = src[_index[1]]->Axis(0);
                }
                else if (_operation == EltwiseOperationTypeSum && src[_index[0]]->Count() == 4 && src[_index[1]]->Count() == 3)
                {
                    _trans = src[_index[0]]->Format() == TensorFormatNhwc;
                    _batch = src[_index[0]]->Axis(0);
                    _channels = src[_index[0]]->Axis(_trans ? 3 : 1);
                    _spatial = src[_index[0]]->Size() / _batch / _channels;
                    size_t size = src[_index[1]]->Count() == 4 ? src[_index[1]]->Size(1) : src[_index[1]]->Size(0);
                    if (size == _channels)
                        _special = SpecialBiasChannel;
                    else
                        assert(0);
                }
                else
                    assert(0);
            }
            else
            {
                _type = src[_index[0]]->GetType();
                _src.resize(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                {
                    assert(src[i]->Size() == src[_index[0]]->Size());
                    _src[i] = src[i]->RawCpuData();
                }
                _batch = 1, _channels = 1, _spatial = src[_index[0]]->Size();
            }
            if (dst[0] != src[_index[0]] && !resized)
            {
                _type = src[_index[0]]->GetType();
                switch (_type)
                {
                case TensorType32f: dst[0]->As32f().Reshape(src[_index[0]]->Shape(), src[_index[0]]->Format()); break;
                case TensorType64i: dst[0]->As64i().Reshape(src[_index[0]]->Shape(), src[_index[0]]->Format()); break;
                default:
                    assert(0);
                }
            }
            this->UsePerfStat();
            return true;
        }

        virtual int64_t Flop() const
        {
            return _batch * _channels * _spatial * (_coefficients.size() - 1) * (_operation == EltwiseOperationTypeSum ? 2 : 1);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_special == SpecialNone)
            {
                switch (_type)
                {
                case TensorType32f:
                    Detail::EltwiseLayerForwardCpu((float const* const*)_src.data(), (const float*)_coefficients.data(), _src.size(), _spatial, _operation, dst[0]->As32f().CpuData());
                    break;
                case TensorType64i:
                {
                    int64_t coefficients[2] = { 1, 1 };
                    Detail::EltwiseLayerForwardCpu((int64_t const* const*)_src.data(), coefficients, _src.size(), _spatial, _operation, dst[0]->As64i().CpuData());
                    break;
                }
                default:
                    assert(0);
                }
                return;
            }
            const Type * pSrc0 = src[_index[0]]->CpuData();
            Type * pDst = dst[0]->CpuData();
            const Type* pBias = NULL;
            switch (_special)
            {
            case SpecialScaleChannel:
            {
                const Type * pScale = src[_index[1]]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    ScaleForward32f(pSrc0, pScale, pBias, _channels, 1, _spatial, pDst, src[_index[0]]->Format(), 0);
                    pSrc0 += _channels*_spatial;
                    pDst += _channels*_spatial;
                    pScale += _channels;
                }
                break;
            }
            case SpecialScaleSpatial:
            {
                const Type * pScale = src[_index[1]]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    ScaleForward32f(pSrc0, pScale, pBias, _spatial, 1, _channels, pDst,
                        src[_index[0]]->Format() == TensorFormatNhwc ? TensorFormatNchw : TensorFormatNhwc, 0);
                    pSrc0 += _channels*_spatial;
                    pDst += _channels*_spatial;
                    pScale += _spatial;
                }
                break;
            }
            case SpecialScaleComplex:
            {
                const Type * pScale = src[_index[1]]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    if (_channelsInner == 1)
                    {
                        ScaleForward32f(pSrc0, pScale, pBias, _channelsOuter, 1, _spatial, pDst, TensorFormatNchw, 0);
                        pSrc0 += _channelsOuter * _spatial;
                        pDst += _channelsOuter * _spatial;
                    }
                    else
                    {
                        for (size_t c = 0; c < _channelsOuter; ++c)
                        {
                            ScaleForward32f(pSrc0, pScale, pBias, _channelsInner, 1, _spatial, pDst, TensorFormatNhwc, 0);
                            pSrc0 += _channelsInner*_spatial;
                            pDst += _channelsInner*_spatial;
                            pScale += _channelsInner;
                        }
                    }
                }
                break;
            }
            case SpecialBiasChannel:
            {
                pBias = src[_index[1]]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    BiasLayerForward(pSrc0, pBias, _channels, _spatial, pDst, _trans ? TensorFormatNhwc : TensorFormatNchw);
                    pSrc0 += _channels * _spatial;
                    pDst += _channels * _spatial;
                    pBias += _channels;
                }
                break;
            }
            case SpecialBatch:
            {
                Type * src01[2] = {src[_index[0]]->CpuData(), src[_index[1]]->CpuData()};
                for (size_t b = 0; b < _batch; ++b)
                {
                    Detail::EltwiseLayerForwardCpu(src01, _coefficients.data(), 2, _spatial, _operation, pDst);
                    if (src[_index[0]]->Axis(0) > 1)
                        _src[_index[0]] += _channels * _spatial;
                    if (src[_index[1]]->Axis(0) > 1)
                        _src[_index[1]] += _channels * _spatial;
                    pDst += _channels * _spatial;
                }
                break;
            }
            case SpecialBiasChannelV2:
            {
                pBias = src[_index[1]]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    BiasLayerForward(pSrc0, pBias, _channels, _spatial, pDst, _trans ? TensorFormatNhwc : TensorFormatNchw);
                    pDst += _channels * _spatial;
                    pBias += _channels;
                }
                break;
            }
            default: assert(0);
            }
        }

    private:
        typedef std::vector<Type> Vector;
        typedef std::vector<uint8_t*> Pointers;

        enum Special
        {
            SpecialNone = 0,
            SpecialScaleChannel,
            SpecialScaleSpatial,
            SpecialScaleComplex,
            SpecialBiasChannel,
            SpecialBatch,
            SpecialBiasChannelV2,
        } _special;

        EltwiseOperationType _operation;
        Vector _coefficients;
        Pointers _src;
        TensorType _type;
        int _trans;
        size_t _batch, _channels, _spatial;
        size_t _channelsInner, _channelsOuter;
        int _index[2];
    };
}