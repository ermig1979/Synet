/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar,
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

#include "Synet/Layers/AddLayer.h"
#include "Synet/Utils/Math.h"
#include "Synet/Quantization/Convert.h"

namespace Synet
{
    template <typename T> void Add(const T& a, const T& b, T& dst)
    {
        dst = a + b;
    }

    template <> void Add(const uint16_t& a, const uint16_t& b, uint16_t& dst)
    {
        dst = Float32ToBFloat16(BFloat16ToFloat32(a) + BFloat16ToFloat32(b));
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D> void Add(const A& a, const B& b, D& dst)
    {
        float _a = Convert<A, float>(a);
        float _b = Convert<B, float>(b);
        dst = Convert<float, D>(_a + _b);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D> static void Uniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const A* a = (const A*)a8;
        const B* b = (const B*)b8;
        D* dst = (D*)dst8;
        for (size_t i = 0; i < size; ++i)
            Add(a[i], b[i], dst[i]);
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void Uniform<float, float, float>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        float weight[2] = { 1.0f, 1.0f };
        const float *src[] = {(const float*)a8, (const float*)b8};
        SimdSynetEltwiseLayerForward(src, weight, 2, size, ::SimdSynetEltwiseOperationSum, (float*)dst8);
    }
#endif

    template<class A, class B> static AddLayer::UniformPtr GetUniform(TensorType typeD)
    {
        switch (typeD)
        {
        case TensorType32f: return Uniform<A, B, float>;
        case TensorType16b: return Uniform<A, B, uint16_t>;
        default:
            return NULL;
        }
    }

    template<class A> static AddLayer::UniformPtr GetUniform(TensorType typeB, TensorType typeD)
    {
        switch (typeB)
        {
        case TensorType32f: return GetUniform<A, float>(typeD);
        case TensorType16b: return GetUniform<A, uint16_t>(typeD);
        default:
            return NULL;
        }
    }

    static AddLayer::UniformPtr GetUniform(TensorType typeA, TensorType typeB, TensorType typeD)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return Uniform<int64_t, int64_t, int64_t>;
        switch (typeA)
        {
        case TensorType32f: return GetUniform<float>(typeB, typeD);
        case TensorType16b: return GetUniform<uint16_t>(typeB, typeD);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D>  void AddBias(const uint8_t* src8, const uint8_t* bias8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        const A* src = (const A*)src8;
        const B* bias = (const B*)bias8;
        D* dst = (D*)dst8;
        if (format == TensorFormatNhwc)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    Add(src[i], bias[i], dst[i]);
                src += count;
                dst += count;
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    Add(src[j], bias[i], dst[j]);
                src += size;
                dst += size;
            }
        }
    }

    template<class A, class B> static AddLayer::AddBiasPtr GetAddBias(TensorType typeD)
    {
        switch (typeD)
        {
        case TensorType32f: return AddBias<A, B, float>;
        case TensorType16b: return AddBias<A, B, uint16_t>;
        default:
            return NULL;
        }
    }

    template<class A> static AddLayer::AddBiasPtr GetAddBias(TensorType typeB, TensorType typeD)
    {
        switch (typeB)
        {
        case TensorType32f: return GetAddBias<A, float>(typeD);
        case TensorType16b: return GetAddBias<A, uint16_t>(typeD);
        default:
            return NULL;
        }
    }

    static AddLayer::AddBiasPtr GetAddBias(TensorType typeA, TensorType typeB, TensorType typeD)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return AddBias<int64_t, int64_t, int64_t>;
        switch (typeA)
        {
        case TensorType32f: return GetAddBias<float>(typeB, typeD);
        case TensorType16b: return GetAddBias<uint16_t>(typeB, typeD);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D, size_t N> static void Universal(const uint8_t* a8, const Shape& aSteps, const uint8_t* b8, const Shape& bSteps, uint8_t* dst8, const Shape& dstShape)
    {
        const A* a = (const A*)a8;
        const B* b = (const B*)b8;
        D* dst = (D*)dst8;
        if (N == 1)
        {
            const A* a0 = a;
            const B* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                Add(*a0, *b0, *dst);
                a0 += aSteps[0];
                b0 += bSteps[0];
                dst += 1;
            }
        }
        else if (N == 2)
        {
            const A* a0 = a;
            const B* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const A* a1 = a0;
                const B* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    Add(*a1, *b1, *dst);
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                    dst += 1;
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 3)
        {
            const A* a0 = a;
            const B* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const A* a1 = a0;
                const B* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const A* a2 = a1;
                    const B* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        Add(*a2, *b2, *dst);
                        a2 += aSteps[2];
                        b2 += bSteps[2];
                        dst += 1;
                    }
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 4)
        {
            const A* a0 = a;
            const B* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const A* a1 = a0;
                const B* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const A* a2 = a1;
                    const B* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        const A* a3 = a2;
                        const B* b3 = b2;
                        for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                        {
                            Add(*a3, *b3, *dst);
                            a3 += aSteps[3];
                            b3 += bSteps[3];
                            dst += 1;
                        }
                        a2 += aSteps[2];
                        b2 += bSteps[2];
                    }
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else
            assert(0);

    }

    template<class A, class B, class D> static AddLayer::UniversalPtr GetUniversal(size_t dim)
    {
        switch (dim)
        {
        case 1: return Universal<A, B, D, 1>;
        case 2: return Universal<A, B, D, 2>;
        case 3: return Universal<A, B, D, 3>;
        case 4: return Universal<A, B, D, 4>;
        default:
            return NULL;
        }
    }

    template<class A, class B> static AddLayer::UniversalPtr GetUniversal(TensorType typeD, size_t dim)
    {
        switch (typeD)
        {
        case TensorType32f: return GetUniversal<A, B, float>(dim);
        case TensorType16b: return GetUniversal<A, B, uint16_t>(dim);
        default:
            return NULL;
        }
    }

    template<class A> static AddLayer::UniversalPtr GetUniversal(TensorType typeB, TensorType typeD, size_t dim)
    {
        switch (typeB)
        {
        case TensorType32f: return GetUniversal<A, float>(typeD, dim);
        case TensorType16b: return GetUniversal<A, uint16_t>(typeD, dim);
        default:
            return NULL;
        }
    }

    static AddLayer::UniversalPtr GetUniversal(TensorType typeA, TensorType typeB, TensorType typeD, size_t dim)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return GetUniversal<int64_t, int64_t, int64_t>(dim);
        switch (typeA)
        {
        case TensorType32f: return GetUniversal<float>(typeB, typeD, dim);
        case TensorType16b: return GetUniversal<uint16_t>(typeB, typeD, dim);
        default:
            return NULL;
        }
    }

    SYNET_INLINE bool GetSteps(const Shape& src, const Shape& dst, Shape& steps)
    {
        steps.resize(src.size(), 0);
        size_t step = 1;
        for (ptrdiff_t i = src.size() - 1; i >= 0; --i)
        {
            if (src[i] != dst[i] && src[i] != 1)
                return false;
            steps[i] = src[i] == 1 ? 0 : step;
            step *= src[i];
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    AddLayer::AddLayer(const LayerParam & param, Context* context, QuantizationMethod method)
        : Layer(param, context)
        , _method(method)
        , _quant(false)
    {
    }

    LowPrecisionType AddLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType8u && _method != QuantizationMethodUnknown)
            return LowPrecisionTypeActive;
        if (type == TensorType16b && Options().BFloat16Enable() && _method == QuantizationMethodUnknown && p.lowPrecision().bf16Type() >= LowPrecisionTypePassive)
        {
            if (p.src()[0] != p.dst()[0] && (p.src().size() == 1 || p.src()[1] != p.dst()[0]))
                return p.lowPrecision().bf16Type();
            return LowPrecisionTypePassive;
        }
        return LowPrecisionTypeNone;
    }

    int64_t AddLayer::Flop() const
    {
        if (_dstShape.size())
            return Detail::Size(_dstShape);
        return _batch * _channels * _spatial;
    }

    bool AddLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _src = src;
        if (this->Weight().size())
            _src.push_back((Tensor*)this->Weight().data() + 0);
        if (_src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("AddLayer  supports 2 inputs (or 1 input and 1 weight) and 1 output!");
        if (_src[0]->Shape() != _src[1]->Shape() && _src[0]->Size() < _src[1]->Size())
            std::swap(_src[0], _src[1]);

        _typeA = _src[0]->GetType();
        _typeB = _src[1]->GetType();
        _typeD = _typeA == TensorType64i ? TensorType64i : dst[0]->GetType();
        _format = _src[0]->Format();

        if (_method != QuantizationMethodUnknown && (_typeA == TensorType8u || _typeB == TensorType8u || _typeD == TensorType8u))
        {
            if (_typeA != _typeB)
                SYNET_ERROR("AddLayer inputs for INT8 must have the same type!");
            if (_src[0]->Count() != 4 && _format != TensorFormatUnknown)
                SYNET_ERROR("AddLayer inputs for INT8 must be 4D tensors!");
            _quant = 
                ((_typeA == TensorType8u && _typeD == TensorType8u) || (_typeA == TensorType8u && _typeD == TensorType32f) || (_typeA == TensorType32f && _typeD == TensorType8u));
        }

        if (_quant)
        {
            if (_format == TensorFormatNhwc)
            {
                _batch = _src[0]->Axis(0);
                _spatial = _src[0]->Axis(1) * _src[0]->Axis(2);
                _channels = _src[0]->Axis(3);
            }
            else if (_format == TensorFormatNchw)
            {
                _batch = _src[0]->Axis(0);
                _channels = _src[0]->Axis(1);
                _spatial = _src[0]->Axis(2) * _src[0]->Axis(3);
            }
            else
                assert(0);

            this->Stats(0)[0]->Init8u(_method);
            this->Stats(0)[1]->Init8u(_method);
            this->Stats(2)[0]->Init8u(_method);

            dst[0]->Reshape(_typeD, _src[0]->Shape(), _format);
        }
        else
        {
            _special = SpecialNone;
            _elemA = GetTensorTypeSize(_typeA);
            _elemB = GetTensorTypeSize(_typeB);
            _elemD = GetTensorTypeSize(_typeD);

            bool resized = false;
            if (_src[0]->Shape() != _src[1]->Shape() && _src[0]->Size() != _src[1]->Size())
            {
                _format = _src[0]->Format();
                size_t signDims1 = SignificantDimsCount(_src[1]->Shape());
                if (_src[0]->Count() > 1 && _src[0]->Count() == _src[1]->Count() && _src[0]->Size(1) == _src[1]->Size(1))
                {
                    _special = SpecialBatch;
                    _batch = Max(_src[0]->Axis(0), _src[1]->Axis(0));
                    _channels = 1, _spatial = _src[0]->Size(1);
                    Shape shape = _src[0]->Shape();
                    shape[0] = _batch;
                    if (dst[0] != _src[0] && dst[0] != _src[1])
                    {
                        dst[0]->Reshape(_typeD, shape, _format);
                        resized = true;
                    }
                    if(shape != dst[0]->Shape())
                        SYNET_ERROR("AddLayer can't process inputs with this shape!");
                }
                else if (_src[0]->Count() == _src[1]->Count())
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _channels = 1;
                    _spatial = 1;
                    bool invalid = false;
                    for (size_t i = 0, already = 0; i < _src[0]->Count(); ++i)
                    {
                        if (_src[0]->Axis(i) == _src[1]->Axis(i))
                        {
                            if (already)
                                _channels *= _src[0]->Axis(i);
                            else
                                _batch *= _src[0]->Axis(i);
                        }
                        else
                        {
                            if (_src[1]->Axis(i) != 1)
                                invalid = true;
                            already = 1;
                            _spatial *= _src[0]->Axis(i);
                        }
                    }
                    if (invalid)
                    {
                        Shape aShape = _src[0]->Shape(), bShape = _src[1]->Shape();
                        _dstShape.resize(_src[0]->Count(), 1);
                        for (size_t i = 0; i < _src[0]->Count(); ++i)
                            _dstShape[i] = Max(aShape[i], bShape[i]);
                        if (!(GetSteps(aShape, _dstShape, _aSteps) && GetSteps(bShape, _dstShape, _bSteps)))
                            SYNET_ERROR("AddLayer has incompatible inputs!");
                        _universal = GetUniversal(_typeA, _typeB, _typeD, _src[0]->Count());
                        if(_universal == NULL)
                            SYNET_ERROR("AddLayer can create universal worker!");
                        _special = SpecialUniversal;
                        if (dst[0] != _src[0])
                            dst[0]->Reshape(_typeD, _dstShape, _src[0]->Format());
                        resized = true;
                    }
                }
                else if (_src[1]->Size() == 1)
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _channels = 1;
                    _spatial = _src[0]->Size();
                }
                else if (_src[1]->Count() == 2)
                {
                    _special = SpecialBiasChannelV2;
                    _format = TensorFormatNhwc;
                    _batch = _src[1]->Axis(0);
                    _channels = 1;
                    _spatial = _src[0]->Size();
                    if (dst[0] != _src[0] && dst[0] != _src[1])
                    {
                        dst[0]->Reshape(_typeD, Shp(_batch, _spatial), _src[1]->Format());
                        resized = true;
                    }
                }
                else if (_src[1]->Count() == 3 && _src[0]->Size(1) == _src[1]->Size(0))
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _spatial = _src[0]->Axis(0);
                    _channels = _src[0]->Size(1);
                }
                else if (_src[0]->Count() == 3 && _src[1]->Count() == 1 && _src[0]->Axis(2) == _src[1]->Axis(0))
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _spatial = _src[0]->Axis(0) * _src[0]->Axis(1);
                    _channels = _src[1]->Axis(0);
                }
                else if (_src[0]->Count() == 4 && _src[1]->Count() == 3)
                {
                    _format = _src[0]->Format();
                    _batch = _src[0]->Axis(0);
                    _channels = _src[0]->Axis(_format == TensorFormatNhwc ? 3 : 1);
                    _spatial = _src[0]->Size() / _batch / _channels;
                    size_t size = _src[1]->Count() == 4 ? _src[1]->Size(1) : _src[1]->Size(0);
                    if (size == _channels)
                        _special = SpecialBiasChannel;
                    else
                        SYNET_ERROR("AddLayer can't process inputs with this shape!");
                }
                else if (_src[0]->Count() == 4 && _src[1]->Count() == 1 && _src[0]->Axis(-1) == _src[1]->Axis(0))
                {
                    Shape aShape = _src[0]->Shape(), bShape = Shp(1, 1, 1, _src[1]->Axis(0));
                    _dstShape.resize(_src[0]->Count(), 1);
                    for (size_t i = 0; i < _src[0]->Count(); ++i)
                        _dstShape[i] = Max(aShape[i], bShape[i]);
                    if (!(GetSteps(aShape, _dstShape, _aSteps) && GetSteps(bShape, _dstShape, _bSteps)))
                        SYNET_ERROR("AddLayer has incompatible inputs!");
                    _universal = GetUniversal(_typeA, _typeB, _typeD, _src[0]->Count());
                    if (_universal == NULL)
                        SYNET_ERROR("AddLayer can create universal worker!");
                    _special = SpecialUniversal;
                    if (dst[0] != _src[0])
                        dst[0]->Reshape(_typeD, _dstShape, _src[0]->Format());
                    resized = true;
                }
                else if (_src[0]->Count() == 2 && _src[1]->Count() == 1)
                {
                    _special = SpecialBiasChannel;
                    _batch = 1;
                    _channels = _src[1]->Axis(0);
                    _spatial = _src[0]->Size() / _channels;
                    _format = _src[0]->Axis(0) == _src[1]->Axis(0) ? TensorFormatNchw : TensorFormatNhwc;
                }
                else
                    SYNET_ERROR("AddLayer can't process inputs with this shape!");
            }
            else
                _batch = 1, _channels = 1, _spatial = _src[0]->Size();

            if (dst[0] != _src[0] && !resized)
            {
                //if (TensorUsers(_src[0]->Name()) == 1 && _typeA == _typeD)
                //    dst[0]->Share(*_src[0]);
                //else if (TensorUsers(_src[1]->Name()) == 1 && _typeB == _typeD && _src[0]->Shape() == _src[1]->Shape())
                //    dst[0]->Share(*_src[1]);
                //else
                    dst[0]->Reshape(_typeD, _src[0]->Shape(), _src[0]->Format());
            }

            _uniform = GetUniform(_typeA, _typeB, _typeD);
            _addBias = GetAddBias(_typeA, _typeB, _typeD);
            if(_uniform == NULL || _addBias == NULL)
                SYNET_ERROR("AddLayer can't process input type!");
            if(_src[0]->Shape() == _src[1]->Shape())
                _add16b.Init(_src[0]->Shape(), _typeA, _src[1]->Shape(), _typeB, _typeD, _src[0]->Format());
        }

        if (_src[0]->Const() && _src[1]->Const())
        {
            ForwardCpu(_src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            if(Options().BFloat16Enable())
                this->UsePerfStat(ToChar(_typeA) + ToChar(_typeB) + ToChar(_typeD));
            else
                this->UsePerfStat();
            _const = false;
        }

        return true;
    }

    void AddLayer::ForwardCpu(const TensorPtrs & src_, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_quant)
        {
            if (_typeA == TensorType8u && _typeD == TensorType8u)
                Add8i(_src[0]->Data<uint8_t>(), _src[1]->Data<uint8_t>(), dst[0]->Data<uint8_t>());
            else if (_typeA == TensorType8u && _typeD == TensorType32f)
                Add8i(_src[0]->Data<uint8_t>(), _src[1]->Data<uint8_t>(), dst[0]->Data<float>());
            else if (_typeA == TensorType32f && _typeD == TensorType8u)
                Add8i(_src[0]->Data<float>(), _src[1]->Data<float>(), dst[0]->Data<uint8_t>());
            else
                assert(0);
        }
        else
        {
            const uint8_t* srcA = _src[0]->RawData();
            const uint8_t* srcB = _src[1]->RawData();
            uint8_t* dst0 = dst[0]->RawData();
            if (_add16b.Enable())
            {
                _add16b.Forward(srcA, srcB, dst0);
                return;
            }
            switch (_special)
            {
            case SpecialNone:
                _uniform(srcA, srcB, _spatial, dst0);
                break;
            case SpecialBiasChannel:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _addBias(srcA, srcB, _channels, _spatial, dst0, _format);
                    srcA += _channels * _spatial * _elemA;
                    srcB += _channels * _elemB;
                    dst0 += _channels * _spatial * _elemD;
                }
                break;
            }
            case SpecialBatch:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _uniform(srcA, srcB, _spatial, dst0);
                    srcA += _channels * _spatial * _elemA;
                    dst0 += _channels * _spatial * _elemD;
                }
                break;
            }
            case SpecialBiasChannelV2:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _addBias(srcA, srcB, _channels, _spatial, dst0, _format);
                    srcB += _channels * _elemB;
                    dst0 += _channels * _spatial * _elemD;
                }
                break;
            }
            case SpecialUniversal:
                _universal(srcA, _aSteps, srcB, _bSteps, dst0, _dstShape);
                break;
            default: 
                assert(0);
            }
        }
    }

    //-------------------------------------------------------------------------------------------------
       
    void AddLayer::Add8i(const uint8_t* src0, const uint8_t* src1, uint8_t * dst)
    {
        const float* scaleSrc0 = this->Stats(0)[0]->scale8uTo32f.data();
        const float* shiftSrc0 = this->Stats(0)[0]->shift8uTo32f.data();
        const float* scaleSrc1 = this->Stats(0)[1]->scale8uTo32f.data();
        const float* shiftSrc1 = this->Stats(0)[1]->shift8uTo32f.data();
        const float* scaleDst = this->Stats(2)[0]->scale32fTo8u.data();
        const float* shiftDst = this->Stats(2)[0]->shift32fTo8u.data();
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
        {
            SimdSynetCompatibilityType compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
            ::SimdSynetAdd8i(src0, scaleSrc0, shiftSrc0, src1, scaleSrc1, shiftSrc1, dst, scaleDst, shiftDst, 
                _batch, _channels, _spatial, (SimdTensorFormatType)_format, compatibility);
            return;
        }
#endif
        int upper = ((_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ? 
            QUANT_SYMM_NARR_SRC_U8_MAX : QUANT_IE_COMP_SRC_U8_MAX);
        for (size_t b = 0; b < _batch; ++b)
        {
            if (_format == TensorFormatNchw)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t s = 0; s < _spatial; ++s)
                    {
                        float s0 = Detail::Convert<uint8_t, float, float>(src0[s], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                        float s1 = Detail::Convert<uint8_t, float, float>(src1[s], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                        dst[s] = Detail::Convert<float, uint8_t, float>(s0 + s1, scaleDst[c], shiftDst[c], 0, upper);
                    }
                    dst += _spatial, src0 += _spatial, src1 += _spatial;
                }
            }
            else if (_format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < _spatial; ++s)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        float s0 = Detail::Convert<uint8_t, float, float>(src0[c], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                        float s1 = Detail::Convert<uint8_t, float, float>(src1[c], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                        dst[c] = Detail::Convert<float, uint8_t, float>(s0 + s1, scaleDst[c], shiftDst[c], 0, upper);
                    }
                    dst += _channels, src0 += _channels, src1 += _channels;
                }
            }
            else
                assert(0);
        }
    }

    void AddLayer::Add8i(const uint8_t* src0, const uint8_t* src1, float* dst)
    {
        const float* scaleSrc0 = this->Stats(0)[0]->scale8uTo32f.data();
        const float* shiftSrc0 = this->Stats(0)[0]->shift8uTo32f.data();
        const float* scaleSrc1 = this->Stats(0)[1]->scale8uTo32f.data();
        const float* shiftSrc1 = this->Stats(0)[1]->shift8uTo32f.data();
        for (size_t b = 0; b < _batch; ++b)
        {
            if (_format == TensorFormatNchw)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t s = 0; s < _spatial; ++s)
                    {
                        float s0 = Detail::Convert<uint8_t, float, float>(src0[s], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                        float s1 = Detail::Convert<uint8_t, float, float>(src1[s], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                        dst[s] = s0 + s1;
                        dst += _spatial, src0 += _spatial, src1 += _spatial;
                    }
                }
            }
            else if (_format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < _spatial; ++s)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        float s0 = Detail::Convert<uint8_t, float, float>(src0[c], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                        float s1 = Detail::Convert<uint8_t, float, float>(src1[c], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                        dst[c] = s0 + s1;
                    }
                    dst += _channels, src0 += _channels, src1 += _channels;
                }
            }
            else
                assert(0);
        }
    }

    void AddLayer::Add8i(const float* src0, const float* src1, uint8_t * dst)
    {
        const float* scaleDst = this->Stats(2)[0]->scale32fTo8u.data();
        const float* shiftDst = this->Stats(2)[0]->shift32fTo8u.data();
        int upper = ((_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ?
            QUANT_SYMM_NARR_SRC_U8_MAX : QUANT_IE_COMP_SRC_U8_MAX);
        for (size_t b = 0; b < _batch; ++b)
        {
            if (_format == TensorFormatNchw)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t s = 0; s < _spatial; ++s)
                         dst[s] = Detail::Convert<float, uint8_t, float>(src0[s] + src1[s], scaleDst[c], shiftDst[c], 0, upper);
                    dst += _spatial, src0 += _spatial, src1 += _spatial;
                }
            }
            else if (_format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < _spatial; ++s)
                {
                    for (size_t c = 0; c < _channels; ++c)
                        dst[c] = Detail::Convert<float, uint8_t, float>(src0[c] + src1[c], scaleDst[c], shiftDst[c], 0, upper);
                    dst += _channels, src0 += _channels, src1 += _channels;
                }
            }
            else
                assert(0);
        }
    }
}