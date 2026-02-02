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

#include "Synet/Utils/Math.h"
#include "Synet/Utils/UniversalBinary.h"
#include "Synet/Layers/Math/ScaleLayer.h"
#include "Synet/Layers/Math/MulLayer.h"

namespace Synet
{
    template <typename T> void Mul(const T& a, const T& b, T& dst)
    {
        dst = a * b;
    }

    template <> void Mul(const uint16_t& a, const uint16_t& b, uint16_t& dst)
    {
        dst = Float32ToBFloat16(BFloat16ToFloat32(a) * BFloat16ToFloat32(b));
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D> void Mul(const A& a, const B& b, D& dst)
    {
        float _a = Convert<A, float>(a);
        float _b = Convert<B, float>(b);
        dst = Convert<float, D>(_a * _b);
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D> static void MulUniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const A* a = (const A*)a8;
        const B* b = (const B*)b8;
        D* dst = (D*)dst8;
        for (size_t i = 0; i < size; ++i)
            Mul(a[i], b[i], dst[i]);
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void MulUniform<float, float, float>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const float* src[] = { (const float*)a8, (const float*)b8 };
        SimdSynetEltwiseLayerForward(src, NULL, 2, size, ::SimdSynetEltwiseOperationProduct, (float*)dst8);
    }
#endif

    template<class A, class B> static MulLayer::UniformPtr GetMulUniform(TensorType typeD)
    {
        switch (typeD)
        {
        case TensorType32f: return MulUniform<A, B, float>;
        case TensorType16b: return MulUniform<A, B, uint16_t>;
        default:
            return NULL;
        }
    }

    template<class A> static MulLayer::UniformPtr GetMulUniform(TensorType typeB, TensorType typeD)
    {
        switch (typeB)
        {
        case TensorType32f: return GetMulUniform<A, float>(typeD);
        case TensorType16b: return GetMulUniform<A, uint16_t>(typeD);
        default:
            return NULL;
        }
    }

    static MulLayer::UniformPtr GetMulUniform(TensorType typeA, TensorType typeB, TensorType typeD)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return MulUniform<int64_t, int64_t, int64_t>;
        switch (typeA)
        {
        case TensorType32f: return GetMulUniform<float>(typeB, typeD);
        case TensorType16b: return GetMulUniform<uint16_t>(typeB, typeD);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D> void Scale(const uint8_t* src8, const uint8_t* scale8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        const A* src = (const A*)src8;
        const B* scale = (const B*)scale8;
        D* dst = (D*)dst8;
        if (format == TensorFormatNhwc)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    Mul<A, B, D>(src[i], scale[i], dst[i]);
                src += count;
                dst += count;
            }
        }
        else// if (format == TensorFormatNchw)
        {
            for (size_t i = 0; i < count; ++i)
            {
                B _scale = scale[i];
                for (size_t j = 0; j < size; ++j)
                    Mul<A, B, D>(src[j], _scale, dst[j]);
                src += size;
                dst += size;
            }
        }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void Scale<float, float, float>(const uint8_t* src8, const uint8_t* scale8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        SimdSynetScaleLayerForward((float*)src8, (float*)scale8, NULL, count, 1, size, (float*)dst8, (SimdTensorFormatType)format, SimdSynetCompatibilityFmaUse);
    }
#endif

    template<class A, class B> static MulLayer::ScalePtr GetScale(TensorType typeD)
    {
        switch (typeD)
        {
        case TensorType32f: return Scale<A, B, float>;
        case TensorType16b: return Scale<A, B, uint16_t>;
        default:
            return NULL;
        }
    }

    template<class A> static MulLayer::ScalePtr GetScale(TensorType typeB, TensorType typeD)
    {
        switch (typeB)
        {
        case TensorType32f: return GetScale<A, float>(typeD);
        case TensorType16b: return GetScale<A, uint16_t>(typeD);
        default:
            return NULL;
        }
    }

    static MulLayer::ScalePtr GetScale(TensorType typeA, TensorType typeB, TensorType typeD)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return Scale<int64_t, int64_t, int64_t>;
        switch (typeA)
        {
        case TensorType32f: return GetScale<float>(typeB, typeD);
        case TensorType16b: return GetScale<uint16_t>(typeB, typeD);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename A, typename B, typename D, size_t N> static void MulUniversal(const uint8_t* a8, const Shape& aSteps, const uint8_t* b8, const Shape& bSteps, uint8_t* dst8, const Shape& dstShape)
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
                Mul<A, B, D>(*a0, *b0, *dst);
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
                    Mul<A, B, D>(*a1, *b1, *dst);
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
                        Mul<A, B, D>(*a2, *b2, *dst);
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
                            Mul<A, B, D>(*a3, *b3, *dst);
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

    template<class A, class B, class D> static MulLayer::UniversalPtr GetMulUniversal(size_t dim)
    {
        switch (dim)
        {
        case 1: return MulUniversal<A, B, D, 1>;
        case 2: return MulUniversal<A, B, D, 2>;
        case 3: return MulUniversal<A, B, D, 3>;
        case 4: return MulUniversal<A, B, D, 4>;
        default:
            return NULL;
        }
    }

    template<class A, class B> static MulLayer::UniversalPtr GetMulUniversal(TensorType typeD, size_t dim)
    {
        switch (typeD)
        {
        case TensorType32f: return GetMulUniversal<A, B, float>(dim);
        case TensorType16b: return GetMulUniversal<A, B, uint16_t>(dim);
        default:
            return NULL;
        }
    }

    template<class A> static MulLayer::UniversalPtr GetMulUniversal(TensorType typeB, TensorType typeD, size_t dim)
    {
        switch (typeB)
        {
        case TensorType32f: return GetMulUniversal<A, float>(typeD, dim);
        case TensorType16b: return GetMulUniversal<A, uint16_t>(typeD, dim);
        default:
            return NULL;
        }
    }

    MulLayer::UniversalPtr GetMulUniversal(TensorType typeA, TensorType typeB, TensorType typeD, size_t dim)
    {
        if (typeA == TensorType64i && typeB == TensorType64i && typeD == TensorType64i)
            return GetMulUniversal<int64_t, int64_t, int64_t>(dim);
        if (typeA == TensorType32i && typeB == TensorType32i && typeD == TensorType32i)
            return GetMulUniversal<int32_t, int32_t, int32_t>(dim);
        switch (typeA)
        {
        case TensorType32f: return GetMulUniversal<float>(typeB, typeD, dim);
        case TensorType16b: return GetMulUniversal<uint16_t>(typeB, typeD, dim);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    MulLayer::MulLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t MulLayer::Flop() const
    {
        if (_dstShape.size())
            return Detail::Size(_dstShape);
        return _batch * _channels * _spatial;
    }

    bool MulLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() + this->Weight().size() != 2 || dst.size() != 1)
            SYNET_ERROR("MulLayer supports 2 inputs (or 1 input and 1 weight) and 1 output!");
        TensorPtrs _src = GetSrc(src);
        if (_src[0]->GetType() != _src[1]->GetType())
            SYNET_ERROR("MulLayer inputs must have the same type!");

        Shape shapeA = _src[0]->Shape(), shapeB = _src[1]->Shape();
        TensorFormat formatA = _src[0]->Format(), formatB = _src[1]->Format();
        if (!IsCompatible(shapeA, shapeB))
        {
            if (formatA != formatB)
            {
                if (shapeA.size() == 4 && shapeB.size() == 3 && SignificantDimsCount(shapeB) == 1)
                {
                    shapeB = Shp(1, shapeB[1], shapeB[2], shapeB[0]);
                    formatB = formatA;
                }
                else if (shapeA.size() == 4 && shapeB.size() == 4 && shapeB[2] == 1 && shapeB[3] == 1)
                {
                    shapeB = Shp(shapeB[0], shapeB[2], shapeB[3], shapeB[1]);
                }
            }
            else if(formatA == TensorFormatNhwc)
            {
                if (shapeA.size() == 4 && shapeB.size() == 4 && shapeB[2] == 1 && shapeB[3] == 1)
                {
                    shapeB = Shp(shapeB[0], shapeB[2], shapeB[3], shapeB[1]);
                }
            }
            if (!IsCompatible(shapeA, shapeB))
                SYNET_ERROR("MulLayer incompatible input shapes and they can't be corrected!");
        }
        Shape shapeD = OutputShape(shapeA, shapeB);

        _type = _src[0]->GetType();
        _format = _src[0]->Format();
        _sizeT = GetTensorTypeSize(_type);
        _special = SpecialNone;
        _index[0] = 0;
        _index[1] = 1;
        bool resized = false;

        if (_src.size() == 2 && _src[0]->Shape() != _src[1]->Shape() && _src[0]->Size() != _src[1]->Size())
        {
            if (src.size() == 2)
            {
                _index[0] = _src[0]->Size() > _src[1]->Size() ? 0 : 1;
                _index[1] = _src[0]->Size() > _src[1]->Size() ? 1 : 0;
            }
            _format = _src[_index[0]]->Format();
            size_t signDims1 = SignificantDimsCount(_src[_index[1]]->Shape());
            if (_src[0]->Count() > 1 && _src[0]->Count() == _src[1]->Count() && _src[0]->Size(1) == _src[1]->Size(1))
            {
                _special = SpecialBatch;
                _batch = Max(_src[0]->Axis(0), _src[1]->Axis(0));
                _channels = 1, _spatial = _src[_index[0]]->Size(1);
                Shape shape = _src[_index[0]]->Shape();
                shape[0] = _batch;
                if (dst[0] != _src[_index[0]] && dst[0] != _src[_index[1]])
                {
                    dst[0]->Reshape(_type, shape, _src[_index[0]]->Format());
                    resized = true;
                }
                if (shape != dst[0]->Shape())
                    SYNET_ERROR("MulLayer can't process inputs with this shape!");
            }
            else if (_src[_index[0]]->Count() == 3 && signDims1 == 1)
            {
                size_t size1 = _src[_index[1]]->Size();
                if (_src[_index[0]]->Axis(2) == size1)
                {
                    _batch = 1;
                    _channelsOuter = 1;
                    _spatial = _src[_index[0]]->Axis(0) * _src[_index[0]]->Axis(1);
                    _channelsInner = _src[_index[0]]->Axis(2);
                    _special = SpecialScaleComplex;
                    _channels = _channelsOuter * _channelsInner;
                }
                else if (_src[_index[0]]->Axis(1) == size1)
                {
                    _batch = _src[_index[0]]->Axis(0);
                    _channelsOuter = _src[_index[0]]->Axis(1);
                    _spatial = _src[_index[0]]->Axis(2);
                    _channelsInner = 1;
                    _special = SpecialScaleComplex;
                    _channels = _channelsOuter * _channelsInner;
                }
                else if (_src[_index[1]]->Count() == 3)
                    _special = SpecialUniversal;
                else
                    SYNET_ERROR("MulLayer can't process inputs with this shape!");
            }
            else if (_src[_index[0]]->Count() == 4)
            {
                _batch = _src[_index[0]]->Axis(0);
                if (_src[_index[0]]->Axis(3) == _src[_index[1]]->Size(0))
                    _format = TensorFormatNhwc;
                _channels = _src[_index[0]]->Axis(_format == TensorFormatNhwc ? 3 : 1);
                _spatial = _src[_index[0]]->Size() / _batch / _channels;
                size_t size = _src[_index[1]]->Count() == 4 ? _src[_index[1]]->Size(1) : _src[_index[1]]->Size(0);
                if (size == 1)
                {
                    _special = SpecialScaleChannel;
                    _spatial *= _channels;
                    _channels = 1;
                }
                else if (size == _channels)
                    _special = SpecialScaleChannel;
                else if (size == _spatial)
                    _special = SpecialScaleSpatial;
                else if (_src[_index[1]]->Count() == 4)
                    _special = SpecialUniversal;
                else
                    SYNET_ERROR("MulLayer can't process inputs with this shape!");
            }
            else if (_src[0]->Count() == 5)
            {
                _batch = _src[_index[0]]->Axis(0);
                _channelsOuter = _src[_index[0]]->Axis(1);
                _spatial = _src[_index[0]]->Size(2, 4);
                _channelsInner = _src[_index[0]]->Axis(4);
                assert(_src[_index[1]]->Size(2, 4) == 1);
                assert(_src[_index[0]]->Size(0, 2) == _src[_index[1]]->Size(0, 2));
                assert(_src[_index[0]]->Size(4, 5) == _src[_index[1]]->Size(4, 5));
                _special = SpecialScaleComplex;
                _channels = _channelsOuter * _channelsInner;
            }
            else if (_src[0]->Count() == 6)
            {
                _batch = _src[_index[0]]->Axis(0);
                _channelsOuter = _src[_index[0]]->Axis(1);
                _spatial = _src[_index[0]]->Size(2, 5);
                _channelsInner = _src[_index[0]]->Axis(5);
                assert(_src[_index[1]]->Size(2, 5) == 1);
                assert(_src[_index[0]]->Size(0, 2) == _src[_index[1]]->Size(0, 2));
                assert(_src[_index[0]]->Size(5, 6) == _src[_index[1]]->Size(5, 6));
                _special = SpecialScaleComplex;
                _channels = _channelsOuter * _channelsInner;
            }
            else if (_src[_index[1]]->Size() == 1)
            {
                _batch = 1;
                _channels = 1;
                _spatial = _src[_index[0]]->Size();
                _special = SpecialScaleChannel;
            }
            else
                _special = SpecialUniversal;
        }
        else
        {
            _type = _src[_index[0]]->GetType();
            _batch = 1, _channels = 1, _spatial = _src[_index[0]]->Size();
        }

        if (_special == SpecialUniversal)
        {
            Shape aShape = _src[0]->Shape(), bShape = _src[1]->Shape();
            _dstShape = shapeD;
            if (!IsCompatible(shapeA, shapeB))
                SYNET_ERROR("MulLayer has incompatible inputs!");
            _aSteps = SourceSteps(shapeA, _dstShape);
            _bSteps = SourceSteps(shapeB, _dstShape);
            _universal = GetMulUniversal(_type, _type, _type, shapeA.size());
            if (_universal == NULL)
                SYNET_ERROR("MulLayer can create universal worker!");
        }
        _uniform = GetMulUniform(_type, _type, _type);
        _scale = GetScale(_type, _type, _type);
        if (_uniform == NULL || _scale == NULL)
            SYNET_ERROR("MulLayer can't process input type!");

        if (dst[0] != _src[_index[0]] && !resized)
            dst[0]->Reshape(_type, _src[_index[0]]->Shape(), _src[_index[0]]->Format());

        if (_src[0]->Const() && _src[1]->Const())
        {
            Forward(_src, buf, dst, 0);
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

    void MulLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        TensorPtrs _src = GetSrc(src);
        const uint8_t* src0 = _src[_index[0]]->RawData();
        const uint8_t* src1 = _src[_index[1]]->RawData();
        uint8_t* dst0 = dst[0]->RawData();
        switch (_special)
        {
        case SpecialNone:
            _uniform(src0, src1, _spatial, dst0);
            break;
        case SpecialScaleChannel:
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _scale(src0, src1, _channels, _spatial, dst0, _format);
                src0 += _channels * _spatial * _sizeT;
                src1 += _channels * _sizeT;
                dst0 += _channels * _spatial * _sizeT;
            }
            break;
        }
        case SpecialScaleSpatial:
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _scale(src0, src1, _spatial, _channels, dst0,
                    src[_index[0]]->Format() == TensorFormatNhwc ? TensorFormatNchw : TensorFormatNhwc);
                src0 += _channels * _spatial * _sizeT;
                src1 += _spatial * _sizeT;
                dst0 += _channels * _spatial * _sizeT;
            }
            break;
        }
        case SpecialScaleComplex:
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_channelsInner == 1)
                {
                    _scale(src0, src1, _channelsOuter, _spatial, dst0, TensorFormatNchw);
                    src0 += _channelsOuter * _spatial * _sizeT;
                    dst0 += _channelsOuter * _spatial * _sizeT;
                }
                else
                {
                    for (size_t c = 0; c < _channelsOuter; ++c)
                    {
                        _scale(src0, src1, _channelsInner, _spatial, dst0, TensorFormatNhwc);
                        src0 += _channelsInner * _spatial * _sizeT;
                        src1 += _channelsInner * _sizeT;
                        dst0 += _channelsInner * _spatial * _sizeT;
                    }
                }
            }
            break;
        }
        case SpecialUniversal:
            _universal(src0, _aSteps, src1, _bSteps, dst0, _dstShape);
            break;
        default:
            assert(0);
        }
    }

    MulLayer::TensorPtrs MulLayer::GetSrc(const TensorPtrs& src)
    {
        TensorPtrs _src = src;
        if (this->Weight().size())
            _src.push_back((Tensor*)this->Weight().data() + 0);
        return _src;
    }
}