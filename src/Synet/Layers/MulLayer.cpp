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

#include "Synet/Utils/Math.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/MulLayer.h"

namespace Synet
{
    template <typename T> static void Uniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const T* a = (const T*)a8;
        const T* b = (const T*)b8;
        T* dst = (T*)dst8;
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] * b[i];
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void Uniform<float>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const float* src[] = { (const float*)a8, (const float*)b8 };
        SimdSynetEltwiseLayerForward(src, NULL, 2, size, ::SimdSynetEltwiseOperationProduct, (float*)dst8);
    }
#endif

    static MulLayer::UniformPtr GetUniform(TensorType type)
    {
        switch (type)
        {
        case TensorType32f: return Uniform<float>;
        case TensorType64i: return Uniform<int64_t>;
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T> void Scale(const uint8_t* src8, const uint8_t* scale8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        const T* src = (const T*)src8;
        const T* scale = (const T*)scale8;
        T* dst = (T*)dst8;
        if (format == TensorFormatNhwc)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = src[i] * scale[i];
                src += count;
                dst += count;
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t i = 0; i < count; ++i)
            {
                T s = scale[i];
                for (size_t j = 0; j < size; ++j)
                    dst[j] = src[j] * s;
                src += size;
                dst += size;
            }
        }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void Scale<float>(const uint8_t* src8, const uint8_t* scale8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        SimdSynetScaleLayerForward((float*)src8, (float*)scale8, NULL, count, 1, size, (float*)dst8, (SimdTensorFormatType)format, SimdSynetCompatibilityFmaUse);
    }
#endif

    MulLayer::ScalePtr GetScale(TensorType type)
    {
        switch (type)
        {
        case TensorType32f: return Scale<float>;
        case TensorType64i: return Scale<int64_t>;
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <typename T, size_t N> static void Universal(const uint8_t* a8, const Shape& aSteps, const uint8_t* b8, const Shape& bSteps, uint8_t* dst8, const Shape& dstShape)
    {
        const T* a = (const T*)a8;
        const T* b = (const T*)b8;
        T* dst = (T*)dst8;
        if (N == 1)
        {
            const T *a0 = a, *b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                *dst++ = (*a0) * (*b0);
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 2)
        {
            const T* a0 = a, * b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0, * b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    *dst++ = (*a1) * (*b1);
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 3)
        {
            const T* a0 = a, * b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0, * b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const T* a2 = a1, * b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        *dst++ = (*a2) * (*b2);
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
        else if (N == 4)
        {
            const T* a0 = a, * b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0, * b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const T* a2 = a1, * b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        const T* a3 = a2, * b3 = b2;
                        for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                        {
                            *dst++ = (*a3) * (*b3);
                            a3 += aSteps[3];
                            b3 += bSteps[3];
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

    template<class T> static MulLayer::UniversalPtr GetUniversal(size_t dim)
    {
        switch (dim)
        {
        case 1: return Universal<T, 1>;
        case 2: return Universal<T, 2>;
        case 3: return Universal<T, 3>;
        case 4: return Universal<T, 4>;
        default:
            return NULL;
        }
    }

    static MulLayer::UniversalPtr GetUniversal(TensorType type, size_t dim)
    {
        switch (type)
        {
        case TensorType32f: return GetUniversal<float>(dim);
        case TensorType64i: return GetUniversal<int64_t>(dim);
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

    MulLayer::MulLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    int64_t MulLayer::Flop() const
    {
        return _batch * _channels * _spatial;
    }

    bool MulLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _src = src;
        if (this->Weight().size())
            _src.push_back((Tensor*)this->Weight().data() + 0);
        if (_src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MulLayer supports 2 inputs (or 1 input and 1 weight) and 1 output!");
        if (_src[0]->GetType() != _src[1]->GetType())
            SYNET_ERROR("MulLayer inputs must have the same type!");

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
                {
                    Shape aShape = _src[0]->Shape(), bShape = _src[1]->Shape();
                    _dstShape.resize(3, 1);
                    for (size_t i = 0; i < 3; ++i)
                        _dstShape[i] = Max(aShape[i], bShape[i]);
                    if (!(GetSteps(aShape, _dstShape, _aSteps) && GetSteps(bShape, _dstShape, _bSteps)))
                        SYNET_ERROR("MulLayer has incompatible inputs!");
                    _universal = GetUniversal(_type, 3);
                    _special = SpecialUniversal;
                }
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
                if (size == _channels)
                    _special = SpecialScaleChannel;
                else if (size == _spatial)
                    _special = SpecialScaleSpatial;
                else if (_src[_index[1]]->Count() == 4)
                {
                    Shape aShape = _src[0]->Shape(), bShape = _src[1]->Shape();
                    _dstShape.resize(4, 1);
                    for (size_t i = 0; i < 4; ++i)
                        _dstShape[i] = Max(aShape[i], bShape[i]);
                    if (!(GetSteps(aShape, _dstShape, _aSteps) && GetSteps(bShape, _dstShape, _bSteps)))
                        SYNET_ERROR("MulLayer has incompatible inputs!");
                    _universal = GetUniversal(_type, 4);
                    _special = SpecialUniversal;
                    if (dst[0] != _src[_index[0]])
                        dst[0]->Reshape(_type, _dstShape, _src[_index[0]]->Format());
                    resized = true;
                }
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
                SYNET_ERROR("MulLayer can't process inputs with this shape!");
        }
        else
        {
            _type = _src[_index[0]]->GetType();
            _batch = 1, _channels = 1, _spatial = _src[_index[0]]->Size();
        }

        _uniform = GetUniform(_type);
        _scale = GetScale(_type);
        if (_uniform == NULL || _scale == NULL)
            SYNET_ERROR("MulLayer can't process input type!");

        if (dst[0] != _src[_index[0]] && !resized)
            dst[0]->Reshape(_type, _src[_index[0]]->Shape(), _src[_index[0]]->Format());

        if (_src[0]->Const() && _src[1]->Const())
        {
            ForwardCpu(_src, buf, dst);
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

    void MulLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
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
}