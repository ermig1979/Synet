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
    template <typename T> static void Uniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        const T* a = (const T*)a8;
        const T* b = (const T*)b8;
        T* dst = (T*)dst8;
        for (size_t i = 0; i < size; ++i)
            dst[i] = a[i] + b[i];
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> void Uniform<float>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
    {
        float weight[2] = { 1.0f, 1.0f };
        const float *src[] = {(const float*)a8, (const float*)b8};
        SimdSynetEltwiseLayerForward(src, weight, 2, size, ::SimdSynetEltwiseOperationSum, (float*)dst8);
    }
#endif

    static AddLayer::UniformPtr GetUniform(TensorType type)
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

    template <typename T> void AddBias(const uint8_t* src8, const uint8_t* bias8, size_t count, size_t size, uint8_t* dst8, TensorFormat format)
    {
        const T* src = (const T*)src8;
        const T* bias = (const T*)bias8;
        T* dst = (T*)dst8;
        if (format == TensorFormatNhwc)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = src[i] + bias[i];
                src += count;
                dst += count;
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t i = 0; i < count; ++i)
            {
                T b = bias[i];
                for (size_t j = 0; j < size; ++j)
                    dst[j] = src[j] + b;
                src += size;
                dst += size;
            }
        }
    }

    AddLayer::AddBiasPtr GetAddBias(TensorType type)
    {
        switch (type)
        {
        case TensorType32f: return AddBias<float>;
        case TensorType64i: return AddBias<int64_t>;
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
            const T* a0 = a, * b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                *dst++ = (*a0) + (*b0);
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
                    *dst++ = (*a1) + (*b1);
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
                        *dst++ = (*a2) + (*b2);
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
                            *dst++ = (*a3) + (*b3);
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

    template<class T> static AddLayer::UniversalPtr GetUniversal(size_t dim)
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

    static AddLayer::UniversalPtr GetUniversal(TensorType type, size_t dim)
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

    AddLayer::AddLayer(const LayerParam & param, Context* context, QuantizationMethod method)
        : Base(param, context)
        , _method(method)
    {
    }

    bool AddLayer::Can8i() const
    {
        return _method != QuantizationMethodUnknown;
    }

    bool AddLayer::Is8i() const
    {
        return _method != QuantizationMethodUnknown;
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
        if (_src[0]->GetType() != _src[1]->GetType())
            SYNET_ERROR("AddLayer inputs must have the same type!");

        _srcT = _src[0]->GetType();
        _dstT = dst[0]->GetType();
        _format = _src[0]->Format();
        _quant = _method != QuantizationMethodUnknown && _src[0]->Count() == 4 && _format != TensorFormatUnknown &&
            ((_srcT == TensorType8u && _dstT == TensorType8u) || (_srcT == TensorType8u && _dstT == TensorType32f) || (_srcT == TensorType32f && _dstT == TensorType8u));

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

            dst[0]->Reshape(_dstT, _src[0]->Shape(), _format);
        }
        else
        {
            _special = SpecialNone;
            _index[0] = 0;
            _index[1] = 1;
            _dstT = _srcT;
            _sizeT = GetTensorTypeSize(_srcT);

            bool resized = false;
            if (_src[0]->Shape() != _src[1]->Shape() && _src[0]->Size() != _src[1]->Size())
            {
                if (src.size() == 2)
                {
                    _index[0] = _src[0]->Size() > _src[1]->Size() ? 0 : 1;
                    _index[1] = _src[0]->Size() > _src[1]->Size() ? 1 : 0;
                }
                _format = _src[_index[0]]->Format();
                size_t signDims1 = SignificantDimsCount(_src[_index[1]]->Shape());
                if (_src[_index[0]]->Count() > 1 && _src[0]->Count() == _src[1]->Count() && _src[0]->Size(1) == _src[1]->Size(1))
                {
                    _special = SpecialBatch;
                    _batch = Max(_src[0]->Axis(0), _src[1]->Axis(0));
                    _channels = 1, _spatial = _src[_index[0]]->Size(1);
                    Shape shape = _src[_index[0]]->Shape();
                    shape[0] = _batch;
                    if (dst[0] != _src[_index[0]] && dst[0] != _src[_index[1]])
                    {
                        dst[0]->Reshape(_dstT, shape, _src[_index[0]]->Format());
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
                    for (size_t i = 0, already = 0; i < _src[_index[0]]->Count(); ++i)
                    {
                        if (_src[_index[0]]->Axis(i) == _src[_index[1]]->Axis(i))
                        {
                            if (already)
                                _channels *= _src[_index[0]]->Axis(i);
                            else
                                _batch *= _src[_index[0]]->Axis(i);
                        }
                        else
                        {
                            if (_src[_index[1]]->Axis(i) != 1)
                                invalid = true;
                            already = 1;
                            _spatial *= _src[_index[0]]->Axis(i);
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
                        _universal = GetUniversal(_srcT, _src[0]->Count());
                        if(_universal == NULL)
                            SYNET_ERROR("AddLayer can create universal worker!");
                        _special = SpecialUniversal;
                        if (dst[0] != _src[_index[0]])
                            dst[0]->Reshape(_srcT, _dstShape, _src[_index[0]]->Format());
                        resized = true;
                    }
                }
                else if (_src[_index[1]]->Size() == 1)
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _channels = 1;
                    _spatial = _src[_index[0]]->Size();
                }
                else if (_src[_index[1]]->Count() == 2)
                {
                    _special = SpecialBiasChannelV2;
                    _format = TensorFormatNhwc;
                    _batch = _src[_index[1]]->Axis(0);
                    _channels = 1;
                    _spatial = _src[_index[0]]->Size();
                    if (dst[0] != _src[_index[0]] && dst[0] != _src[_index[1]])
                    {
                        dst[0]->Reshape(Shp(_batch, _spatial), _src[_index[1]]->Format());
                        resized = true;
                    }
                }
                else if (_src[_index[1]]->Count() == 3 && _src[_index[0]]->Size(1) == _src[_index[1]]->Size(0))
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _spatial = _src[_index[0]]->Axis(0);
                    _channels = _src[_index[0]]->Size(1);
                }
                else if (_src[_index[0]]->Count() == 3 && _src[_index[1]]->Count() == 1 && _src[_index[0]]->Axis(2) == _src[_index[1]]->Axis(0))
                {
                    _special = SpecialBiasChannel;
                    _format = TensorFormatNhwc;
                    _batch = 1;
                    _spatial = _src[_index[0]]->Axis(0) * _src[_index[0]]->Axis(1);
                    _channels = _src[_index[1]]->Axis(0);
                }
                else if (_src[_index[0]]->Count() == 4 && _src[_index[1]]->Count() == 3)
                {
                    _format = _src[_index[0]]->Format();
                    _batch = _src[_index[0]]->Axis(0);
                    _channels = _src[_index[0]]->Axis(_format == TensorFormatNhwc ? 3 : 1);
                    _spatial = _src[_index[0]]->Size() / _batch / _channels;
                    size_t size = _src[_index[1]]->Count() == 4 ? _src[_index[1]]->Size(1) : _src[_index[1]]->Size(0);
                    if (size == _channels)
                        _special = SpecialBiasChannel;
                    else
                        SYNET_ERROR("AddLayer can't process inputs with this shape!");
                }
                else if (_src[_index[0]]->Count() == 4 && _src[_index[1]]->Count() == 1 && _src[_index[0]]->Axis(-1) == _src[_index[1]]->Axis(0))
                {
                    Shape aShape = _src[0]->Shape(), bShape = Shp(1, 1, 1, _src[_index[1]]->Axis(0));
                    _dstShape.resize(_src[0]->Count(), 1);
                    for (size_t i = 0; i < _src[0]->Count(); ++i)
                        _dstShape[i] = Max(aShape[i], bShape[i]);
                    if (!(GetSteps(aShape, _dstShape, _aSteps) && GetSteps(bShape, _dstShape, _bSteps)))
                        SYNET_ERROR("AddLayer has incompatible inputs!");
                    _universal = GetUniversal(_srcT, _src[0]->Count());
                    if (_universal == NULL)
                        SYNET_ERROR("AddLayer can create universal worker!");
                    _special = SpecialUniversal;
                    if (dst[0] != _src[_index[0]])
                        dst[0]->Reshape(_srcT, _dstShape, _src[_index[0]]->Format());
                    resized = true;
                }
                else if (_src[_index[0]]->Count() == 2 && _src[_index[1]]->Count() == 1)
                {
                    _special = SpecialBiasChannel;
                    _batch = 1;
                    _channels = _src[_index[1]]->Axis(0);
                    _spatial = _src[_index[0]]->Size() / _channels;
                    _format = _src[_index[0]]->Axis(0) == _src[_index[1]]->Axis(0) ? TensorFormatNchw : TensorFormatNhwc;
                }
                else
                    SYNET_ERROR("AddLayer can't process inputs with this shape!");
            }
            else
                _batch = 1, _channels = 1, _spatial = _src[_index[0]]->Size();

            if (dst[0] != _src[_index[0]] && !resized)
                dst[0]->Reshape(_dstT, _src[_index[0]]->Shape(), _src[_index[0]]->Format());

            _uniform = GetUniform(_srcT);
            _addBias = GetAddBias(_srcT);
            if(_uniform == NULL || _addBias == NULL)
                SYNET_ERROR("AddLayer can't process input type!");
        }

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

    void AddLayer::ForwardCpu(const TensorPtrs & src_, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_quant)
        {
            if (_srcT == TensorType8u && _dstT == TensorType8u)
                Add8i(_src[0]->Data<uint8_t>(), _src[1]->Data<uint8_t>(), dst[0]->Data<uint8_t>());
            else if (_srcT == TensorType8u && _dstT == TensorType32f)
                Add8i(_src[0]->Data<uint8_t>(), _src[1]->Data<uint8_t>(), dst[0]->Data<float>());
            else if (_srcT == TensorType32f && _dstT == TensorType8u)
                Add8i(_src[0]->Data<float>(), _src[1]->Data<float>(), dst[0]->Data<uint8_t>());
            else
                assert(0);
        }
        else
        {
            const uint8_t* src0 = _src[_index[0]]->RawData();
            const uint8_t* src1 = _src[_index[1]]->RawData();
            uint8_t* dst0 = dst[0]->RawData();
            switch (_special)
            {
            case SpecialNone:
                _uniform(src0, src1, _spatial, dst0);
                break;
            case SpecialBiasChannel:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _addBias(src0, src1, _channels, _spatial, dst0, _format);
                    src0 += _channels * _spatial * _sizeT;
                    src1 += _channels * _sizeT;
                    dst0 += _channels * _spatial * _sizeT;
                }
                break;
            }
            case SpecialBatch:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _uniform(src0, src1, _spatial, dst0);
                    src0 += _channels * _spatial * _sizeT;
                    dst0 += _channels * _spatial * _sizeT;
                }
                break;
            }
            case SpecialBiasChannelV2:
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _addBias(src0, src1, _channels, _spatial, dst0, _format);
                    dst0 += _channels * _spatial * _sizeT;
                    src1 += _channels * _sizeT;
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