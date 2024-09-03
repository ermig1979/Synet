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

#include "Synet/Layers/ScaleLayer.h"

#include "Synet/Quantization/Bf16.h"

#include "Synet/Utils/Math.h"

namespace Synet
{
    void ScaleForward32f(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, TensorFormat format, int compatibility)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetScaleLayerForward(src, scale, bias, channels, height, width, dst, (SimdTensorFormatType)format, (SimdSynetCompatibilityType)compatibility);
#else
        if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                const float s = scale[c];
                const float b = bias ? bias[c] : 0;
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        dst[w] = src[w] * s + b;
                    }
                    src += width;
                    dst += width;
                }
            }
        }
        else if (format == TensorFormatNhwc)
        {
            if (bias)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * scale[c] + bias[c];
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * scale[c];
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }
        else
            assert(0);
#endif        
    }

    //-------------------------------------------------------------------------------------------------

    template<class S, class D> D ScaleForward8i(S value, float scale, float shift, int lower, int upper);

    template<> SYNET_INLINE float ScaleForward8i<uint8_t, float>(uint8_t value, float scale, float shift, int lower, int upper)
    {
        return float(value) * scale + shift;
    }

    template<> SYNET_INLINE uint8_t ScaleForward8i<uint8_t, uint8_t>(uint8_t value, float scale, float shift, int lower, int upper)
    {
        return (uint8_t)Synet::RestrictRange(Round(float(value) * scale + shift), lower, upper);
    }

    template<> SYNET_INLINE uint8_t ScaleForward8i<float, uint8_t>(float value, float scale, float shift, int lower, int upper)
    {
        return (uint8_t)Synet::RestrictRange(Round(value * scale + shift), lower, upper);
    }

    template<> SYNET_INLINE float ScaleForward8i<float, float>(float value, float scale, float shift, int lower, int upper)
    {
        return value * scale + shift;
    }

    template<class S, class D> void ScaleForward8i(const S* src, size_t batch, size_t channels, size_t spatial,
        TensorFormat format, const float* scale, const float* shift, int lower, int upper, D* dst)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _shift = shift[c];
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = ScaleForward8i<S, D>(src[s], _scale, _shift, lower, upper);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = ScaleForward8i<S, D>(src[c], scale[c], shift[c], lower, upper);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class S, class D> void ScaleForward16b(const S* src, size_t batch, size_t channels, size_t spatial,
        TensorFormat format, const float* scale, const float* shift, D* dst)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _shift = shift[c];
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        float value = Convert<S, float>(src[s]);
                        dst[s] = Convert<float, D>(value * _scale + _shift);
                    }
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float value = Convert<S, float>(src[c]);
                        dst[c] = Convert<float, D>(value * scale[c] + shift[c]);
                    }
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    ScaleLayer::ScaleLayer(const LayerParam & param, Context* context, QuantizationMethod method)
        : Base(param, context)
        , _method(method)
    {
        _is8i = (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) &&
            param.scale().quantizationLevel() != TensorType32f;
    }

    bool ScaleLayer::Is8i() const
    {
        return _is8i;
    }

    bool ScaleLayer::Can8i() const
    {
        return _is8i;
    }

    bool ScaleLayer::Can16b() const
    {
        return Options().BFloat16Enable();
    }

    bool ScaleLayer::Is16b() const
    {
        const LayerParam& p = this->Param();
        return Options().BFloat16Enable() && !_is8i && p.src()[0] != p.dst()[0];
    }

    bool ScaleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const ScaleParam & param = this->Param().scale();
        _axis = param.axis();
        _biasTerm = param.biasTerm();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ScaleLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType16b && src[0]->GetType() != TensorType8u)
            SYNET_ERROR("ScaleLayer input must have FP32, BF16 or INT8 type!");
        if (dst[0]->GetType() != TensorType32f && dst[0]->GetType() != TensorType16b && dst[0]->GetType() != TensorType8u)
            SYNET_ERROR("ScaleLayer output must have FP32, BF16 or INT8 type!");
        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = dst[0]->GetType() == TensorType8u;
        _src16b = src[0]->GetType() == TensorType16b;
        _dst16b = dst[0]->GetType() == TensorType16b;
        _format = src[0]->Format();
        if (this->Weight().empty())
            SYNET_ERROR("ScaleLayer weights are absent!");
        if (_biasTerm)
        {
            if (this->Weight().size() < 2)
                SYNET_ERROR("ScaleLayer bias weight is absent!");
            if (this->Weight()[0].Shape() != this->Weight()[1].Shape())
            {
                if (SignificantDimsCount(this->Weight()[0].Shape()) != 1 || 
                    SignificantDimsCount(this->Weight()[1].Shape()) != 1 ||
                    this->Weight()[0].Size() != this->Weight()[1].Size())
                    SYNET_ERROR("ScaleLayer scale and bias weights have different shapes: " << ToStr(this->Weight()[0].Shape()) << " != " << ToStr(this->Weight()[1].Shape()) << "!");
            }
        }
        if (_src16b || _dst16b)
        {
            if (_biasTerm)
                _shift.Share(this->Weight()[1]);
            else
                _shift.Reshape(TensorType32f, Shp(_channels), TensorFormatUnknown, Type(0));
        }

        const Tensor & scale = this->Weight()[0];
        _channels = scale.Size();
        if (scale.Size() == src[0]->Size())
        {
            _batch = 1;
            _height = 1;
            _width = 1;
            if (_format == TensorFormatUnknown)
                _format = TensorFormatNchw;
        }
        else
        {
            _batch = src[0]->Size(0, _axis);
            if (src[0]->Count() < 4)
            {
                _height = 1;
                _width = src[0]->Size() / _batch / _channels;
            }
            else
            {
                _height = _format == TensorFormatNhwc ? src[0]->Axis(1) : src[0]->Axis(2);
                _width = _format == TensorFormatNhwc ? src[0]->Axis(2) : src[0]->Axis(3);
            }
        }
        if (src[0]->Size() != _batch * _channels * _height * _width)
            SYNET_ERROR("ScaleLayer: can't process input shape: " << ToStr(src[0]->Shape()) << " for weight size " << _channels << " and axis " << _axis << " !");
        if (_is8i)
        {
            _scale8i.Init(_batch, _channels, _height * _width, src[0]->GetType(), dst[0]->GetType(), _format, _method);
            if (_scale8i.Enable())
            {
                const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
                const float* stats[4] = {
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data() };
                _scale8i.SetParams(this->Weight()[0].Data<float>(), bias, stats);
            }
            else
                Init8i();
            if (_dst8u)
                dst[0]->Reshape(TensorType8u, src[0]->Shape(), _format);
            else
                dst[0]->Reshape(TensorType32f, src[0]->Shape(), _format);
        }
        else if (src[0] != dst[0])
        {
            if(_dst16b)
                dst[0]->Reshape(TensorType16b, src[0]->Shape(), _format);
            else
                dst[0]->Reshape(TensorType32f, src[0]->Shape(), _format);
        }
        this->UsePerfStat();
        _compatibility = 1;
        return true;
    }

    size_t ScaleLayer::MemoryUsage() const
    { 
        return Base::MemoryUsage() + _scale.MemoryUsage() + _shift.MemoryUsage() + _scale8i.InternalBufferSize();
    }

    void ScaleLayer::CompactWeight()
    {
        if (_is8i)
        {
            ((Tensor&)this->Weight()[0]).Clear();
            if(_biasTerm)
                ((Tensor&)this->Weight()[1]).Clear();
        }
    }

    int64_t ScaleLayer::Flop() const
    {
        return _batch * _channels * _height * _width * 2;
    }

    void ScaleLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_is8i)
        {
            if (_scale8i.Enable())
                _scale8i.Forward(src[0]->RawData(), dst[0]->RawData());
            else
            {
                const float* scale = _scale.Data<float>();
                const float* shift = _shift.Data<float>();
                if (_src8u && _dst8u)
                    ScaleForward8i(src[0]->Data<uint8_t>(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->Data<uint8_t>());
                else if (!_src8u && _dst8u)
                    ScaleForward8i(src[0]->Data<float>(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->Data<uint8_t>());
                else if (_src8u && !_dst8u)
                    ScaleForward8i(src[0]->Data<uint8_t>(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->Data<float>());
                else
                    ScaleForward8i(src[0]->Data<float>(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->Data<float>());
            }
        }
        else if (_src16b || _dst16b)
        {
            const float* scale = this->Weight()[0].Data<float>();
            const float* shift = _shift.Data<float>();
            if (_src16b && _dst16b)
                ScaleForward16b(src[0]->Data<uint16_t>(), _batch, _channels, _height * _width, _format, scale, shift, dst[0]->Data<uint16_t>());
            else if (!_src16b && _dst16b)
                ScaleForward16b(src[0]->Data<float>(), _batch, _channels, _height * _width, _format, scale, shift, dst[0]->Data<uint16_t>());
            else if (_src16b && !_dst16b)
                ScaleForward16b(src[0]->Data<uint16_t>(), _batch, _channels, _height * _width, _format, scale, shift, dst[0]->Data<float>());
        }
        else
            Scale32f(src[0]->Data<float>(), dst[0]->Data<float>());
    }

    void ScaleLayer::Scale32f(const float * src, float * dst)
    {
        const float* scale = this->Weight()[0].Data<float>();
        const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
        for (size_t b = 0; b < _batch; ++b)
        {
            ScaleForward32f(src, scale, bias, _channels, _height, _width, dst, _format, _compatibility);
            src += _channels * _height * _width;
            dst += _channels * _height * _width;
        }
    }

    void ScaleLayer::Init8i()
    {
        Stat& statS = *this->Stats(0)[0];
        Stat& statD = *this->Stats(2)[0];
        statS.Init8u(_method);
        statD.Init8u(_method);
        _scale.Reshape(Shp(_channels), 1.0f);
        _shift.Reshape(Shp(_channels), 0.0f);
        if (_src8u)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                _scale.Data<float>()[c] = statS.scale8uTo32f[c];
                _shift.Data<float>()[c] = statS.shift8uTo32f[c];
            }
        }
        const float* scale = this->Weight()[0].Data<float>();
        if (_biasTerm)
        {

            const float* bias = this->Weight()[1].Data<float>();
            for (size_t c = 0; c < _channels; ++c)
            {
                _scale.Data<float>()[c] = _scale.Data<float>()[c] * scale[c];
                _shift.Data<float>()[c] = _shift.Data<float>()[c] * scale[c] + bias[c];
            }
        }
        else
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                _scale.Data<float>()[c] = _scale.Data<float>()[c] * scale[c];
                _shift.Data<float>()[c] = _shift.Data<float>()[c] * scale[c];
            }
        }
        if (_dst8u)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                _scale.Data<float>()[c] = _scale.Data<float>()[c] * statD.scale32fTo8u[c];
                _shift.Data<float>()[c] = _shift.Data<float>()[c] * statD.scale32fTo8u[c] + statD.shift32fTo8u[c];
            }
        }
        if (_method == QuantizationMethodIECompatible)
            _lower = QUANT_IE_COMP_SRC_U8_MIN, _upper = QUANT_IE_COMP_SRC_U8_MAX;
        else if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
            _lower = QUANT_SYMM_NARR_SRC_U8_MIN, _upper = QUANT_SYMM_NARR_SRC_U8_MAX;
    }
}