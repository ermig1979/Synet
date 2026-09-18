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

#include "Synet/Layers/Math/ScaleLayer.h"

#include "Synet/Quantization/Bf16.h"

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

    ScaleLayer::ScaleLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType ScaleLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType16b && Options().BFloat16Enable())
            return p.src()[0] != p.dst()[0] ? LowPrecisionTypeActive : LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    bool ScaleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        const ScaleParam & param = this->Param().scale();
        _axis = src[0]->Index(param.axis());
        _biasTerm = param.biasTerm();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ScaleLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType16b)
            SYNET_ERROR("ScaleLayer input must have FP32 or BF16 type!");
        if (dst[0]->GetType() != TensorType32f && dst[0]->GetType() != TensorType16b)
            SYNET_ERROR("ScaleLayer output must have FP32 or BF16 type!");
        _src16b = src[0]->GetType() == TensorType16b;
        _dst16b = dst[0]->GetType() == TensorType16b;
        _format = src[0]->Format();
        _processFormat = _format;
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
        const Tensor & scale = this->Weight()[0];
        _channels = scale.Size();
        if (_src16b || _dst16b)
        {
            if (_biasTerm)
                _shift.Share(this->Weight()[1]);
            else
                _shift.Reshape(TensorType32f, Shp(_channels), TensorFormatUnknown, 0.0f);
        }
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
            if (_axis == 3)
            {
                _processFormat = TensorFormatNhwc;
                _batch = src[0]->Axis(0);
                _height = src[0]->Axis(1);
                _width = src[0]->Axis(2);
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
            if (src[0]->Count() == 3 && _format == TensorFormatNchw && src[0]->Axis(2) == _channels && src[0]->Axis(1) != _channels)
            {
                _processFormat = TensorFormatNhwc;
                _batch = src[0]->Axis(0);
                _height = 1;
                _width = src[0]->Axis(1);
            }
        }
        if (src[0]->Size() != _batch * _channels * _height * _width)
            SYNET_ERROR("ScaleLayer: can't process input shape: " << ToStr(src[0]->Shape()) << " for weight size " << _channels << " and axis " << _axis << " !");
        if (src[0] != dst[0])
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
            if (_src16b || _dst16b)
                _scale16b.Init(_channels, _height * _width, (SimdTensorDataType)src[0]->GetType(),
                    (SimdTensorDataType)dst[0]->GetType(), (SimdTensorFormatType)_format, SimdTrue, _biasTerm ? SimdTrue : SimdFalse);
#endif
            if (_src16b == _dst16b && TensorUsers(Param().src()[0]) == 1 && !src[0]->Const())
                dst[0]->Share(*src[0]);
            else
            {
                if (_dst16b)
                    dst[0]->Reshape(TensorType16b, src[0]->Shape(), _format);
                else
                    dst[0]->Reshape(TensorType32f, src[0]->Shape(), _format);
            }
        }
        if (src[0]->Const())
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            if (Options().BFloat16Enable())
            {
                UsePerfStat(ToChar(src[0]->GetType()) + ToChar(dst[0]->GetType()) + " " + Cpl::ToStr(_batch) + "x" + Cpl::ToStr(_channels) + "x" + Cpl::ToStr(_height) + "x" + Cpl::ToStr(_width));
            }
            else
                UsePerfStat();
            _const = false;
        }
        _compatibility = 1;
        return true;
    }

    size_t ScaleLayer::MemoryUsage() const
    { 
        return Layer::MemoryUsage() + _shift.MemoryUsage();
    }

    int64_t ScaleLayer::Flop() const
    {
        return _batch * _channels * _height * _width * 2;
    }

    void ScaleLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if (_src16b || _dst16b)
        {
            const float* scale = this->Weight()[0].Data<float>();
            const float* shift = _shift.Data<float>();
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
            if (_scale16b.Enable())
            {
                const uint8_t* src8 = src[0]->RawData();
                uint8_t* dst8 = dst[0]->RawData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    _scale16b.Forward(src8, scale, shift, dst8);
                    src8 += _channels * _height * _width * (_src16b ? 2 : 4);
                    dst8 += _channels * _height * _width * (_dst16b ? 2 : 4);
                }
            }
            else
#endif
            {
                if (_src16b && _dst16b)
                    ScaleForward16b(src[0]->Data<uint16_t>(), _batch, _channels, _height * _width, _processFormat, scale, shift, dst[0]->Data<uint16_t>());
                else if (!_src16b && _dst16b)
                    ScaleForward16b(src[0]->Data<float>(), _batch, _channels, _height * _width, _processFormat, scale, shift, dst[0]->Data<uint16_t>());
                else if (_src16b && !_dst16b)
                    ScaleForward16b(src[0]->Data<uint16_t>(), _batch, _channels, _height * _width, _processFormat, scale, shift, dst[0]->Data<float>());
            }
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
            ScaleForward32f(src, scale, bias, _channels, _height, _width, dst, _processFormat, _compatibility);
            src += _channels * _height * _width;
            dst += _channels * _height * _width;
        }
    }
}
