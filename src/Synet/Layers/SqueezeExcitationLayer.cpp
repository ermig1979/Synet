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

#include "Synet/Layers/SqueezeExcitationLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/InnerProduct32fLayer.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Quantization/Convert.h"
#include "Synet/Quantization/Bf16.h"

namespace Synet
{
    template <class T, class S> void ChannelSum(const T* src, size_t channels, size_t height, size_t width, TensorFormat format, S * sum)
    {
        //SYNET_PERF_FUNC();
        if (format == TensorFormatNhwc)
        {
            for (size_t c = 0; c < channels; ++c)
                sum[c] = S(0);
            for (size_t h = 0; h < height; ++h)
            {
                for (size_t w = 0; w < width; ++w)
                {
                    for (size_t c = 0; c < channels; ++c)
                        sum[c] += Convert<T, S>(src[c]);
                    src += channels;
                }
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                sum[c] = S(0);
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                        sum[c] += Convert<T, S>(src[w]);
                    src += width;
                }
            }
        }
        else
            assert(0);
    }
#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template <> inline void ChannelSum<uint16_t, float>(const uint16_t* src, size_t channels, size_t height, size_t width, TensorFormat format, float* sum)
    {
        size_t spatial = height * width;
        SimdSynetChannelSum16b(src, channels, spatial, (SimdTensorFormatType)format, sum);
    }

    template <> inline void ChannelSum<uint8_t, int32_t>(const uint8_t* src, size_t channels, size_t height, size_t width, TensorFormat format, int32_t* sum)
    {
        //SYNET_PERF_FUNC();
        size_t spatial = height * width;
        if (format == TensorFormatNhwc)
            SimdGetColSums(src, channels, channels, spatial, (uint32_t*)sum);
        else if (format == TensorFormatNchw)
            SimdGetRowSums(src, spatial, spatial, channels, (uint32_t*)sum);
        else
            assert(0);
    }
#endif

    //-------------------------------------------------------------------------------------------------

    SqueezeExcitationLayer::SqueezeExcitationLayer(const LayerParam& param, Context* context, QuantizationMethod method)
        : Layer(param, context)
        , _method(method)
    {
    }

    void SqueezeExcitationLayer::CompactWeight()
    {
        ((Tensor&)this->Weight()[0]).Clear();
        ((Tensor&)this->Weight()[1]).Clear();
    }

    size_t SqueezeExcitationLayer::MemoryUsage() const
    {
        return (_sumScale.size() + _sumShift.size() + _rWeight[0].size() + _rWeight[1].size())*sizeof(float) + _scale8i.InternalBufferSize();
    }

    int64_t SqueezeExcitationLayer::Flop() const
    {
        return _batch * (_channels * _height * _width * 2 + _squeeze * _channels * 4 + _squeeze * 2 + _channels * 22);
    }

    bool SqueezeExcitationLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const Tensors& weight = this->Weight();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SqueezeExcitationLayer supports only 1 input and 1 output!");
        if(weight[0].Count() != 4 || weight[1].Count() != 4)
            SYNET_ERROR("SqueezeExcitationLayer: check weight size!");
        if (src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType16b && src[0]->GetType() != TensorType8u)
            SYNET_ERROR("SqueezeExcitationLayer input must have FP32, BF16 or INT8 type!");
        if (dst[0]->GetType() != TensorType32f && dst[0]->GetType() != TensorType16b && dst[0]->GetType() != TensorType8u)
            SYNET_ERROR("SqueezeExcitationLayer output must have FP32, BF16 or INT8 type!");

        _src16b = src[0]->GetType() == TensorType16b;
        _dst16b = dst[0]->GetType() == TensorType16b;
        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = dst[0]->GetType() == TensorType8u;
        _format = src[0]->Format();
        _batch = src[0]->Axis(0);
        if (_format == TensorFormatNchw)
        {
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
            _squeeze = weight[0].Axis(0);
            if(weight[1].Axis(0) != _channels)
                SYNET_ERROR("SqueezeExcitationLayer: check weight[1] axis 0!");
            assert(weight[1].Axis(0) == _channels);
            _rWeight[0].assign(weight[0].Data<float>(), weight[0].Data<float>() + _squeeze * _channels);
            _rWeight[1].assign(weight[1].Data<float>(), weight[1].Data<float>() + _squeeze * _channels);
        }
        else if (_format == TensorFormatNhwc)
        {
            _height = src[0]->Axis(1);
            _width = src[0]->Axis(2);
            _channels = src[0]->Axis(3);
            _squeeze = weight[0].Axis(3);
            if(weight[1].Axis(3) != _channels)
                SYNET_ERROR("SqueezeExcitationLayer: check weight[1] axis 3!");
            _rWeight[0].resize(_squeeze * _channels);
            for (size_t s = 0; s < _squeeze; ++s)
                for (size_t c = 0; c < _channels; ++c)
                    _rWeight[0][s * _channels + c] = weight[0].Data<float>()[c * _squeeze + s];
            _rWeight[1].resize(_squeeze * _channels);
            for (size_t c = 0; c < _channels; ++c)
                for (size_t s = 0; s < _squeeze; ++s)
                    _rWeight[1][c * _squeeze + s] = weight[1].Data<float>()[s * _channels + c];
        }
        else
            assert(0);
        _size = _channels * _height * _width;
        _kAvg = 1.0f / (_height * _width);

        if (_src8u)
        {
            Layer::Extend32i(buf, 0, Shp(_channels));
            Init8i();
        }
        else
            Layer::Extend32f(buf, 0, Shp(_channels));
        Layer::Extend32f(buf, 1, Shp(_channels + _squeeze));

        if (src[0] != dst[0])
        {
            if(_dst8u)
                dst[0]->Reshape(TensorType8u, src[0]->Shape(), _format);
            else if (_dst16b)
                dst[0]->Reshape(TensorType16b, src[0]->Shape(), _format);
            else
                dst[0]->Reshape(TensorType32f, src[0]->Shape(), _format);
        }
        if (_src8u)
        {
            _scale8i.Init(1, _channels, _height * _width, src[0]->GetType(), dst[0]->GetType(), _format, _method);
            if (_scale8i.Enable())
            {
                const float* stats[4] = {
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data() };
                _scale8i.SetParams(_sumScale.data(), NULL, stats);
            }
        }
        if (Options().BFloat16Enable())
            this->UsePerfStat(ToChar(src[0]->GetType()) + ToChar(dst[0]->GetType()));
        else
            this->UsePerfStat();
        return true;
    }

    LowPrecisionType SqueezeExcitationLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType8u && _method != QuantizationMethodUnknown)
            return LowPrecisionTypeActive;
        if (type == TensorType16b && Options().BFloat16Enable() && _method == QuantizationMethodUnknown)
            return p.src()[0] != p.dst()[0] ? LowPrecisionTypeActive : LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    void SqueezeExcitationLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        float * norm0 = Layer::Buf32f(buf, 1);
        float * norm1 = norm0 + _channels;
        if (_src8u)
        {
            if(_dst8u)
                Forward8i(src[0]->Data<uint8_t>(), Layer::Buf32i(buf, 0), norm0, norm1, dst[0]->Data<uint8_t>(), NULL);
            else
                Forward8i(src[0]->Data<uint8_t>(), Layer::Buf32i(buf, 0), norm0, norm1, NULL, dst[0]->Data<float>());
        }
        else if (_src16b)
        {
            if (_dst16b)
                Forward16b(src[0]->Data<uint16_t>(), Layer::Buf32f(buf, 0), norm0, norm1, dst[0]->Data<uint16_t>(), NULL);
            else
                Forward16b(src[0]->Data<uint16_t>(), Layer::Buf32f(buf, 0), norm0, norm1, NULL, dst[0]->Data<float>());
        }
        else
            Forward32f(src[0]->Data<float>(), Layer::Buf32f(buf, 0), norm0, norm1, dst[0]->Data<float>());
    }

    void SqueezeExcitationLayer::Forward32f(const float * src, float * sum, float * norm0, float * norm1, float * dst)
    {
        for (size_t b = 0; b < _batch; ++b)
        {
            ChannelSum(src, _channels, _height, _width, _format, sum);
            CpuScale(sum, _channels, _kAvg, norm0);
            Detail::InnerProductLayerForwardCpu<float>(norm0, _rWeight[0].data(), NULL, _squeeze, _channels, norm1);
            CpuRelu(norm1, _squeeze, 0.0f, norm1);
            Detail::InnerProductLayerForwardCpu<float>(norm1, _rWeight[1].data(), NULL, _channels, _squeeze, norm0);
            CpuSigmoid(norm0, _channels, norm0);
            ScaleForward32f(src, norm0, NULL, _channels, _height, _width, dst, _format, 0);
            src += _size, dst += _size;
        }
    }

    void SqueezeExcitationLayer::Forward8i(const uint8_t* src, int32_t * sum, float* norm0, float* norm1, uint8_t* dst8u, float* dst32f)
    {
        for (size_t b = 0; b < _batch; ++b)
        {
            ChannelSum(src, _channels, _height, _width, _format, sum);
            Detail::Convert<int32_t, float, float>(sum, 1, _channels, 1, 1, TensorFormatNhwc, 
                _sumScale.data(), _sumShift.data(), INT_MIN, INT_MAX, norm0);
            Detail::InnerProductLayerForwardCpu<float>(norm0, _rWeight[0].data(), NULL, _squeeze, _channels, norm1);
            CpuRelu(norm1, _squeeze, 0.0f, norm1);
            Detail::InnerProductLayerForwardCpu<float>(norm1, _rWeight[1].data(), NULL, _channels, _squeeze, norm0);
            CpuSigmoid(norm0, _channels, norm0);
            if(_dst8u)
                Scale8i(src, norm0, dst8u), dst8u += _size;
            else
                Scale8i(src, norm0, dst32f), dst32f += _size;
            src += _size;
        }
    }

    void SqueezeExcitationLayer::Init8i()
    {
        _sumScale.resize(_channels);
        _sumShift.resize(_channels);
        Stat& statS = *this->Stats(0)[0];
        Stat& statD = *this->Stats(2)[0];
        statS.Init8u(_method);
        statD.Init8u(_method);
        for (size_t c = 0; c < _channels; c++)
        {
            _sumScale[c] = statS.scale8uTo32f[c] * _kAvg;
            _sumShift[c] = statS.shift8uTo32f[c];
        }
    }

    void SqueezeExcitationLayer::Scale8i(const uint8_t* src, float* norm, uint8_t* dst)
    {
        if (_scale8i.Enable())
        {
            _scale8i.SetParams(norm, NULL, NULL);
            _scale8i.Forward(src, dst);
            return;
        }
        int lower, upper;
        if (_method == QuantizationMethodIECompatible)
            lower = QUANT_IE_COMP_SRC_U8_MIN, upper = QUANT_IE_COMP_SRC_U8_MAX;
        else if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
            lower = QUANT_SYMM_NARR_SRC_U8_MIN, upper = QUANT_SYMM_NARR_SRC_U8_MAX;
        const float* srcScale = this->Stats(0)[0]->scale8uTo32f.data();
        const float* srcShift = this->Stats(0)[0]->shift8uTo32f.data();
        const float* dstScale = this->Stats(2)[0]->scale32fTo8u.data();
        const float* dstShift = this->Stats(2)[0]->shift32fTo8u.data();
        if (_format == TensorFormatNchw)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                for (size_t h = 0; h < _height; ++h)
                {
                    for (size_t w = 0; w < _width; ++w)
                    {
                        float value = Detail::Convert<uint8_t, float, float>(src[w], srcScale[c], srcShift[c], INT_MIN, INT_MAX);
                        dst[w] = Detail::Convert<float, uint8_t, float>(value*norm[c], dstScale[c], dstShift[c], lower, upper);
                    }
                    dst += _width; 
                    src += _width;
                }
            }
        }
        else if (_format == TensorFormatNhwc)
        {
            for (size_t h = 0; h < _height; ++h)
            {
                for (size_t w = 0; w < _width; ++w)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        float value = Detail::Convert<uint8_t, float, float>(src[c], srcScale[c], srcShift[c], INT_MIN, INT_MAX);
                        dst[c] = Detail::Convert<float, uint8_t, float>(value * norm[c], dstScale[c], dstShift[c], lower, upper);
                    }
                    dst += _channels;
                    src += _channels;
                }
            }
        }
        else
            assert(0);
    }

    void SqueezeExcitationLayer::Scale8i(const uint8_t* src, float* norm, float* dst)
    {
        if (_scale8i.Enable())
        {
            _scale8i.SetParams(norm, NULL, NULL);
            _scale8i.Forward(src, (uint8_t*)dst);
            return;
        }
        const float* srcScale = this->Stats(0)[0]->scale8uTo32f.data();
        const float* srcShift = this->Stats(0)[0]->shift8uTo32f.data();
        if (_format == TensorFormatNchw)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                for (size_t h = 0; h < _height; ++h)
                {
                    for (size_t w = 0; w < _width; ++w)
                        dst[w] = Detail::Convert<uint8_t, float, float>(src[w], srcScale[c], srcShift[c], INT_MIN, INT_MAX) * norm[c];
                    dst += _width;
                    src += _width;
                }
            }
        }
        else if (_format == TensorFormatNhwc)
        {
            for (size_t h = 0; h < _height; ++h)
            {
                for (size_t w = 0; w < _width; ++w)
                {
                    for (size_t c = 0; c < _channels; ++c)
                        dst[c] = Detail::Convert<uint8_t, float, float>(src[c], srcScale[c], srcShift[c], INT_MIN, INT_MAX) * norm[c];
                    dst += _channels;
                    src += _channels;
                }
            }
        }
        else
            assert(0);
    }

    void SqueezeExcitationLayer::Forward16b(const uint16_t* src, float* sum, float* norm0, float* norm1, uint16_t* dst16b, float* dst32f)
    {
        for (size_t b = 0; b < _batch; ++b)
        {
            ChannelSum(src, _channels, _height, _width, _format, sum);
            CpuScale(sum, _channels, _kAvg, norm0);
            Detail::InnerProductLayerForwardCpu<float>(norm0, _rWeight[0].data(), NULL, _squeeze, _channels, norm1);
            CpuRelu(norm1, _squeeze, 0.0f, norm1);
            Detail::InnerProductLayerForwardCpu<float>(norm1, _rWeight[1].data(), NULL, _channels, _squeeze, norm0);
            CpuSigmoid(norm0, _channels, norm0);
            if (_dst16b)
                Scale16b(src, norm0, dst16b), dst16b += _size;
            else
                Scale16b(src, norm0, dst32f), dst32f += _size;
            src += _size;
        }
    }

    void SqueezeExcitationLayer::Scale16b(const uint16_t* src, float* norm, uint16_t* dst)
    {
        //SYNET_PERF_FUNC();
        if (_format == TensorFormatNchw)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                for (size_t h = 0; h < _height; ++h)
                {
                    for (size_t w = 0; w < _width; ++w)
                        dst[w] = Float32ToBFloat16(BFloat16ToFloat32(src[w]) * norm[c]);
                    dst += _width;
                    src += _width;
                }
            }
        }
        else if (_format == TensorFormatNhwc)
        {
            for (size_t h = 0; h < _height; ++h)
            {
                for (size_t w = 0; w < _width; ++w)
                {
                    for (size_t c = 0; c < _channels; ++c)
                        dst[c] = Float32ToBFloat16(BFloat16ToFloat32(src[c]) * norm[c]);
                    dst += _channels;
                    src += _channels;
                }
            }
        }
        else
            assert(0);
    }

    void SqueezeExcitationLayer::Scale16b(const uint16_t* src, float* norm, float* dst)
    {
        //SYNET_PERF_FUNC();
        if (_format == TensorFormatNchw)
        {
            for (size_t c = 0; c < _channels; ++c)
            {
                for (size_t h = 0; h < _height; ++h)
                {
                    for (size_t w = 0; w < _width; ++w)
                        dst[w] = BFloat16ToFloat32(src[w]) * norm[c];
                    dst += _width;
                    src += _width;
                }
            }
        }
        else if (_format == TensorFormatNhwc)
        {
            for (size_t h = 0; h < _height; ++h)
            {
                for (size_t w = 0; w < _width; ++w)
                {
                    for (size_t c = 0; c < _channels; ++c)
                        dst[c] = BFloat16ToFloat32(src[c]) * norm[c];
                    dst += _channels;
                    src += _channels;
                }
            }
        }
        else
            assert(0);
    }
}