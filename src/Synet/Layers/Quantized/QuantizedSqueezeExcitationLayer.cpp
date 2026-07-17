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

#include "Synet/Layers/Quantized/QuantizedSqueezeExcitationLayer.h"

#include "Synet/Layers/Math/ScaleLayer.h"
#include "Synet/Layers/InnerProduct/InnerProduct32fLayer.h"
#include "Synet/Layers/Activation/PreluLayer.h"
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

    QuantizedSqueezeExcitationLayer::QuantizedSqueezeExcitationLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    void QuantizedSqueezeExcitationLayer::CompactWeight()
    {
        ((Tensor&)this->Weight()[0]).Clear();
        ((Tensor&)this->Weight()[_sci]).Clear();
    }

    size_t QuantizedSqueezeExcitationLayer::MemoryUsage() const
    {
        return (_sumScale.size() + _sumShift.size() + _rWeight[0].size() + _rWeight[1].size() + _params.size()) * sizeof(float);
    }

    int64_t QuantizedSqueezeExcitationLayer::Flop() const
    {
        return _batch * (_channels * _height * _width * 2 + _squeeze * _channels * 4 + _squeeze * 2 + _channels * 22);
    }

    bool QuantizedSqueezeExcitationLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
#if !defined(SYNET_SIMD_LIBRARY_ENABLE) || defined(SYNET_SIMD_SYNET_DISABLE)
        SYNET_ERROR("QuantizedSqueezeExcitationLayer work only with SimdLibrary support!");
#endif
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType8u)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer input must have INT8 type!");
        const Tensors& weight = this->Weight();
        const SqueezeExcitationParam& param = this->Param().squeezeExcitation();
        _actType = param.activationType();
        _hasBias[0] = param.biasTerm0();
        _hasBias[1] = param.biasTerm1();
        _hardSigmoid = param.hardSigmoid();
        _params.resize(2);
        _params[0] = param.activationParam0();
        _params[1] = param.activationParam1();
        _sci = 1 + (_hasBias[0] ? 1 : 0) + (_actType == ActivationFunctionTypePrelu ? 1 : 0);
        if(weight.size() != _sci + 1 + (_hasBias[1] ? 1 : 0))
            SYNET_ERROR("QuantizedSqueezeExcitationLayer: check weight count!");
        if(weight[0].Count() != 4 || weight[_sci].Count() != 4)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer: check weight dims!");

        _format = src[0]->Format();
        _batch = src[0]->Axis(0);
        if (_format == TensorFormatNchw)
        {
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
            _squeeze = weight[0].Axis(0);
            if(weight[_sci].Axis(0) != _channels)
                SYNET_ERROR("QuantizedSqueezeExcitationLayer: check weight[" << _sci << "] axis 0!");
        }
        else if (_format == TensorFormatNhwc)
        {
            _height = src[0]->Axis(1);
            _width = src[0]->Axis(2);
            _channels = src[0]->Axis(3);
            _squeeze = weight[0].Axis(3);
            if(weight[_sci].Axis(3) != _channels)
                SYNET_ERROR("QuantizedSqueezeExcitationLayer: check weight[" << _sci << "] axis 3!");
        }
        else
            assert(0);
        _size = _channels * _height * _width;
        _kAvg = 1.0f / (_height * _width);

        Layer::Extend8u(buf, 0, Shp(_channels + _squeeze));

        if (src[0] != dst[0])
        {
            if (TensorUsers(Param().src()[0]) == 1 && !src[0]->Const())
                dst[0]->Share(*src[0]);
            else
                dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        this->UsePerfStat();
        return true;
    }

    LowPrecisionType QuantizedSqueezeExcitationLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    void QuantizedSqueezeExcitationLayer::Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
        assert(0);
    }
}