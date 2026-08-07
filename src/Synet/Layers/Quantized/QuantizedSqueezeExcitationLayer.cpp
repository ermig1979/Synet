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
    QuantizedSqueezeExcitationLayer::QuantizedSqueezeExcitationLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    void QuantizedSqueezeExcitationLayer::CompactWeight()
    {
        ((Tensor&)this->Weight()[0]).Clear();
        ((Tensor&)this->Weight()[_spi]).Clear();
    }

    size_t QuantizedSqueezeExcitationLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _params.size() * sizeof(float) +
            _quantizedInnerProduct[0].InternalBufferSize() + _quantizedInnerProduct[1].InternalBufferSize();
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
        const LayerParam& param = this->Param();
        const SqueezeExcitationParam& seParam = param.squeezeExcitation();
        _actType = seParam.activationType();
        if(_actType != ActivationFunctionTypeIdentity)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer doesn't support any primarily activation function!");
        _hasBias[0] = seParam.biasTerm0();
        _hasBias[1] = seParam.biasTerm1();
        _hardSigmoid = seParam.hardSigmoid();
        if (!_hardSigmoid)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer support only HardSigmoid secondary activation function!");
        _params.resize(2);
        _params[0] = seParam.activationParam0();
        _params[1] = seParam.activationParam1();
        _scale = 1.0f / 6.0f;
        _shift = 0.5f;

        size_t qSrcSize = 1 + (_hasBias[0] ? 3 : 2) + (_actType != ActivationFunctionTypeIdentity ? 1 : 0) +
            (_hasBias[1] ? 3 : 2) + 1 + 1;
        if (param.qSrc().size() < qSrcSize)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer must have at least " << qSrcSize << " input dequantizers!");

        _format = src[0]->Format();
        _batch = src[0]->Axis(0);
        if (_format == TensorFormatNchw)
        {
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
            _squeeze = weight[0].Axis(0);
        }
        else if (_format == TensorFormatNhwc)
        {
            _height = src[0]->Axis(1);
            _width = src[0]->Axis(2);
            _channels = src[0]->Axis(3);
            _squeeze = weight[0].Axis(3);
        }
        else
            assert(0);

        _srcScale = float(param.qSrc()[0].scale());
        _srcZero = param.qSrc()[0].zero();
        if (param.qSrc()[0].weights() != 0)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer supports only uniform input quantization!");

        _avgScale = float(param.qSrc()[1].scale());
        _avgZero = param.qSrc()[1].zero();
        if (param.qSrc()[1].weights() != 0)
            SYNET_ERROR("QuantizedSqueezeExcitationLayer supports only uniform averaging quantization!");

        Layer::Extend8u(buf, 0, Shp(_batch, _channels + _squeeze));

        int weight0 = 0;
        int bias0 = weight0 + param.qSrc()[2].weights();
        int dst0 = _hasBias[0] ? 4 : 3;
        _ipScale[0] = float(param.qSrc()[dst0].scale());
        _ipZero[0] = param.qSrc()[dst0].zero();
        
        _quantizedInnerProduct[0].Init(_batch, _squeeze, _channels, TensorType8u, TensorType8i, TensorType8u, _format == TensorFormatNchw ? 1 : 0, 1, _hasBias[0] ? 1 : 0);
        if (_quantizedInnerProduct[0].Enable())
        {
            Layer::Extend8u(buf, 1, Shp(_quantizedInnerProduct[0].ExternalBufferSize()));
            uint8_t srcZero = (uint8_t)_avgZero, dstZero = (uint8_t)_ipZero[0];
            _quantizedInnerProduct[0].SetParams(&_avgScale, &srcZero, weight[weight0 + 0].Data<int8_t>(), weight[weight0 + 1].Data<float>(),
                _hasBias[0] ? weight[bias0 + 0].Data<int32_t>() : NULL, &_ipScale[0], &dstZero);
        }
        else
            SYNET_ERROR("QuantizedSqueezeExcitationLayer can't initalize primarily QuantizedInnerProduct!");

        int weight1 = bias0 + (_hasBias[0] ? param.qSrc()[3].weights() : 2);
        int bias1 = weight1 + param.qSrc()[dst0 + 1].weights();
        int dst1 = dst0 + (_hasBias[1] ? 3 : 2);
        _ipScale[1] = float(param.qSrc()[dst1].scale());
        _ipZero[1] = param.qSrc()[dst1].zero();
        _spi = weight1;

        _quantizedInnerProduct[1].Init(_batch, _channels, _squeeze, TensorType8u, TensorType8i, TensorType8u, _format == TensorFormatNchw ? 1 : 0, 1, _hasBias[1] ? 1 : 0);
        if (_quantizedInnerProduct[1].Enable())
        {
            Layer::Extend8u(buf, 1, Shp(_quantizedInnerProduct[1].ExternalBufferSize()));
            uint8_t srcZero = (uint8_t)_ipZero[0], dstZero = (uint8_t)_ipZero[1];
            _quantizedInnerProduct[1].SetParams(&_ipScale[0], &srcZero, weight[weight1 + 0].Data<int8_t>(), weight[weight1 + 1].Data<float>(),
                _hasBias[1] ? weight[bias1 + 0].Data<int32_t>() : NULL, &_ipScale[1], &dstZero);
        }
        else
            SYNET_ERROR("QuantizedSqueezeExcitationLayer can't initalize secondary QuantizedInnerProduct!");

        _actScale[1] = float(param.qSrc()[dst1 + 1].scale());
        _actZero[1] = param.qSrc()[dst1 + 1].zero();

        _dstScale = float(param.qDst()[0].scale());
        _dstZero = param.qDst()[0].zero();

        _quantizedMul.Init(Shp(_batch, _channels), TensorType8u, _actScale[1], _actZero[1], 
            src[0]->Shape(), TensorType8u, _srcScale, _srcZero, TensorType8u, _dstScale, _dstZero);
        if(!_quantizedMul.Enable())
            SYNET_ERROR("QuantizedSqueezeExcitationLayer can't initalize QuantizedMul!");

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
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        uint8_t* buf0 = Layer::Buf8u(buf, 0), * buf1 = buf0 + _batch * _channels, * buf2 = Layer::Buf8u(buf, 1);
        const uint8_t* src0 = src[0]->Data<uint8_t>();
        uint8_t* dst0 = dst[0]->Data<uint8_t>();
        SimdSynetQuantizedPoolingAverage(src0, &_srcScale, _srcZero, _batch, _channels, _height, _width, _height, _width,
            1, 1, 0, 0, SimdTrue, buf0, &_avgScale, _avgZero, 1, 1, (SimdTensorFormatType)_format);
        _quantizedInnerProduct[0].Forward(buf0, NULL, buf2, buf1);
        _quantizedInnerProduct[1].Forward(buf1, NULL, buf2, buf0);
        SimdSynetQuantizedHardSigmoid(buf0, &_ipScale[1], _ipZero[1], _batch*_channels, &_scale, &_shift, buf0, &_actScale[1], _actZero[1]);
        _quantizedMul.Forward(buf0, src0, dst0);
#else
        assert(0);
#endif
    }
}