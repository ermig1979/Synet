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

#include "Synet/Layers/Quantized/QuantizedAddLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

#include "Synet/Utils/Activation.h"

namespace Synet
{
    template<ActivationFunctionType activation, class D> void QuantizedAddUniformV0(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst8)
    {
        D* dst = (D*)dst8;
        int aBias = -aZero, bBias = -bZero;
        if (std::is_same<D, uint8_t>())
        {
            float scale = 1.0f / dScale;
            for (size_t i = 0; i < size; ++i)
            {
                float _a = DequantizeLinear(a[i], aBias, aScale);
                float _b = DequantizeLinear(b[i], bBias, bScale);
                dst[i] = (D)QuantizeLinear(Activation<activation>(_a + _b, 0, params), scale, dZero, 0, 255);
            }
        }
        if (std::is_same<D, float>())
        {
            for (size_t i = 0; i < size; ++i)
            {
                float _a = DequantizeLinear(a[i], aBias, aScale);
                float _b = DequantizeLinear(b[i], bBias, bScale);
                dst[i] = (D)Activation<activation>(_a + _b, 0, params);
            }
        }
    }

    template<ActivationFunctionType activation, class D> void QuantizedAddUniformV1(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst8)
    {
        D* dst = (D*)dst8;
        if (std::is_same<D, uint8_t>())
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - adScale * float(aZero) - bdScale * float(bZero);
            for (size_t i = 0; i < size; ++i)
            {
                float val = float(a[i]) * adScale + float(b[i]) * bdScale + term;
                dst[i] = RestrictRange(NearByInt(val), 0, 255);
            }
        }
        if (std::is_same<D, float>())
        {
            float term = - aScale * float(aZero) - bScale * float(bZero);
            for (size_t i = 0; i < size; ++i)
                dst[i] = Activation<activation>(float(a[i]) * aScale + float(b[i]) * bScale + term, 0, params);
        }
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedAddLayer::QuantizedAddLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _uniform(NULL)
    {
    }

    int64_t QuantizedAddLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * (_dstType == TensorType8u ? 7 : 5);
    }

    LowPrecisionType QuantizedAddLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedAddLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("QuantizedAddLayer supports only 2 inputs and 1 output!");
        if (src[0]->Shape() != src[1]->Shape() && src[0]->Size() != src[1]->Size())
            SYNET_ERROR("QuantizedAddLayer supports only inputs with the same shape!");

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 2)
            SYNET_ERROR("QuantizedAddLayer must have 2 inputs quantization parameters!");
        if (src[0]->GetType() != TensorType8u || src[1]->GetType() != TensorType8u)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 inputs!");

        _aScale = float(param.qSrc()[0].scale());
        _aZero = param.qSrc()[0].zero();
        _bScale = float(param.qSrc()[1].scale());
        _bZero = param.qSrc()[1].zero();

        if (param.qDst().size())
        {
            _dstType = param.qDst()[0].type();
            _dZero = param.qDst()[0].zero();
            _dScale = float(param.qDst()[0].scale());
        }
        else
        {
            _dstType = TensorType32f;
            _dZero = 0;
            _dScale = 1.0f;
        }
        if(_dstType != TensorType8u && _dstType != TensorType32f)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 or FP32 output!");

        const ActivationParam& activ = param.activation();
        _size = src[0]->Size();
        _activationType = activ.type();
        _params[0] = activ.param0();
        _params[1] = activ.param1();

        //_quantizedAdd.Init(
        //    src[0]->Shape(), src[0]->GetType(), _aBias, _aNorm,
        //    src[1]->Shape(), src[1]->GetType(), _bBias, _bNorm,
        //    _activationType, _params, _dstType, _scale, _zero);
        if (!_quantizedAdd.Enable())
        {
            switch (_activationType)
            {
            case ActivationFunctionTypeIdentity:
                _uniform = _dstType == TensorType8u ? 
                    QuantizedAddUniformV1<ActivationFunctionTypeIdentity, uint8_t> :
                    QuantizedAddUniformV0<ActivationFunctionTypeIdentity, float>;
                break;
            case ActivationFunctionTypeRelu:
                _uniform = _dstType == TensorType8u ?
                    QuantizedAddUniformV1<ActivationFunctionTypeIdentity, uint8_t> :
                    QuantizedAddUniformV0<ActivationFunctionTypeRelu, float>;
                break;
            default:
                SYNET_ERROR("QuantizedAddLayer does not support " << Cpl::ToStr(_activationType) << " !");
            }
        }

        dst[0]->Reshape(_dstType, src[0]->Shape(), src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
        {
            ForwardCpu(src, buf, dst);
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

    void QuantizedAddLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_quantizedAdd.Enable())
            _quantizedAdd.Forward(src[0]->RawData(), src[1]->RawData(), dst[0]->RawData());
        else if(_uniform)
            _uniform(src[0]->Data<uint8_t>(), _aScale, _aZero, src[1]->Data<uint8_t>(), _bScale, _bZero, _size, _params, _dScale, _dZero, dst[0]->RawData());
    }
}