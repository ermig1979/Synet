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
    template<ActivationFunctionType activation, class D> void QuantizedAaddUniform(const uint8_t* a, int aBias, float aNorm, const uint8_t* b, int bBias, float bNorm, size_t size, float* params, float scale, int zero, uint8_t* dst8)
    {
        D* dst = (D*)dst8;
        if (std::is_same<D, uint8_t>())
        {
            for (size_t i = 0; i < size; ++i)
            {
                float _a = DequantizeLinear(a[i], aBias, aNorm);
                float _b = DequantizeLinear(b[i], bBias, bNorm);
                dst[i] = QuantizeLinear(Activation<activation>(_a + _b, 0, params), scale, zero, 0, 255);
            }
        }
        if (std::is_same<D, float>())
        {
            for (size_t i = 0; i < size; ++i)
            {
                float _a = DequantizeLinear(a[i], aBias, aNorm);
                float _b = DequantizeLinear(b[i], bBias, bNorm);
                dst[i] = Activation<activation>(_a + _b, 0, params);
            }
        }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
#endif

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

    bool QuantizedAddLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("QuantizedAddLayer supports only 2 inputs and 1 output!");
        if (src[0]->Shape() != src[1]->Shape())
            SYNET_ERROR("QuantizedAddLayer supports only inputs with the same shape!");

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 2)
            SYNET_ERROR("QuantizedAddLayer must have 2 inputs quantization parameters!");
        if (src[0]->GetType() != TensorType8u || src[1]->GetType() != TensorType8u)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 inputs!");

        _aBias = -param.qSrc()[0].zero();
        _aNorm = float(param.qSrc()[0].scale());
        _bBias = -param.qSrc()[1].zero();
        _bNorm = float(param.qSrc()[1].scale());

        if (param.qDst().size())
        {
            _dstType = param.qDst()[0].type();
            _zero = param.qDst()[0].zero();
            _scale = 1.0f / float(param.qDst()[0].scale());
        }
        else
        {
            _dstType = TensorType32f;
            _zero = 0;
            _scale = 1.0f;
        }
        if(_dstType != TensorType8u && _dstType != TensorType32f)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 or FP32 output!");

        const ActivationParam& activ = param.activation();
        _size = src[0]->Size();
        _activationType = activ.type();
        _params[0] = activ.param0();
        _params[1] = activ.param1();

        switch (_activationType)
        {
        case ActivationFunctionTypeIdentity:
            _uniform = _dstType == TensorType8u ? 
                QuantizedAaddUniform<ActivationFunctionTypeIdentity, uint8_t> :
                QuantizedAaddUniform<ActivationFunctionTypeIdentity, float>;
            break;
        case ActivationFunctionTypeRelu:
            _uniform = _dstType == TensorType8u ?
                QuantizedAaddUniform<ActivationFunctionTypeRelu, uint8_t> :
                QuantizedAaddUniform<ActivationFunctionTypeRelu, float>;
            break;
        default:
            SYNET_ERROR("QuantizedAddLayer does not support " << Cpl::ToStr(_activationType) << " !");
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
        if(_uniform)
            _uniform(src[0]->Data<uint8_t>(), _aBias, _aNorm, src[1]->Data<uint8_t>(), _bBias, _bNorm, _size, _params, _scale, _zero, dst[0]->RawData());
    }
}