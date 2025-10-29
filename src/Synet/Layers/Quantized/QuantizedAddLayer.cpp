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
#include "Synet/Utils/UniversalBinary.h"

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

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE void QuantizedAdd(int a, float aNorm, int b, float bNorm, float term, uint8_t &dst)
    {
        float val = Fmadd(float(a), aNorm, Fmadd(float(b), bNorm, term));
        dst = (uint8_t)RestrictRange(NearByInt(val), 0, 255);
    }

    template<ActivationFunctionType activation, class D> void QuantizedAddUniformV1(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst8)
    {
        D* dst = (D*)dst8;
        if (std::is_same<D, uint8_t>())
        {
            float aNorm = aScale / dScale;
            float bNorm = bScale / dScale;
            float term = float(dZero) - (aNorm * float(aZero) + bNorm * float(bZero));
            for (size_t i = 0; i < size; ++i)
                QuantizedAdd(a[i], aNorm, b[i], bNorm, term, dst[i]);
        }
        if (std::is_same<D, float>())
        {
            float term = - aScale * float(aZero) - bScale * float(bZero);
            for (size_t i = 0; i < size; ++i)
                dst[i] = (D)Activation<activation>(float(a[i]) * aScale + float(b[i]) * bScale + term, 0, params);
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <size_t N> void QuantizedAddUniversal8u(const uint8_t* a, const Shape& aSteps, float aScale, int aZero,
        const uint8_t* b, const Shape& bSteps, float bScale, int bZero,
        const float* params, uint8_t* dst, const Shape& dstShape, float dScale, int dZero)
    {
        float aNorm = aScale / dScale;
        float bNorm = bScale / dScale;
        float term = float(dZero) - (aNorm * float(aZero) + bNorm * float(bZero));
        if (N == 1)
        {
            const uint8_t* a0 = a;
            const uint8_t* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                QuantizedAdd(*a0, aNorm, *b0, bNorm, term, *dst);
                a0 += aSteps[0];
                b0 += bSteps[0];
                dst += 1;
            }
        }
        else if (N == 2)
        {
            const uint8_t* a0 = a;
            const uint8_t* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const uint8_t* a1 = a0;
                const uint8_t* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    QuantizedAdd(*a1, aNorm, *b1, bNorm, term, *dst);
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
            const uint8_t* a0 = a;
            const uint8_t* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const uint8_t* a1 = a0;
                const uint8_t* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const uint8_t* a2 = a1;
                    const uint8_t* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        QuantizedAdd(*a2, aNorm, *b2, bNorm, term, *dst);
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
            const uint8_t* a0 = a;
            const uint8_t* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const uint8_t* a1 = a0;
                const uint8_t* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const uint8_t* a2 = a1;
                    const uint8_t* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        const uint8_t* a3 = a2;
                        const uint8_t* b3 = b2;
                        for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                        {
                            QuantizedAdd(*a3, aNorm, *b3, bNorm, term, *dst);
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

    QuantizedAddLayer::UniversalPtr GetQuantizedAddUniversal(TensorType typeD, size_t dim)
    {
        if (typeD == TensorType32f)
            return NULL;
        else
        {
            switch (dim)
            {
            case 1: return QuantizedAddUniversal8u<1>;
            case 2: return QuantizedAddUniversal8u<2>;
            case 3: return QuantizedAddUniversal8u<3>;
            case 4: return QuantizedAddUniversal8u<4>;
            default:
                return NULL;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedAddLayer::QuantizedAddLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _uniform(NULL)
        , _universal(NULL)
    {
    }

    int64_t QuantizedAddLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * (_dType == TensorType8u ? 7 : 5);
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

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 2)
            SYNET_ERROR("QuantizedAddLayer must have 2 inputs quantization parameters!");

        _aType = src[0]->GetType(), _bType = src[1]->GetType();
        if (_aType != TensorType8u || _bType != TensorType8u)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 inputs!");

        _aScale = float(param.qSrc()[0].scale());
        _aZero = param.qSrc()[0].zero();
        _bScale = float(param.qSrc()[1].scale());
        _bZero = param.qSrc()[1].zero();

        if (param.qDst().size())
        {
            _dType = param.qDst()[0].type();
            _dZero = param.qDst()[0].zero();
            _dScale = float(param.qDst()[0].scale());
        }
        else
        {
            _dType = TensorType32f;
            _dZero = 0;
            _dScale = 1.0f;
        }
        if(_dType != TensorType8u && _dType != TensorType32f)
            SYNET_ERROR("QuantizedAddLayer supports only INT8 or FP32 output!");

        const ActivationParam& act = param.activation();
        _size = src[0]->Size();
        _activationType = act.type();
        _params[0] = act.param0();
        _params[1] = act.param1();

        Shape shapeA = src[0]->Shape(), shapeB = src[1]->Shape();
        if (!IsCompatible(shapeA, shapeB))
            SYNET_ERROR("QuantizedAddLayer incompatible input shapes!");
        Shape shapeD = OutputShape(shapeA, shapeB);

        shapeB = FullSrcShape(shapeB, shapeD);
        shapeA = FullSrcShape(shapeA, shapeD);
        _quantizedAdd.Init(shapeA, _aType, _aScale, _aZero, shapeB, _bType, _bScale, _bZero,
            _activationType, _params, _dType, _dScale, _dZero);
        if (!_quantizedAdd.Enable())
        {
            _uniform = NULL, _universal = NULL;
            if (shapeA == shapeB)
            {
                switch (_activationType)
                {
                case ActivationFunctionTypeIdentity:
                    _uniform = _dType == TensorType8u ?
                        QuantizedAddUniformV1<ActivationFunctionTypeIdentity, uint8_t> :
                        QuantizedAddUniformV0<ActivationFunctionTypeIdentity, float>;
                    break;
                case ActivationFunctionTypeRelu:
                    _uniform = _dType == TensorType8u ?
                        QuantizedAddUniformV1<ActivationFunctionTypeIdentity, uint8_t> :
                        QuantizedAddUniformV0<ActivationFunctionTypeRelu, float>;
                    break;
                default:
                    SYNET_ERROR("QuantizedAddLayer does not support " << Cpl::ToStr(_activationType) << " !");
                }
            }
            else
            {
                _dShape = shapeD;
                CompactShapes(shapeA, shapeB, _dShape);
                _aSteps = SourceSteps(shapeA, _dShape);
                _bSteps = SourceSteps(shapeB, _dShape);
                _universal = GetQuantizedAddUniversal(_dType, shapeA.size());
                if (_universal == NULL)
                    SYNET_ERROR("QuantizedAddLayer can create universal worker!");
            }
        }

        if (TensorUsers(Param().src()[0]) == 1 && !src[0]->Const() && shapeD == src[0]->Shape() && dst[0] != src[0])
            dst[0]->Share(*src[0]);
        else if (TensorUsers(Param().src()[1]) == 1 && !src[1]->Const() && shapeD == src[1]->Shape() && dst[0] != src[1])
            dst[0]->Share(*src[1]);
        else
            dst[0]->Reshape(_dType, shapeD, src[0]->Format());

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
        else if (_universal)
            _universal(src[0]->Data<uint8_t>(), _aSteps, _aScale, _aZero, src[1]->Data<uint8_t>(), _bSteps, _bScale, _bZero, _params, dst[0]->RawData(), _dShape, _dScale, _dZero);
    }
}