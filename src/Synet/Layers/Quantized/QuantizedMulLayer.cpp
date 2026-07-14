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

#include "Synet/Layers/Quantized/QuantizedMulLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

#include "Synet/Utils/Activation.h"
#include "Synet/Utils/UniversalBinary.h"

namespace Synet
{
    SYNET_INLINE void QuantizedMul(int a, int aBias, float aScale, int b, int bBias, float bScale, float scale, int dZero, uint8_t& dst)
    {
        float _a = DequantizeLinear(a, aBias, aScale);
        float _b = DequantizeLinear(b, bBias, bScale);
        dst = (uint8_t)QuantizeLinear(_a * _b, scale, dZero, 0, 255);
    }

    template<class D> void QuantizedMulUniformV0(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, float dScale, int dZero, uint8_t* dst8)
    {
        int aBias = -aZero, bBias = -bZero;
        if (std::is_same<D, uint8_t>())
        {
            float scale = 1.0f / dScale;
            for (size_t i = 0; i < size; ++i)
                QuantizedMul(a[i], aBias, aScale, b[i], bBias, bScale, scale, dZero, dst8[i]);
        }
        if (std::is_same<D, float>())
        {
            D* dst = (D*)dst8;
            for (size_t i = 0; i < size; ++i)
            {
                float _a = DequantizeLinear(a[i], aBias, aScale);
                float _b = DequantizeLinear(b[i], bBias, bScale);
                dst[i] = (D)(_a * _b);
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <size_t N> void QuantizedMulUniversal8u(const uint8_t* a, const Shape& aSteps, float aScale, int aZero,
        const uint8_t* b, const Shape& bSteps, float bScale, int bZero, uint8_t* dst, const Shape& dstShape, float dScale, int dZero)
    {
        int aBias = -aZero, bBias = -bZero;
        float scale = 1.0f / dScale;
        if (N == 1)
        {
            const uint8_t* a0 = a;
            const uint8_t* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                QuantizedMul(*a0, aBias, aScale, *b0, bBias, bScale, scale, dZero, *dst);
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
                    QuantizedMul(*a1, aBias, aScale, *b1, bBias, bScale, scale, dZero, *dst);
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
                        QuantizedMul(*a2, aBias, aScale, *b2, bBias, bScale, scale, dZero, *dst);
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
                            QuantizedMul(*a3, aBias, aScale, *b3, bBias, bScale, scale, dZero, *dst);
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

    QuantizedMulLayer::UniversalPtr GetQuantizedMulUniversal(TensorType typeD, size_t dim)
    {
        if (typeD == TensorType32f)
            return NULL;
        else
        {
            switch (dim)
            {
            case 1: return QuantizedMulUniversal8u<1>;
            case 2: return QuantizedMulUniversal8u<2>;
            case 3: return QuantizedMulUniversal8u<3>;
            case 4: return QuantizedMulUniversal8u<4>;
            default:
                return NULL;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    QuantizedMulLayer::QuantizedMulLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _uniform(NULL)
        , _universal(NULL)
    {
    }

    int64_t QuantizedMulLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * (_dType == TensorType8u ? 7 : 5);
    }

    LowPrecisionType QuantizedMulLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedMulLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("QuantizedMulLayer supports only 2 inputs and 1 output!");

        const LayerParam& param = this->Param();
        if (param.qSrc().size() != 2)
            SYNET_ERROR("QuantizedMulLayer must have 2 inputs quantization parameters!");

        _aType = src[0]->GetType(), _bType = src[1]->GetType();
        if (_aType != TensorType8u || _bType != TensorType8u)
            SYNET_ERROR("QuantizedMulLayer supports only INT8 inputs!");

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
            SYNET_ERROR("QuantizedMulLayer supports only INT8 or FP32 output!");

        const ActivationParam& act = param.activation();
        _size = src[0]->Size();

        Shape shapeA = src[0]->Shape(), shapeB = src[1]->Shape();
        if (!IsCompatible(shapeA, shapeB))
            SYNET_ERROR("QuantizedMulLayer incompatible input shapes!");
        Shape shapeD = OutputShape(shapeA, shapeB);

        shapeB = FullSrcShape(shapeB, shapeD);
        shapeA = FullSrcShape(shapeA, shapeD);

        _quantizedMul.Init(shapeA, _aType, _aScale, _aZero, shapeB, _bType, _bScale, _bZero, _dType, _dScale, _dZero);
        if (!_quantizedMul.Enable())
        {
            _uniform = NULL, _universal = NULL;
            if (shapeA == shapeB)
            {
                _uniform = _dType == TensorType8u ? QuantizedMulUniformV0<uint8_t> : QuantizedMulUniformV0<float>;
            }
            else
            {
                _dShape = shapeD;
                CompactShapes(shapeA, shapeB, _dShape);
                _aSteps = SourceSteps(shapeA, _dShape);
                _bSteps = SourceSteps(shapeB, _dShape);
                _universal = GetQuantizedMulUniversal(_dType, shapeA.size());
                if (_universal == NULL)
                    SYNET_ERROR("QuantizedMulLayer can create universal worker!");
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
            Forward(src, buf, dst, 0);
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

    void QuantizedMulLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if (_quantizedMul.Enable())
            _quantizedMul.Forward(src[0]->RawData(), src[1]->RawData(), dst[0]->RawData());
        else if(_uniform)
            _uniform(src[0]->Data<uint8_t>(), _aScale, _aZero, src[1]->Data<uint8_t>(), _bScale, _bZero, _size, _dScale, _dZero, dst[0]->RawData());
        else if (_universal)
            _universal(src[0]->Data<uint8_t>(), _aSteps, _aScale, _aZero, src[1]->Data<uint8_t>(), _bSteps, _bScale, _bZero, dst[0]->RawData(), _dShape, _dScale, _dZero);
    }
}