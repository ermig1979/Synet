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

#pragma once

#include "Synet/Layer.h"

#include "Synet/Utils/Add.h"

namespace Synet
{
    class QuantizedAddLayer : public Layer
    {
    public:
        QuantizedAddLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual int64_t Flop() const;

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        typedef void (*UniformPtr)(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst);
        typedef void (*UniversalPtr)(
            const uint8_t* a, const Shape& aSteps, float aScale, int aZero, 
            const uint8_t* b, const Shape& bSteps, float bScale, int bZero, 
            const float* params, uint8_t* dst, const Shape& dstShape, float dScale, int dZero);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

        TensorType _aType, _bType, _dType;
        ActivationFunctionType _activationType;
        int32_t _aZero, _bZero, _dZero;
        float _aScale, _bScale, _dScale, _params[2];
        size_t _axis, _size;
        Shape _aSteps, _bSteps, _dShape;

        UniformPtr _uniform;
        UniversalPtr _universal;
        QuantizedAdd _quantizedAdd;
    };
}