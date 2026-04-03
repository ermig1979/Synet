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
#include "Synet/Utils/Scale.h"

namespace Synet
{
    class SqueezeExcitationLayer : public Layer
    {
    public:
        SqueezeExcitationLayer(const LayerParam& param, Context* context, QuantizationMethod method);

        virtual void CompactWeight();

        virtual size_t MemoryUsage() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual int64_t Flop() const;

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

        void Forward32f(const float* src, float* sum, float* norm0, float* norm1, float* dst);

        void Forward8i(const uint8_t* src, int32_t* sum, float* norm0, float* norm1, uint8_t* dst8u, float* dst32f);

        void Init8i();

        void Scale8i(const uint8_t* src, float* norm, uint8_t* dst);

        void Scale8i(const uint8_t* src, float* norm, float* dst);

        void Forward16b(const uint16_t* src, float* sum, float* norm0, float* norm1, uint16_t* dst16u, float* dst32f);

        void Scale16b(const uint16_t* src, float* norm, uint16_t* dst);

        void Scale16b(const uint16_t* src, float* norm, float* dst);

        void Normalize(float* norm0, float* norm1);

    private:
        bool _src8u, _dst8u, _src16b, _dst16b, _hasBias[2];
        TensorFormat _format;
        size_t _batch, _channels, _height, _width, _size, _squeeze, _sci; 
        ActivationFunctionType _actType;
        float _kAvg;
        QuantizationMethod _method;
        Floats _sumScale, _sumShift, _rWeight[2], _params;
        Synet::Scale8i _scale8i;
        Synet::Scale16b _scale16b;
    };
}