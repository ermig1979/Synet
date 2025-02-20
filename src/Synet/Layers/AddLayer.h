/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar,
*               2019-2019 Artur Voronkov.
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
    class AddLayer : public Layer
    {
    public:
        AddLayer(const LayerParam& param, Context* context, QuantizationMethod method);

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        typedef void (*UniformPtr)(const uint8_t* a, const uint8_t* b, size_t size, uint8_t* dst);
        typedef void (*AddBiasPtr)(const uint8_t* a, const uint8_t* b, size_t count, size_t size, uint8_t* dst, TensorFormat format);
        typedef void (*UniversalPtr)(const uint8_t* a, const Shape& aSteps, const uint8_t* b, const Shape& bSteps, uint8_t* dst, const Shape& dstShape);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);
        
        void Add8i(const uint8_t* src0, const uint8_t* src1, uint8_t* dst);

        void Add8i(const uint8_t* src0, const uint8_t* src1, float* dst);

        void Add8i(const float* src0, const float* src1, uint8_t* dst);

    private:
       enum Special
        {
            SpecialNone = 0,
            SpecialBiasChannel,
            SpecialBatch,
            SpecialBiasChannelV2,
            SpecialUniversal,
        } _special;
        TensorPtrs _src;
        bool _quant;
        QuantizationMethod _method;
        TensorFormat _format;
        TensorType _typeA, _typeB, _typeD;
        size_t _batch, _channels, _spatial, _elemA, _elemB, _elemD;
        size_t _channelsInner, _channelsOuter;
        Shape _aSteps, _bSteps, _dstShape;
        UniformPtr _uniform;
        AddBiasPtr _addBias;
        UniversalPtr _universal;
        Add16b _add16b;
    };
}