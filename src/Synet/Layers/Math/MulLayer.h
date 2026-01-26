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

#pragma once

#include "Synet/Layer.h"

namespace Synet
{
    class MulLayer : public Layer
    {
    public:
        MulLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        typedef void (*UniformPtr)(const uint8_t* a, const uint8_t* b, size_t size, uint8_t* dst);
        typedef void (*ScalePtr)(const uint8_t* a, const uint8_t* b, size_t count, size_t size, uint8_t* dst, TensorFormat format);
        typedef void (*UniversalPtr)(const uint8_t* a, const Shape& aSteps, const uint8_t* b, const Shape& bSteps, uint8_t* dst, const Shape& dstShape);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        TensorPtrs GetSrc(const TensorPtrs& src);
    private:
        enum Special
        {
            SpecialNone = 0,
            SpecialScaleChannel,
            SpecialScaleSpatial,
            SpecialScaleComplex,
            SpecialBatch,
            SpecialUniversal,
        } _special;
        TensorFormat _format;
        TensorType _type;
        size_t _batch, _channels, _spatial, _sizeT;
        size_t _channelsInner, _channelsOuter;
        Shape _aSteps, _bSteps, _dstShape;
        int _index[2];
        UniformPtr _uniform;
        ScalePtr _scale;
        UniversalPtr _universal;
    };
}