/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
    class MetaLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        MetaLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        bool ReshapeAdd(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapeCast(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst);
        bool ReshapeConst(const TensorParam& alpha, const TensorPtrs& dst);
        bool ReshapeDiv(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapeEqual(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapeExpandDims(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst);
        bool ReshapeFloor(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapeGather(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapeMul(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapePack(const TensorPtrs& src, const TensorPtrs& dst);
        bool ReshapePermute(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst);
        void ReshapeRange(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeReduceMin(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeReduceProd(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeReshape(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeSelect(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeShape(const TensorPtrs& src, int version, const TensorPtrs& dst);
        bool ReshapeSlice(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeSqueeze(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeStridedSlice(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeStub(const TensorPtrs& src, const TensorPtrs& dst);
        void ReshapeSub(const TensorPtrs& src, const TensorPtrs& dst);
    };
}