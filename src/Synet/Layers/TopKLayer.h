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

namespace Synet
{
    class TopKLayer : public Layer
    {
    public:
        TopKLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        typedef void (*TopKPtr)(const uint8_t* src8, size_t outer, size_t count, size_t inner, size_t k, uint8_t* buf8, uint8_t* dst8, uint8_t* idx8);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _axis, _outer, _count, _inner, _k;
        TensorType _srcType, _idxType;
        TopKMode _mode;
        TopKSort _sort;
        TopKPtr _topK;
    };
}