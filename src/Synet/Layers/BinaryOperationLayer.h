/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
    class BinaryOperationLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        BinaryOperationLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        typedef void(*Func32fPtr)(const float * a, const float* b, size_t outer, size_t aSize, size_t bSize, size_t inner, float* dst);
        typedef void(*FuncBoolPtr)(const bool* a, const bool* b, size_t outer, size_t aSize, size_t bSize, size_t inner, bool* dst);

        TensorType _srcType;
        BinaryOperationType _opType;
        size_t _outer, _aSize, _bSize, _inner;
        Func32fPtr _func32f;
        FuncBoolPtr _funcBool;
    };
}