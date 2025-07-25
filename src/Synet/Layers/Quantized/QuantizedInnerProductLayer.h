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
#include "Synet/Utils/InnerProduct.h"

namespace Synet
{
    class QuantizedInnerProductLayer : public Layer
    {
    public:
        QuantizedInnerProductLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual bool Resizable() const;

        virtual size_t MemoryUsage() const;

        virtual void CompactWeight();

        virtual int64_t Flop() const;

        virtual LowPrecisionType LowPrecision(TensorType type) const;

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        bool Compartible() const;

        bool InitParams();

        void Gemm(const uint8_t* src, int32_t* sum);
        void PostProcess(const int32_t* sum, float* dst);
        void PostProcess(const int32_t* sum, uint8_t* dst);

        bool _src8u, _dst8u;
        size_t _axis, _batch, _M, _N, _K;
        bool _biasTerm, _transA, _transB;
        Tensor _dstZero8u, _bias32i, _norm32f;
        QuantizedInnerProduct _quantizedInnerProduct;
    };
}