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
#include "Synet/Utils/ConvParam.h"
#include "Synet/Utils/MergedConvolution.h"

namespace Synet
{
    class QuantizedMergedConvolutionLayer : public Layer
    {
    public:
        QuantizedMergedConvolutionLayer(const LayerParam& param, Context* context);

        virtual size_t MemoryUsage() const;

        virtual int64_t Flop() const;

        virtual void CompactWeight();

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        void ForwardCpu(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t * dst);

        void DepthwiseConvolution(const uint8_t* src, const uint8_t* zero, const ConvParam& conv, const int8_t * weight, int32_t* dst);

        void Add(const uint8_t* a, const uint8_t* b, uint8_t* dst);

    protected:
        static const size_t COUNT_MAX = 3;
        size_t _count, _batch, _srcS, _dstS, _indexQ[COUNT_MAX], _indexW[COUNT_MAX];
        int32_t _add, *_ptrB[3];
        uint8_t _ioZero[5];
        ConvParam _conv[COUNT_MAX];
        bool _bias[COUNT_MAX];
        float _params[COUNT_MAX][2], _aNorm, _bNorm, _term, _ioScale[5], *_ptrS[3];
        const int8_t* _ptrW[COUNT_MAX];
        Tensor _bias32i[COUNT_MAX], _norm32f[COUNT_MAX], _dwSrcZero8u;

        QuantizedMergedConvolution _quantizedMergedConvolution;
    };
}