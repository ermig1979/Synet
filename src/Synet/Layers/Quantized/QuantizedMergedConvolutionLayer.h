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
#include "Synet/Utils/Convolution.h"

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

        bool Compartible() const;

        bool InitParams();

    protected:
        static const size_t COUNT_MAX = 3;
        size_t _count, _batch;
        ConvParam _conv[COUNT_MAX];
        bool _bias[COUNT_MAX];
        //struct AlgParam
        //{
        //    int is1x1, bias, trans, internal;
        //    size_t batch, sSize, dSize, ldW, ldS, ldD, grW, grS, grD, siW, siS, siD;
        //    float params[2];
        //} _alg;
        //bool _src8u, _dst8u;
        //Tensor _srcZero8u, _dstZero8u, _bias32i, _norm32f;

        //struct AlgParam
        //{
        //    int internal[COUNT_MAX], add;
        //    size_t index[COUNT_MAX], sSize, dSize, batch, count;
        //    ConvParam conv[COUNT_MAX];
        //    float params[COUNT_MAX][2];
        //    const float* weight[COUNT_MAX], * bias[Detail::MCC_MAX], * params[Detail::MCC_MAX];

        //    bool IsCdc() const { return count == 3; }
        //    bool IsCd() const { return count == 2 && conv[0].group == 1; }
        //    bool IsDc() const { return conv[0].group != 1; }
        //} _alg;
    };
}