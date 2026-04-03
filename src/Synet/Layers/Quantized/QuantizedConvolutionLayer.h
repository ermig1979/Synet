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
    class QuantizedConvolutionLayer : public Layer
    {
    public:
        QuantizedConvolutionLayer(const LayerParam& param, Context* context);

        virtual size_t MemoryUsage() const;

        virtual int64_t Flop() const;

        virtual void CompactWeight();

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

        bool CheckParams();

        bool InitParams();

        void Convolution(const uint8_t* src, uint8_t* buf, int32_t* sum);
        void PostProcess(const int32_t* sum, float* dst);
        void PostProcess(const int32_t* sum, float* buf, uint8_t* dst);

    protected:
        ConvParam _conv;
        struct AlgParam
        {
            int is1x1, bias, trans, internal;
            size_t batch, sSize, dSize, ldW, ldS, ldD, grW, grS, grD, siW, siS, siD;
            float params[2];
        } _alg;
        Tensor _srcZero8u, _bias32i, _norm32f;
        int32_t _srcZero, _intZero, _dstZero;
        float _srcScale, _intScale, _dstScale;
        size_t _biasStart, _actStart;

        QuantizedConvolution _quantizedConvolution;
    };
}