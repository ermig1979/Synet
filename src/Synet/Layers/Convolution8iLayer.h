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

#include "Synet/Layers/ConvolutionLayer.h"
#include "Synet/Utils/Convolution.h"
#include "Synet/Quantization/Convert.h"

namespace Synet
{
    class Convolution8iLayer : public Synet::ConvolutionLayer
    {
    public:
        Convolution8iLayer(const LayerParam& param, Context* context, QuantizationMethod method);

        virtual size_t MemoryUsage() const;

        virtual bool Can8i() const;

        virtual bool Is8i() const;

        virtual void DebugPrint(std::ostream& os, int flag, int first, int last, int precision);

    protected:
        typedef typename ConvolutionLayer::AlgParam AlgParam;

        virtual String InternalInfo() const;

        virtual bool Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst);

        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        void SetNormMinMax(const float* weight, const float* scale, float* norm, float& min, float& max);

        void Quantize();

        void ForwardCpu(const uint8_t* src, uint8_t* buf, int32_t* sum, float* dst);

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u;
        Converter _srcCvt, _dstCvt;
        Tensor _weight8i, _norm32f, _bias32f;

        Convolution8i _convolution8i;
    };
}