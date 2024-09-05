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

#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Quantization/Convert.h"
#include "Synet/Layers/MergedConvolutionLayer.h"

namespace Synet
{
    class MergedConvolution8iLayer : public MergedConvolutionLayer
    {
    public:
        MergedConvolution8iLayer(const LayerParam& param, Context* context, QuantizationMethod method);

        virtual bool Can8i() const;

        virtual bool Is8i() const;

        virtual size_t MemoryUsage() const;

        virtual void DebugPrint(std::ostream& os, int flag, int first, int last, int precision);

    protected:
        typedef MergedConvolutionLayer::AlgParam AlgParam;

        virtual String InternalInfo() const;

        virtual bool Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst);

        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        void DirectConvolution8i(const uint8_t* src, size_t cIdx, size_t wIdx, const uint8_t* zero, uint8_t* buf, int32_t* sum, float* dst);

        void Init();

        void Quantize(size_t srcIdx, const Stat& stat, size_t dstIdx);

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u, _dw0;
        Converter _srcCvt, _intCvt, _dstCvt;
        Tensor _weight8i[2], _norm32f[2], _bias32f[2];
        typedef void(*DepthwiseConvolution32fPtr)(const float* src, const ConvParam& conv, const float* weight, const float* bias, const float* params, float* dst);
        DepthwiseConvolution32fPtr _depthwise;

        MergedConvolution8i _mergedConvolution8i;
    };
}