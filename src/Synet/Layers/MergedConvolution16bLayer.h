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

#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/MergedConvolutionLayer.h"

namespace Synet
{
    class MergedConvolution16bLayer : public MergedConvolutionLayer
    {
    public:
        typedef Layer<float> Base;       
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolution16bLayer(const LayerParam& param, Context* context);

        virtual bool Can16b() const;

        virtual bool Is16b() const;

        virtual size_t MemoryUsage() const;

    protected:
        typedef MergedConvolutionLayer::AlgParam AlgParam;

        virtual String InternalInfo() const;

        virtual bool Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst);

        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        bool _src16b, _dst16b;
        MergedConvolution16b _mergedConvolution16b;
    };
}