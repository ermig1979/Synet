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

#include "Synet/Layers/DeconvolutionLayer.h"

namespace Synet
{
    class Deconvolution32fLayer : public DeconvolutionLayer
    {
    public:
        Deconvolution32fLayer(const LayerParam& param, Context* context);

        virtual size_t MemoryUsage() const;

    protected:
        virtual bool Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst);

        virtual String InternalInfo() const;

        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        void ForwardCpu(const float* src, float* buf, float* dst);

    private:
        bool _transW;
        size_t _ldW, _ldS, _ldD, _grW, _grS, _grD, _siW, _siS, _siD;

        Deconvolution32f _deconvolution32f;
        Tensor _weightT;
    };
}