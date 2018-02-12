/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> class SoftmaxLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        SoftmaxLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            _softmaxAxis = this->Param().softmax().axis();
            dst[0]->Reshape(src[0]->Shape());
            _outerNum = src[0]->Size(0, _softmaxAxis);
            _innerNum = src[0]->Size(_softmaxAxis + 1);
            Shape scaleShape = src[0]->Shape();
            scaleShape[_softmaxAxis] = 1;
            _scale.Reshape(scaleShape);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            size_t channels = src[0]->Axis(_softmaxAxis);
            size_t dim = src[0]->Size() / _outerNum;
            for (size_t i = 0; i < _outerNum; ++i)
                CpuSoftmax(src[0]->Data() + i*dim, channels, _innerNum, _scale.Data(), dst[0]->Data() + i*dim);
        }

    private:
        size_t _outerNum, _innerNum, _softmaxAxis;
        Tensor _scale;
    };
}