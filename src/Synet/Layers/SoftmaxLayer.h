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
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void SoftmaxLayerForwardCpu(const T * src, size_t channels, size_t inner, T * buffer, T * dst)
        {
            Synet::CpuCopy(src, inner, buffer);
            const T * s = src + inner;
            for (size_t i = 1; i < channels; ++i)
            {
                Synet::CpuMax(s, buffer, inner, buffer);
                s += inner;
            }

            s = src;
            T * d = dst;
            for (size_t i = 0; i < channels; ++i)
            {
                Synet::CpuSub(s, buffer, inner, d);
                s += inner;
                d += inner;
            }

            CpuExp(dst, channels*inner, dst);

            Synet::CpuCopy(dst, inner, buffer);
            d = dst + inner;
            for (size_t i = 1; i < channels; ++i)
            {
                Synet::CpuAdd(d, buffer, inner, buffer);
                d += inner;
            }

            d = dst;
            for (size_t i = 0; i < channels; ++i)
            {
                Synet::CpuDiv(d, buffer, inner, d);
                d += inner;
            }
        }
    }

    template <class T> class SoftmaxLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        SoftmaxLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
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
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            size_t channels = src[0]->Axis(_softmaxAxis);
            size_t dim = src[0]->Size() / _outerNum;
            for (size_t i = 0; i < _outerNum; ++i)
                Detail::SoftmaxLayerForwardCpu(src[0]->CpuData() + i*dim, channels, _innerNum, _scale.CpuData(), dst[0]->CpuData() + i*dim);
        }

    private:
        size_t _outerNum, _innerNum, _softmaxAxis;
        Tensor _scale;
    };
}