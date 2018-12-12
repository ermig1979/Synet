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
#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T> class BiasLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        BiasLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const BiasParam & param = this->Param().bias();
            if (src.size() == 1)
                assert(src[0]->Count() >= param.axis() + param.numAxes());
            Tensor & bias = (src.size() > 1) ? *src[1] : (Tensor &)this->Weight()[0];
            size_t axis = bias.Count() == 0 ? 0 : param.axis();
            assert(src[0]->Count() >= axis + bias.Count());
            for (size_t i = 0; i < bias.Count(); ++i)
                assert(src[0]->Axis(axis + i) == bias.Axis(i));
            _outerDim = src[0]->Size(0, axis);
            _biasDim = bias.Size();
            _innerDim = src[0]->Size(axis + bias.Count());
            _dim = _biasDim * _innerDim;
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pBias = ((src.size() > 1) ? *src[1] : this->Weight()[0]).CpuData();
            Type * pDst = dst[0]->CpuData();
            if (src[0] != dst[0])
                CpuCopy(src[0]->CpuData(), src[0]->Size(), pDst);
            for (size_t n = 0; n < _outerDim; ++n)
            {
                CpuAddBias(pBias, _biasDim, _innerDim, pDst);
                pDst += _dim;
            }
        }

    private:
        size_t _outerDim, _biasDim, _innerDim, _dim;
    };
}