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
    template <class T, template<class> class A> class BiasLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        BiasLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            if (src.size() == 1) 
            {
                const BiasParam & param = this->Param().bias();
                assert(src[0]->Count() >= param.axis() + param.numAxes());
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            const BiasParam & param = this->Param().bias();
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
            _biasMultiplier.Reshape({ _innerDim }, Type(1));
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pBias = ((src.size() > 1) ? *src[1] : this->Weight()[0]).Data();
            Type * pDst = dst[0]->Data();
            if (src[0] != dst[0])
                CpuCopy(src[0]->Data(), src[0]->Size(), pDst);
            for (size_t n = 0; n < _outerDim; ++n)
            {
                CpuGemm(CblasNoTrans, CblasNoTrans, _biasDim, _innerDim, 1, Type(1), pBias, _biasMultiplier.Data(), Type(1), pDst);
                pDst += _dim;
            }
        }

    private:
        Tensor _biasMultiplier;
        size_t _outerDim, _biasDim, _innerDim, _dim;
    };
}