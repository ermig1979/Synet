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
#include "Synet/BiasLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> class ScaleLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        ScaleLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            const ScaleParam & param = this->Param().scale();
            _axis = param.axis();
            if (param.biasTerm()) 
            {
                LayerParam layerParam(this->Param());
                layerParam.type() = LayerTypeBias;
                layerParam.bias().axis() = param.axis();
                if(src.size() > 1)
                    layerParam.bias().numAxes() = (uint32_t)src[1]->Count();
                else
                    layerParam.bias().numAxes() = param.numAxes();
                _biasLayer.reset(new BiasLayer<T, A>(layerParam));
                Tensors & weight = (Tensors &)_biasLayer->Weight();
                weight.resize(1);
                weight[0].Share(this->Weight().back());
                _biasSrc.resize(1);
                _biasSrc[0] = src[0];
                _biasLayer->Setup(_biasSrc, dst);
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            const ScaleParam & param = this->Param().scale();
            Tensor & scale = (src.size() > 1) ? *src[1] : (Tensor &)this->Weight()[0];
            _axis = (scale.Count() == 0) ? 0 : param.axis();
            assert(src[0]->Count() <= _axis + scale.Count());
            for (size_t i = 0; i < scale.Count(); ++i)
                assert(src[0]->Axis(_axis + i) == scale.Axis(i));
            _outerDim = src[0]->Size(0, _axis);
            _scaleDim = scale.Size();
            _innerDim = src[0]->Size(_axis + scale.Count());
            if (src[0] == dst[0])
                _temp.Reshape(src[0]->Shape());
            else
                dst[0]->Reshape(src[0]->Shape());
            if (_biasLayer) 
            {
                _biasSrc[0] = dst[0];
                _biasLayer->Reshape(_biasSrc, dst);
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type* pSrc = src[0]->Data();
            if (src[0] == dst[0]) 
                CpuCopy(pSrc, src[0]->Size(), _temp.Data());
            const Type * pScale = ((src.size() > 1) ? *src[1] : this->Weight()[0]).Data();
            Type * pDst = dst[0]->Data();
            for (size_t n = 0; n < _outerDim; ++n)
            {
                for (size_t d = 0; d < _scaleDim; ++d)
                {
                    CpuScale(pSrc, _innerDim, pScale[d], pDst);
                    pSrc += _innerDim;
                    pDst += _innerDim;
                }
            }
            if (_biasLayer)
                _biasLayer->Forward(_biasSrc, dst);
        }

    private:
        std::shared_ptr<Layer<T, A>> _biasLayer;
        TensorPtrs _biasSrc;
        Tensor _temp;
        size_t _axis, _outerDim, _scaleDim, _innerDim;
    };
}