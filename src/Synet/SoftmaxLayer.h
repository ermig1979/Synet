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
            _sumMultiplier.Reshape({src[0]->Axis(_softmaxAxis)}, Type(1));
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

            const Type * pSrc = src[0]->Data();
            Type * pDst = dst[0]->Data();
            Type * pScale = _scale.Data();
            size_t channels = src[0]->Axis(_softmaxAxis);
            size_t dim = src[0]->Size() / _outerNum;
            CpuCopy(pSrc, src[0]->Size(), pDst);
            for (size_t i = 0; i < _outerNum; ++i)
            {
                CpuCopy(pSrc + i * dim, _innerNum, pScale);
                for (size_t j = 0; j < channels; j++) 
                {
                    for (size_t k = 0; k < _innerNum; k++)
                        pScale[k] = std::max(pScale[k], pSrc[i * dim + j * _innerNum + k]);
                }
                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, channels, _innerNum, 1, Type(-1), _sumMultiplier.Data(), pScale, Type(1), pDst);
                CpuExp(pDst, dim, pDst);
                CpuGemv<Type>(CblasTrans, channels, _innerNum, Type(1), pDst, _sumMultiplier.Data(), Type(0), pScale);
                for (size_t j = 0; j < channels; j++)
                {
                    CpuDiv(pDst, pScale, _innerNum, pDst);
                    pDst += _innerNum;
                }
            }
        }

    private:
        size_t _outerNum, _innerNum, _softmaxAxis;
        Tensor _scale, _sumMultiplier;
    };
}