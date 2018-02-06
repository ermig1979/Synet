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
    template <class T, template<class> class A> class InnerProductLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        InnerProductLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst)
        {
            _biasTerm = this->Param().innerProduct().biasTerm();
            _transpose = this->Param().innerProduct().transpose();
            _axis = this->Param().innerProduct().axis();
            _N = this->Param().innerProduct().outputNum();
            _K = src[0]->Axis(_axis);

            const typename Base::Tensors & weight = this->Weight();
            if (_biasTerm)
                assert(weight.size() == 2);
            else
                assert(weight.size() == 1);
            if (_transpose)
                assert(weight[0].Shape() == Shape({ _K, _N }));
            else
                assert(weight[0].Shape() == Shape({ _N, _K }));
            if (_biasTerm)
                assert(weight[1].Shape() == Shape({ _N }));
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            const size_t newK = src[0]->Axis(_axis);
            _M = src[0]->Size(0, _axis);
            Shape dstShape = src[0]->Shape();
            dstShape.resize(_axis + 1);
            dstShape[_axis] = _N;
            dst[0]->Reshape(dstShape);
            if (_biasTerm)
            {
                Shape biasShape(1, _M);
                _biasMultiplier.Reshape(biasShape, Type(1.0f));
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            CpuGemm<Type>(CblasNoTrans, _transpose ? CblasNoTrans : CblasTrans, _M, _N, _K,
                (Type)1.0, src[0]->Data(), this->Weight()[0].Data(), (Type)0.0, dst[0]->Data());
            if (_biasTerm)
            {
                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _M, _N, 1,
                    (Type)1.0, _biasMultiplier.Data(), this->Weight()[1].Data(), (Type)1.0, dst[0]->Data());
            }
        }

    private:
        size_t _M, _K, _N, _axis;
        bool _biasTerm, _transpose;
        Synet::Tensor<T, A> _biasMultiplier;
    };
}