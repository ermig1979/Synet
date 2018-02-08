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
    template <class T, template<class> class A> class EltwiseLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        EltwiseLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            const EltwiseParam & param = this->Param().eltwise();
            assert(param.coefficients().size() == 0 || param.coeffecients().size() == src.size());
            assert(!(param.operation() == EltwiseOperationTypeProduct && param.coefficients().size()));
            _operation = param.operation();
            _coefficients.resize(src.size(), Type(1));
            if (param.coefficients().size())
            {
                for (size_t i = 0; i < src.size(); ++i)
                    _coefficients[i] = param.coefficients()[i];
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            for (size_t i = 1; i < src.size(); ++i) 
                assert(src[i]->Shape() == src[0]->Shape());
            dst[0]->Reshape(src[0]->Shape());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            int * mask = NULL;
            const Type * bottom_data_a = NULL;
            const Type * bottom_data_b = NULL;
            size_t size = dst[0]->Size();
            Type * pDst = dst[0]->Data();
            switch (_operation) 
            {
            case EltwiseOperationTypeProduct:
                CpuMul(src[0]->Data(), src[1]->Data(), size, pDst);
                for (size_t i = 2; i < src.size(); ++i) 
                    CpuMul(pDst, src[i]->Data(), size, pDst);
                break;
            case EltwiseOperationTypeSum:
                CpuSet(size, Type(0), pDst);
                for (size_t i = 0; i < src.size(); ++i) 
                    CpuAxpy(src[i]->Data(), size, _coefficients[i], pDst);
                break;
            case EltwiseOperationTypeMax:
                CpuMax(src[0]->Data(), src[1]->Data(), size, pDst);
                for (size_t j = 2; j < src.size(); ++j)
                    CpuMax(pDst, src[j]->Data(), size, pDst);
                break;
            default:
                assert(0);
            }
        }

    private:
        typedef std::vector<Type> Vector;

        EltwiseOperationType _operation;
        Vector _coefficients;
    };
}