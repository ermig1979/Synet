/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

namespace Synet
{
    namespace Detail
    {
        template <BinaryOperationType type, class T> struct BinaryOperation;
        
        template <class T> struct BinaryOperation<BinaryOperationTypeDiv, T>
        {
            static T Run(T a, T b)
            {
                return a / b;
            }
        };

        template <class T> struct BinaryOperation<BinaryOperationTypeSub, T>
        {
            static T Run(T a, T b)
            {
                return a - b;
            }
        };

        template <BinaryOperationType type, class T> void BinaryOperationLayerForwardCpu(const T * a, const T * b, size_t outer, size_t aSize, size_t bSize, size_t inner, T * dst)
        {
            if (aSize == bSize)
            {
                size_t size = outer*aSize*inner;
                for (size_t i = 0; i < size; ++i)
                    dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
            }
            else if (aSize == 1)
            {
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t s = 0; s < bSize; ++s)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
                        b += inner;
                        dst += inner;
                    }
                    a += inner;
                }
            }
            else if (bSize == 1)
            {
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t s = 0; s < aSize; ++s)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
                        a += inner;
                        dst += inner;
                    }
                    b += inner;
                }
            }
            else
                assert(0);
        }
    }

    template <class T> class BinaryOperationLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        BinaryOperationLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _type = this->Param().binaryOperation().type();
            switch (_type)
            {
            case BinaryOperationTypeDiv:
                _func = Detail::BinaryOperationLayerForwardCpu<BinaryOperationTypeDiv, Type>;
                break;
            case BinaryOperationTypeSub:
                _func = Detail::BinaryOperationLayerForwardCpu<BinaryOperationTypeSub, Type>;
                break;
            default:
                assert(0);
            }            
            assert(src.size() == 2 && src[0]->Count() == src[1]->Count());
            Shape shape;
            _outer = 1, _aSize = 1, _bSize = 1, _inner = 1;
            for (size_t i = 0; i < src[0]->Count(); ++i)
            {
                if (src[0]->Axis(i) == src[1]->Axis(i))
                {
                    (_aSize*_bSize > 1 ? _inner : _outer) *= src[0]->Axis(i);
                    shape.push_back(src[0]->Axis(i));
                }
                else if (src[0]->Axis(i) == 1)
                {
                    _bSize *= src[1]->Axis(i);
                    shape.push_back(src[1]->Axis(i));
                }
                else if (src[1]->Axis(i) == 1)
                {
                    _aSize *= src[0]->Axis(i);
                    shape.push_back(src[0]->Axis(i));
                }
                else
                    assert(0);
            }
            dst[0]->Reshape(shape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _func(src[0]->CpuData(), src[1]->CpuData(), _outer, _aSize, _bSize, _inner, dst[0]->CpuData());
        }

    private:
        typedef void(*FuncPtr)(const Type * a, const Type * b, size_t outer, size_t aSize, size_t bSize, size_t inner, Type * dst);

        BinaryOperationType _type;
        size_t _outer, _aSize, _bSize, _inner;
        FuncPtr _func;
    };
}