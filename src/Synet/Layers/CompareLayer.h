/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
        template <class T> void CompareLayerForward(const T * a, const T * b, size_t size, CompareType type, T * dst)
        {
            T neg = T(0), pos = Not(neg);
            switch (type)
            {
            case CompareTypeEqual:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] == b[i] ? pos : neg;
                break;
            case CompareTypeNotEqual:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] != b[i] ? pos : neg;
                break;
            case CompareTypeGreaterThan:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] > b[i] ? pos : neg;
                break;
            case CompareTypeGreaterOrEqual:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] >= b[i] ? pos : neg;
                break;
            case CompareTypeLessThan:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] < b[i] ? pos : neg;
                break;
            case CompareTypeLessOrEqual:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = a[i] <= b[i] ? pos : neg;
                break;
            default:
                assert(0);
            }
        }
    }

    template <class T> class CompareLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        CompareLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape());
            _compareType = this->Param().compare().type();
            _size = src[0]->Size();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::CompareLayerForward(src[0]->CpuData(), src[1]->CpuData(), _size, _compareType, dst[0]->CpuData());
        }

        CompareType _compareType;
        size_t _size;
    };
}