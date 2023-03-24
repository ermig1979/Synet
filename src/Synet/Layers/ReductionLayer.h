/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
        template <class T> void ReductionLayerForwardCpuMax(const T * src, size_t outer, size_t count, size_t inner, T * dst)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = -FLT_MAX;
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] = Max(dst[i], src[i]);
                    src += inner;
                }
                dst += inner;
            }
        }

        template <class T> void ReductionLayerForwardCpuMin(const T* src, size_t outer, size_t count, size_t inner, T* dst)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = FLT_MAX;
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] = Min(dst[i], src[i]);
                    src += inner;
                }
                dst += inner;
            }
        }

        template <class T> void ReductionLayerForwardCpuSum(const T * src, size_t outer, size_t count, size_t inner, T * dst)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = T(0);
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] += src[i];
                    src += inner;
                }
                dst += inner;
            }
        }

        template <class T> void ReductionLayerForwardCpuProd(const T* src, size_t outer, size_t count, size_t inner, T* dst)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = T(1);
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] *= src[i];
                    src += inner;
                }
                dst += inner;
            }
        }

        template <class T> void ReductionLayerForwardCpuL2(const T * src, size_t outer, size_t count, size_t inner, T * dst)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = T(0);
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] += Square(src[i]);
                    src += inner;
                }
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = T(::sqrt(dst[i]));
                dst += inner;
            }
        }

        template <class T> void ReductionLayerForwardCpuMean(const T* src, size_t outer, size_t count, size_t inner, T* dst)
        {
            T k = T(1) / T(count);
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = T(0);
                for (size_t c = 0; c < count; ++c)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] += src[i];
                    src += inner;
                }
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = dst[i] * k;
                dst += inner;
            }
        }
    }

    template <class T> class ReductionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReductionLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ReductionParam & param = this->Param().reduction();
            _type = param.type();
            switch (_type)
            {
            case ReductionTypeMax:
                _func = Detail::ReductionLayerForwardCpuMax;
                break;
            case ReductionTypeMin:
                _func = Detail::ReductionLayerForwardCpuMin;
                break;
            case ReductionTypeSum:
                _func = Detail::ReductionLayerForwardCpuSum;
                break;
            case ReductionTypeProd:
                _func = Detail::ReductionLayerForwardCpuProd;
                break;
            case ReductionTypeL2:
                _func = Detail::ReductionLayerForwardCpuL2;
                break;
            case ReductionTypeMean:
                _func = Detail::ReductionLayerForwardCpuMean;
                break;
            default:
                assert(0);
            }

            Shape shape;
            _outer = 1, _count = 1, _inner = 1;
            if (src[0]->Count() > 1)
            {
                std::set<size_t> axis;
                for (size_t i = 0; i < param.axis().size(); ++i)
                    axis.insert(src[0]->Index(param.axis()[i]));
                for (size_t i = 0; i < src[0]->Count(); ++i)
                {
                    size_t size = src[0]->Axis(i);
                    if (axis.find(i) != axis.end())
                    {
                        _count *= size;
                        if (param.keepDims())
                            shape.push_back(1);
                    }
                    else
                    {
                        (_count > 1 ? _inner : _outer) *= size;
                        shape.push_back(size);
                    }
                }
            }
            else
            {
                _count = src[0]->Size();
                shape.push_back(1);
            }
            assert(_outer*_count*_inner == src[0]->Size());
            dst[0]->Reshape(shape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _func(src[0]->CpuData(), _outer, _count, _inner, dst[0]->CpuData());
        }

    private:
        typedef void(*FuncPtr)(const Type * src, size_t outer, size_t count, size_t inner, Type * dst);

        ReductionType _type;
        FuncPtr _func;
        size_t _outer, _count, _inner;
    };
}