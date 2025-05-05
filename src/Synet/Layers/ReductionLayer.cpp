/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Layers/ReductionLayer.h"

#include <limits>

namespace Synet
{
    template <class T> void ReduceMax(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T *)src8;
        T* dst = (T*)dst8;
        for (size_t o = 0; o < outer; ++o)
        {
            for (size_t i = 0; i < inner; ++i)
                dst[i] = -std::numeric_limits<T>::max();
            for (size_t c = 0; c < count; ++c)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = Max(dst[i], src[i]);
                src += inner;
            }
            dst += inner;
        }
    }

    template <class T> void ReduceMin(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        for (size_t o = 0; o < outer; ++o)
        {
            for (size_t i = 0; i < inner; ++i)
                dst[i] = std::numeric_limits<T>::max();
            for (size_t c = 0; c < count; ++c)
            {
                for (size_t i = 0; i < inner; ++i)
                    dst[i] = Min(dst[i], src[i]);
                src += inner;
            }
            dst += inner;
        }
    }

    template <class T> void ReduceSum(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
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

    template <class T> void ReduceProd(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
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

    template <class T> void ReduceL2(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
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
                dst[i] = T(::sqrt(float(dst[i])));
            dst += inner;
        }
    }

    template <class T> void ReduceMean(const uint8_t* src8, size_t outer, size_t count, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        float k = 1.0f / float(count);
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
                dst[i] = T(dst[i] * k);
            dst += inner;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> ReductionLayer::ReducePtr GetReduce(ReductionType type)
    {
        switch (type)
        {
        case ReductionTypeMax: return ReduceMax<T>;
        case ReductionTypeMin: return ReduceMin<T>;
        case ReductionTypeSum: return ReduceSum<T>;
        case ReductionTypeProd: return ReduceProd<T>;
        case ReductionTypeL2: return ReduceL2<T>;
        case ReductionTypeMean: return ReduceMean<T>;
        default:
            return NULL;
        }
    }

    ReductionLayer::ReducePtr GetReduce(TensorType src, ReductionType type)
    {
        switch (src)
        {
        case TensorType32f: return GetReduce<float>(type);
        case TensorType32i: return GetReduce<int32_t>(type);
        case TensorType64i: return GetReduce<int64_t>(type);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    ReductionLayer::ReductionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool ReductionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ReductionLayer supports only 1 input and 1 output!");
        const ReductionParam & param = this->Param().reduction();
        _srcType = src[0]->GetType();
        _reduce = GetReduce(_srcType, param.type());
        if(_reduce == NULL)
            SYNET_ERROR("RecuctionLayer can't reduce src " << Cpl::ToStr(_srcType) << " for operation " << Cpl::ToStr(param.type()) << " !");

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
        if(_outer * _count * _inner != src[0]->Size())
            SYNET_ERROR("RecuctionLayer can't reduce src " << ToStr(src[0]->Shape()) << " for axis " << ToStr(Shp(param.axis())) << " !");

        dst[0]->Reshape(_srcType, shape, src[0]->Format());
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void ReductionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _reduce(src[0]->RawData(), _outer, _count, _inner, dst[0]->RawData());
    }
}