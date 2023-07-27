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

#include "Synet/Layers/CompareLayer.h"

namespace Synet
{
    template <class T> void CompareNN(const uint8_t* a8, const uint8_t* b8, size_t size, CompareType type, uint8_t* dst8)
    {
        const T* a = (const T*)a8;
        const T* b = (const T*)b8;
        T* dst = (T*)dst8;
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

    template <class T> void CompareN1(const uint8_t* a8, const uint8_t* b8, size_t size, CompareType type, uint8_t* dst8)
    {
        const T* a = (const T*)a8;
        T b = *(const T*)b8;
        T* dst = (T*)dst8;
        T neg = T(0), pos = Not(neg);
        switch (type)
        {
        case CompareTypeEqual:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] == b ? pos : neg;
            break;
        case CompareTypeNotEqual:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] != b ? pos : neg;
            break;
        case CompareTypeGreaterThan:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] > b ? pos : neg;
            break;
        case CompareTypeGreaterOrEqual:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] >= b ? pos : neg;
            break;
        case CompareTypeLessThan:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] < b ? pos : neg;
            break;
        case CompareTypeLessOrEqual:
            for (size_t i = 0; i < size; ++i)
                dst[i] = a[i] <= b ? pos : neg;
            break;
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    CompareLayer::ComparePtr GetCompare(TensorType type, bool n1)
    {
        switch (type)
        {
        case TensorType32f: return n1 ? CompareN1<float> : CompareNN<float>;
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    CompareLayer::CompareLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool CompareLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 && dst.size() != 1)
            SYNET_ERROR("CompareLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("CompareLayer src[0] type " << Cpl::ToStr(src[0]->GetType()) << " != src[1] type " << Cpl::ToStr(src[1]->GetType()) << " !");
        if(src[0]->Shape() != src[1]->Shape() && src[1]->Size() != 1)
            SYNET_ERROR("CompareLayer has incompartible input shapes src[0] :" << ToStr(src[0]->Shape()) << " and src[1]: " << ToStr(src[1]->Shape()) << " !");
        _compareType = this->Param().compare().type();
        _srcType = src[0]->GetType();
        _size = src[0]->Size();
        _compare = GetCompare(_srcType, src[1]->Size() == 1);
        if(_compare == NULL)
            SYNET_ERROR("CompareLayer don't support " << Cpl::ToStr(_srcType) << " input type !");
        dst[0]->Reshape(_srcType, src[0]->Shape(), src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
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

    void CompareLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _compare(src[0]->RawData(), src[1]->RawData(), _size, _compareType, dst[0]->RawData());
    }
}