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
    template <class T> class ScatterNdLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        ScatterNdLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _offset.Reshape(src[1]->Shape());
            size_t count = src[1]->Count(), size = src[1]->Size();
            const Synet::Tensor<int32_t> &idx = this->Weight()[0].As32i();
            assert(idx.Axis(-1) == count);
            for (size_t a = 0; a < count; ++a)
                assert(idx.Axis(a) == _offset.Axis(a));
            for (size_t o = 0, i = 0; o < size; ++o)
            {
                Shape index;
                for (size_t a = 0; a < count; ++a, ++i)
                    index.push_back(idx.CpuData()[i]);
                _offset.CpuData()[o] = (uint32_t)src[0]->Offset(index);
            }
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (src[0] != dst[0])
                memcpy(dst[0]->RawCpuData(), src[0]->RawCpuData(), src[0]->RawSize());
            const int32_t * pOffs = _offset.CpuData();
            const T * pSrc = src[1]->CpuData();
            T * pDst = dst[0]->CpuData();
            size_t size = src[1]->Size();
            for (size_t i = 0; i < size; ++i)
                pDst[pOffs[i]] = pSrc[i];
        }

    private:
        Synet::Tensor<int32_t> _offset;
    };
}