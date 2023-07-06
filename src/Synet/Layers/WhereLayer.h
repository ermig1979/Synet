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
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void WhereLayerForwardCpu(const T* cond, const T* pos, const T* neg, size_t size, T* dst);

        template <> SYNET_INLINE void WhereLayerForwardCpu<float>(const float* cond, const float* pos, const float* neg, size_t size, float* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = ((uint32_t*)cond)[i] ? pos[i] : neg[i];
        }
    }

    template <class T> class WhereLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        WhereLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 3 && src[0]->Size() == src[1]->Size() && src[0]->Size() == src[2]->Size());
            _size = src[0]->Size();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::WhereLayerForwardCpu(src[0]->CpuData(), src[1]->CpuData(), src[2]->CpuData(), _size, dst[0]->CpuData());
        }

    private:
        size_t _size;
    };
}