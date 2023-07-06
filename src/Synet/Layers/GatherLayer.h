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
    template <class T> class GatherLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        GatherLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const GatherParam& param = this->Param().gather();
            _axis = param.axis();
            _batch = src[0]->Size(0, _axis);
            _size = src[0]->Size(_axis);
            assert(src.size() == 2 && (src[1]->GetType() == TensorType32i || src[1]->GetType() == TensorType64i));
            dst[0]->Reshape(src[1]->Shape(), src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            if (src[1]->GetType() == TensorType32i)
                Gather(src[0]->CpuData(), src[1]->As32i().CpuData(), dst[0]->CpuData());
            else
                Gather(src[0]->CpuData(), src[1]->As64i().CpuData(), dst[0]->CpuData());
        }

        template <class Index> void Gather(const Type* src, const Index* idx, Type* dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                for (size_t i = 0; i < _size; ++i)
                    dst[i] = src[idx[i]];
                src += _size;
                dst += _size;
            }
        }

        size_t _axis, _batch, _size;
    };
}