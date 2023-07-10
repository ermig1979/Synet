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
#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T> class TopKLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        TopKLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const TopKParam & param = this->Param().topK();
            Shape shape = src[0]->Shape();
            shape[param.axis()] = 1;
            _outer = src[0]->Size(0, param.axis());
            _count = src[0]->Axis(param.axis());
            _inner = src[0]->Size(param.axis() + 1);
            _type = param.indexElementType();

            dst[0]->Reshape(shape, src[0]->Format());
            if (_type == TensorType32i)
                dst[1]->As32i().Reshape(shape, src[0]->Format());
            else if (_type == TensorType64i)
                dst[1]->As64i().Reshape(shape, src[0]->Format());
            else
                assert(0);
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T * pSrc = src[0]->CpuData();
            T * pDst = dst[0]->CpuData();
            switch (_type)
            {
            case TensorType32i: ForwardCpu<int32_t>(pSrc, pDst, dst[1]->As32i().CpuData()); break;
            case TensorType64i: ForwardCpu<int64_t>(pSrc, pDst, dst[1]->As64i().CpuData()); break;
            default:
                assert(0);
            }
        }

        template<class I> void ForwardCpu(const T * src, T * dst, I * idx)
        {
        }

    private:
        size_t _outer, _count, _inner;
        TensorType _type;
    };
}