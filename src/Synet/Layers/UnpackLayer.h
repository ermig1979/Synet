/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
    template <class T> class UnpackLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        UnpackLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const UnpackParam & param = this->Param().unpack();
            _axis = src[0]->Index(param.axis());
            _count = dst.size();
            _step = src[0]->Axis(_axis) / _count;
            assert(src[0]->Axis(_axis) == _count*_step);
            _outer = src[0]->Size(0, _axis);
            _inner = src[0]->Size(_axis + 1);
            Shape shape;
            for (size_t i = 0; i < _axis; ++i)
                shape.push_back(src[0]->Axis(i));
            shape.push_back(_step);
            for (size_t i = _axis + 1; i < src[0]->Count(); ++i)
                shape.push_back(src[0]->Axis(i));
            if (dst.size() > 1)
            {
                for (size_t i = 0; i < _count; ++i)
                    dst[i]->Reshape(shape, src[0]->Format());
            }
            else
                dst[0]->ShareAs(*src[0], shape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (dst.size() > 1)
            {
                for (size_t o = 0; o < _outer; ++o)
                {
                    for (size_t c = 0; c < _count; c += 1)
                    {
                        const Type * pSrc = src[0]->CpuData() + (_count*o + c)*_step*_inner;
                        Type * pDst = dst[c]->CpuData() + o*_step*_inner;
                        CpuCopy(pSrc, _inner*_step, pDst);
                    }
                }
            }
        }

    private:
        size_t _axis, _outer, _count, _inner, _step;
    };
}