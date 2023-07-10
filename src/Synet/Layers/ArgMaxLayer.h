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
    template <class T> class ArgMaxLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ArgMaxLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 1);
            const ArgMaxParam & param = this->Param().argMax();
            Shape shape = src[0]->Shape();
            if (param.keepDims() || shape.size() == 1)
                shape[param.axis()] = 1;
            else
                shape.erase(shape.begin() + param.axis());
            _outer = src[0]->Size(0, param.axis());
            _count = src[0]->Axis(param.axis());
            _inner = src[0]->Size(param.axis() + 1);

            dst[0]->As64i().Reshape(shape, src[0]->Format());
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T * pSrc = src[0]->CpuData();
            int64_t * pDst = dst[0]->As64i().CpuData();
            memset(pDst, 0, _outer * _inner * sizeof(int64_t));
            for (size_t o = 0; o < _outer; ++o)
            {
                for (size_t c = 0; c < _count; ++c)
                {
                    for (size_t i = 0; i < _inner; ++i)
                    {
                        if (pSrc[(o * _count + c) * _inner + i] > pSrc[(o * _count + pDst[o * _inner + i]) * _inner + i])
                        {
                            pDst[o * _inner + i] = c;
                        }
                    }
                }
            }
        }

    private:
        size_t _outer, _count, _inner;
    };
}