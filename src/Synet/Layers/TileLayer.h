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
    template <class T> class TileLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        TileLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Shape shape = src[0]->Shape();
            size_t axis = 0;
            if (src.size() == 1)
            {
                axis = this->Param().tile().axis();
                _tiles = this->Param().tile().tiles();
            }
            else
            {
                assert(src[0]->Count() == src[1]->Size() && src[1]->GetType() == TensorType64i);
                const int64_t * pSrc1 = src[1]->As64i().CpuData();
                _tiles = 1;
                for (size_t a = 0; a < src[1]->Size(); ++a)
                {
                    if (pSrc1[a] != src[0]->Axis(a))
                    {
                        axis = a;
                        _tiles = (size_t)pSrc1[a];
                        break;
                    }
                }
            }
            shape[axis] *= _tiles;
            _outer = src[0]->Size(0, axis);
            _inner = src[0]->Size(axis);
            dst[0]->Reshape(shape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T * pSrc = src[0]->CpuData();
            T * pDst = dst[0]->CpuData();
            if (_inner == 1)
            {
                for (size_t o = 0; o < _outer; ++o, pDst += _tiles)
                    CpuSet(_tiles, pSrc[o], pDst);
            }
            else
            {
                for (size_t o = 0; o < _outer; ++o)
                {
                    for (size_t t = 0; t < _tiles; ++t)
                    {
                        CpuCopy(pSrc, _inner, pDst);
                        pDst += _inner;
                    }
                    pSrc += _inner;
                }
            }
        }

    private:
        size_t _tiles, _outer, _inner;
    };
}