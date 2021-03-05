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
#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T> class ReverseSequenceLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReverseSequenceLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            size_t seqAxis = this->Param().reverseSequence().seqAxis(), i = 0;
            const Shape & shape = src[0]->Shape();
            assert(seqAxis < shape.size());
            for (i = 0, _outer = 1; i < seqAxis; i++)
                _outer *= shape[i];
            _reverse = shape[seqAxis];
            for (i = seqAxis + 1, _inner = 1; i < shape.size(); ++i)
                _inner *= shape[i];
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T* pSrc = src[0]->CpuData();
            T * pDst = dst[0]->CpuData();
            if (_inner == 1)
            {
                for (size_t o = 0; o < _outer; ++o)
                {
                    for (size_t r = 0; r < _reverse; ++r)
                        pDst[r] = pSrc[_reverse - 1 - r];
                    pDst += _reverse;
                    pSrc += _reverse;
                }
            }
            else
            {
                for (size_t o = 0; o < _outer; ++o)
                {
                    for (size_t r = 0; r < _reverse; ++r)
                    {
                        for (size_t r = 0; r < _reverse; ++r)
                            memcpy(pDst + r * _inner, pSrc + (_reverse - 1 - r) * _inner, _inner * sizeof(T));
                        pDst += _reverse*_inner;
                        pSrc += _reverse * _inner;
                    }
                }
            }
        }

    private:
        size_t _outer, _reverse, _inner;
    };
}