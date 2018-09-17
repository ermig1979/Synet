/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
    template <class T> class PadLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PadLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->Count() == src[0]->Count() * 2);
            Shape raw = src[1]->Shape();
            size_t n = src[0]->Count();
            _padB.resize(n);
            _padE.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                _padB[i] = raw[i * 2 + 0];
                _padE[i] = raw[i * 2 + 1];
            }
            if (n == 4)
            {
                _padB = Shape({ _padB[0], _padB[3] , _padB[1] , _padB[2] });
                _padE = Shape({ _padE[0], _padE[3] , _padE[1] , _padE[2] });
            }
            Shape dstShape = src[0]->Shape();
            for (size_t i = 0; i < n; ++i)
                dstShape[i] += _padB[i] + _padE[i];
            dst[0]->Reshape(dstShape, 0);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            size_t size = src[0]->Axis(-1);
            switch (src[0]->Count())
            {
            case 4:
                for (size_t b = 0; b < src[0]->Axis(0); ++b)
                {
                    for (size_t c = 0; c < src[0]->Axis(1); ++c)
                    {
                        for (size_t y = 0; y < src[0]->Axis(2); ++y)
                        {
                            const Type * pSrc = src[0]->CpuData({ b, c, y, size_t(0) });
                            Type * pDst = dst[0]->CpuData({ b + _padB[0], c + _padB[1], y + _padB[2], _padB[3] });
                            CpuCopy(pSrc, size, pDst);
                        }
                    }
                }
                break;
            default:
                assert(0);
            }
        }

    private:
        Shape _padB, _padE;
    };
}