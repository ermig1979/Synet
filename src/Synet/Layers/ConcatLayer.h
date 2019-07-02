/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
    template <class T> class ConcatLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ConcatLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _concatAxis = this->Param().concat().axis();
            _concatNum = src[0]->Size(0, _concatAxis);
            _concatInputSize = src[0]->Size(_concatAxis + 1);
            size_t srcSizeSum = src[0]->Size();
            Shape dstShape = src[0]->Shape();
            _srcConcatAxis.resize(src.size());
            _srcConcatAxis[0] = src[0]->Axis(_concatAxis);
            _src.resize(src.size());
            for (size_t i = 1; i < src.size(); ++i)
            {
                assert(src[0]->Count() == src[i]->Count());
                for (size_t j = 0; j < src[0]->Count(); ++j)
                {
                    if (_concatAxis)
                        continue;
                    assert(dstShape[j] == src[i]->Axis(j));
                }
                srcSizeSum += src[i]->Size();
                _srcConcatAxis[i] = src[i]->Axis(_concatAxis);
                dstShape[_concatAxis] += _srcConcatAxis[i];
            }
            dst[0]->Reshape(dstShape, src[0]->Format());
            assert(srcSizeSum == dst[0]->Size());
            if (src.size() == 1)
                dst[0]->Share(*src[0]);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            if (src.size() == 1)
                return;

            Type * pDst = dst[0]->CpuData();
            size_t dstConcatAxis = dst[0]->Axis(_concatAxis);
            if (_concatInputSize == 1)
            {
                for (size_t i = 0; i < src.size(); ++i)
                    _src[i] = src[i]->CpuData();
                for (size_t n = 0; n < _concatNum; ++n)
                {
                    for (size_t i = 0; i < src.size(); ++i)
                    {
                        size_t size = _srcConcatAxis[i];
                        CpuCopy(_src[i], size, pDst);
                        _src[i] += size;
                        pDst += size;
                    }
                }
            }
            else
            {
                size_t concatAxisOffset = 0;
                for (size_t i = 0; i < src.size(); ++i)
                {
                    const Type * pSrc = src[i]->CpuData();
                    for (size_t n = 0; n < _concatNum; ++n)
                        CpuCopy(pSrc + n * _srcConcatAxis[i] * _concatInputSize, _srcConcatAxis[i] * _concatInputSize,
                            pDst + (n * dstConcatAxis + concatAxisOffset) * _concatInputSize);
                    concatAxisOffset += _srcConcatAxis[i];
                }
            }
        }

    private:
        typedef std::vector<Type*> Ptrs;
        size_t _concatNum, _concatInputSize, _concatAxis;
        Index _srcConcatAxis;
        Ptrs _src;
    };
}