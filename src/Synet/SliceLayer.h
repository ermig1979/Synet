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
#include "Synet/Math.h"

namespace Synet
{
    template <class T> class SliceLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SliceLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const SliceParam & param = this->Param().slice();
            _sliceAxis = param.axis();            
            _slicePoint = param.slicePoint();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const SliceParam & param = this->Param().slice();
            Shape dstShape = src[0]->Shape();
            size_t srcSliceAxis = src[0]->Axis(_sliceAxis);
            _numSlices = src[0]->Size(0, _sliceAxis);
            _sliceSize = src[0]->Size(_sliceAxis + 1);
            size_t size = 0;
            if (_slicePoint.size() != 0)
            {
                assert(_slicePoint.size() == dst.size() - 1);
                assert(dst.size() <= srcSliceAxis);
                size_t prev = 0;
                Shape slices;
                for (size_t i = 0; i < _slicePoint.size(); ++i)
                {
                    assert(_slicePoint[i] > prev);
                    slices.push_back(_slicePoint[i] - prev);
                    prev = _slicePoint[i];
                }
                slices.push_back(srcSliceAxis - prev);
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    dstShape[_sliceAxis] = slices[i];
                    dst[i]->Reshape(dstShape);
                    size += dst[i]->Size();
                }
            }
            else
            {
                assert(srcSliceAxis % dst.size() == 0);
                dstShape[_sliceAxis] = srcSliceAxis / dst.size();
                for (int i = 0; i < dst.size(); ++i) 
                {
                    dst[i]->Reshape(dstShape);
                    size += dst[i]->Size();
                }
            }
            assert(size == src[0]->Size());
            if (dst.size() == 1) 
                dst[0]->Share(*src[0]);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            if (dst.size() == 1) 
                return;
            size_t offsetSliceAxis = 0;
            const Type * pSrc = src[0]->CpuData();
            size_t srcSliceAxis = src[0]->Axis(_sliceAxis);
            for (size_t i = 0; i < dst.size(); ++i)
            {
                Type * pDst = dst[i]->CpuData();
                size_t dstSliceAxis = dst[i]->Axis(_sliceAxis);
                for (int n = 0; n < _numSlices; ++n)
                {
                    size_t dstOffset = n * dstSliceAxis * _sliceSize;
                    size_t srcOffset = (n * srcSliceAxis + offsetSliceAxis) * _sliceSize;
                    CpuCopy(pSrc + srcOffset, dstSliceAxis * _sliceSize, pDst + dstOffset);
                }
                offsetSliceAxis += dstSliceAxis;
            }
        }

    private:
        size_t _count, _numSlices, _sliceSize, _sliceAxis;
        Index _slicePoint;
    };
}