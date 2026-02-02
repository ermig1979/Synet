/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/SliceLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    SliceLayer::SliceLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool SliceLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1)
            SYNET_ERROR("SliceLayer supports only 2 input!");
        const SliceParam & param = this->Param().slice();
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("SliceLayer input must be FP32 type!");
        _sliceAxis = param.axis();            
        _slicePoint = param.slicePoint();
        Shape dstShape = src[0]->Shape();
        size_t srcSliceAxis = src[0]->Axis(_sliceAxis);
        _numSlices = src[0]->Size(0, _sliceAxis);
        _sliceSize = src[0]->Size(_sliceAxis + 1);
        size_t size = 0;
        if (_slicePoint.size() != 0)
        {
            if(_slicePoint.size() != dst.size() - 1)
                SYNET_ERROR("SliceLayer parameter slice().slicePoint() is incompatible with outputs number!");
            if (dst.size() > srcSliceAxis)
                SYNET_ERROR("SliceLayer input shape is incompatible with outputs number!");
            size_t prev = 0;
            Shape slices;
            for (size_t i = 0; i < _slicePoint.size(); ++i)
            {
                if (_slicePoint[i] <= prev)
                    SYNET_ERROR("SliceLayer parameter slice().slicePoint() is wrong!");
                slices.push_back(_slicePoint[i] - prev);
                prev = _slicePoint[i];
            }
            slices.push_back(srcSliceAxis - prev);
            for (size_t i = 0; i < dst.size(); ++i)
            {
                dstShape[_sliceAxis] = slices[i];
                dst[i]->Reshape(src[0]->GetType(), dstShape, src[0]->Format());
                size += dst[i]->Size();
            }
        }
        else
        {
            if (srcSliceAxis % dst.size() != 0)
                SYNET_ERROR("SliceLayer input shape is incompatible with outputs number!");
            dstShape[_sliceAxis] = srcSliceAxis / dst.size();
            for (int i = 0; i < dst.size(); ++i) 
            {
                dst[i]->Reshape(src[0]->GetType(), dstShape, src[0]->Format());
                size += dst[i]->Size();
            }
        }
        if(size != src[0]->Size())
            SYNET_ERROR("SliceLayer: can't perform slice oparation!");
        if (dst.size() == 1)
        {
            dst[0]->Share(*src[0]);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void SliceLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        size_t offsetSliceAxis = 0;
        const float * pSrc = src[0]->Data<float>();
        size_t srcSliceAxis = src[0]->Axis(_sliceAxis);
        for (size_t i = 0; i < dst.size(); ++i)
        {
            float * pDst = dst[i]->Data<float>();
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
}