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

#include "Synet/Layers/Detection/YoloLayer.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    YoloLayer::YoloLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool YoloLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("YoloLayer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("YoloLayer supports only FP32!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("YoloLayer supports only 4D tensors!");
        if (src[0]->Format() != TensorFormatNchw)
            SYNET_ERROR("YoloLayer: supports only NCHW input tensor format!");
        const YoloParam & param = this->Param().yolo();
        _num = param.num();
        _mask = param.mask();
        if(_num != _mask.size())
            SYNET_ERROR("YoloLayer: check yolo().num() and yolo().mask() parameters!");
        _classes = param.classes();
        _channels = _num * (_classes + 5);
        _anchors = param.anchors();
        for(size_t i = 0; i < _mask.size(); ++i)
            if(_mask[i] * 2 + 1 >= _anchors.size())
                SYNET_ERROR("YoloLayer: check yolo().mask() and yolo().anchors() parameters!");
        _batch = src[0]->Axis(0);
        if(_channels != src[0]->Axis(1))
            SYNET_ERROR("YoloLayer: wrong input shape!");
        _height = src[0]->Axis(2);
        _width = src[0]->Axis(3);
        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        this->UsePerfStat();
        _const = false;
        return true;
    }

    void YoloLayer::GetRegions(const TensorPtrs & src, size_t netW, size_t netH, float threshold, Regions & dst) const
    {
        SYNET_PERF_FUNC();
        dst.clear();
        size_t b = 0;
        for (size_t y = 0; y < _height; ++y)
        {
            for (size_t x = 0; x < _width; ++x)
            {
                for (size_t n = 0; n < _num; ++n)
                {
                    float objectness = src[0]->Data<float>(Shp(b, n*(_classes + 5) + 4, y, x))[0];
                    if (objectness > threshold)
                    {
                        Region region;
                        region.x = (x + src[0]->Data<float>(Shp(b, n*(_classes + 5) + 0, y, x))[0]) / _width;
                        region.y = (y + src[0]->Data<float>(Shp(b, n*(_classes + 5) + 1, y, x))[0]) / _height;
                        region.w = ::exp(src[0]->Data<float>(Shp(b, n*(_classes + 5) + 2, y, x))[0])*_anchors[2*_mask[n] + 0] / netW;
                        region.h = ::exp(src[0]->Data<float>(Shp(b, n*(_classes + 5) + 3, y, x))[0])*_anchors[2*_mask[n] + 1] / netH;
                        for (size_t i = 0; i < _classes; ++i)
                        {
                            region.id = (int)i;
                            region.prob = objectness* src[0]->Data<float>(Shp(b, n*(_classes + 5) + 5 + i, y, x))[0];
                            if (region.prob > threshold)
                                dst.push_back(region);
                        }
                    }
                }
            }
        }
    }

    void YoloLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        size_t area = src[0]->Axis(2) * src[0]->Axis(3);
        Index i(4, 0);
        for (i[0] = 0; i[0] < _batch; ++i[0])
        {
            for (size_t n = 0; n < _num; ++n)
            {
                i[1] = n * (_classes + 5);
                CpuSigmoid(src[0]->Data<float>(i), 2 * area, dst[0]->Data<float>(i));
                i[1] += 2;
                CpuCopy(src[0]->Data<float>(i), 2 * area, dst[0]->Data<float>(i));
                i[1] += 2;
                CpuSigmoid(src[0]->Data<float>(i), (_classes + 1) * area, dst[0]->Data<float>(i));
            }
        }
    }
}