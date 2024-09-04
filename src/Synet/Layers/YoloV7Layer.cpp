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

#include "Synet/Layers/YoloV7Layer.h"

namespace Synet
{
    YoloV7Layer::YoloV7Layer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool YoloV7Layer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("YoloV7Layer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("YoloV7Layer supports only FP32!");
        if (src[0]->Count() != 3)
            SYNET_ERROR("YoloV7Layer supports only 3D tensors!");

        const YoloV7Param & param = this->Param().yoloV7();
        _maxOutputBoxesPerClass = param.maxOutputBoxesPerClass();
        _iouThreshold = param.iouThreshold();
        _scoreThreshold = param.scoreThreshold();
        _oneClass = param.oneClass() ? 1 : 0;
        if (src[0]->Count() != 3 || src[0]->Axis(0) != 1 || src[0]->Axis(2) < 6)
            SYNET_ERROR("YoloV7Layer has wrong input shape!");
        _size = (int)src[0]->Axis(1);
        _num = (int)src[0]->Axis(2);
        _numClasses = _num - 5;
        Base::Extend32f(buf, 0, Shp(src[0]->Axis(1) * 7), src[0]->Format());
        dst[0]->Reshape(TensorType32f, Shp(0, 7), TensorFormatUnknown);
        _const = false;
        this->UsePerfStat();
        return true;
    }

    void YoloV7Layer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        Box* boxes = (Box*)Base::Buf32f(buf, 0);
        size_t candidates = FindCandidates(src[0]->CpuData(), boxes);
        size_t detections = FilterByIou(boxes, candidates);
        SortByThreshold(boxes, detections);
        dst[0]->Reshape(TensorType32f, Shp(detections, 7), TensorFormatUnknown);
        memcpy(dst[0]->CpuData(), boxes, detections * sizeof(Box));
    }

    size_t YoloV7Layer::FindCandidates(const float* src, Box* dst) const
    {
        size_t count = 0;
        for (size_t i = 0; i < _size; ++i)
        {
            if (_oneClass)
            {
                dst[count].type = 0.0f;
                dst[count].score = src[4];
            }
            else
            {
                dst[count].type = 0.0f;
                dst[count].score = src[5];
                for (int c = 1; c < _numClasses; ++c)
                {
                    if (src[5 + c] > dst[count].score)
                    {
                        dst[count].type = (float)c;
                        dst[count].score = src[5 + c];
                    }
                }
                dst[count].score *= src[4];
            }
            if (dst[count].score >= _scoreThreshold)
            {
                dst[count].stub = 0.0f;
                dst[count].left = src[0] - src[2] * 0.5f;
                dst[count].top = src[1] - src[3] * 0.5f;
                dst[count].right = src[0] + src[2] * 0.5f;
                dst[count].bottom = src[1] + src[3] * 0.5f;
                count++;
            }
            src += _num;
        }
        return count;
    }

    SYNET_INLINE float Iou(const YoloV7Layer::Box & a, const YoloV7Layer::Box& b)
    {
        float aArea = (a.right - a.left) * (a.bottom - a.top);
        float bArea = (b.right - b.left) * (b.bottom - b.top);
        float left = RestrictRange(a.left, b.left, b.right);
        float top = RestrictRange(a.top, b.top, b.bottom);
        float right = RestrictRange(a.right, b.left, b.right);
        float bottom = RestrictRange(a.bottom, b.top, b.bottom);
        float iArea = (right - left) * (bottom - top);
        return iArea / (aArea + bArea - iArea);
    }

    size_t YoloV7Layer::FilterByIou(Box* boxes, size_t size) const
    {
        size_t count = 0;
        for (size_t i = 0; i < size; ++i)
        {
            size_t j = 0;
            const Box& src = boxes[i];
            for (; j < count; ++j)
            {
                Box & dst = boxes[j];
                if (src.type != dst.type)
                    continue;
                if (Iou(src, dst) >= _iouThreshold)
                {
                    if (src.score > dst.score)
                        dst = src;
                    break;
                }
            }
            if (j == count)
                boxes[count++] = src;
        }
        return count;
    }

    void YoloV7Layer::SortByThreshold(Box* boxes, size_t size) const
    {
        std::sort(boxes, boxes + size, [](const Box& a, const Box& b) {return a.score > b.score; });
    }
}