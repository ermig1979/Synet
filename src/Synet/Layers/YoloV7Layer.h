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
    template <class T> class YoloV7Layer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef Synet::Region<T> Region;
        typedef std::vector<Region> Regions;

        YoloV7Layer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const YoloV7Param & param = this->Param().yoloV7();
            _maxOutputBoxesPerClass = param.maxOutputBoxesPerClass();
            _iouThreshold = param.iouThreshold();
            _scoreThreshold = param.scoreThreshold();
            _softNmsSigma = param.softNmsSigma();
            _scale = param.scale();
            assert(src.size() == 1 && src[0]->Count() == 3 && src[0]->Axis(0) == 1 && src[0]->Axis(2) >= 6);
            _size = (int)src[0]->Axis(1);
            _num = (int)src[0]->Axis(2);
            _numClasses = _num - 5;
            Base::Extend32f(buf, 0, Shp(src[0]->Axis(1) * 7), src[0]->Format());
            dst[0]->Reshape(Shp(0, 7));
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Box* boxes = (Box*)Base::Buf32f(buf, 0);
            size_t candidates = FindCandidates(src[0]->CpuData(), boxes);
            size_t detections = FilterByIou(boxes, candidates);
            SortByThreshold(boxes, detections);
            dst[0]->Reshape(Shp(detections, 7));
            memcpy(dst[0]->CpuData(), boxes, detections * sizeof(Box));
        }

    private:
        int _maxOutputBoxesPerClass, _numClasses, _size, _num;
        float _scale, _iouThreshold, _scoreThreshold, _softNmsSigma;

        struct Box
        {
            float stub, left, top, right, bottom, type, score;
        };

        size_t FindCandidates(const float* src, Box* dst) const
        {
            size_t count = 0;
            for (size_t i = 0; i < _size; ++i)
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

        static SYNET_INLINE float Iou(const Box & a, const Box& b)
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

        size_t FilterByIou(Box* boxes, size_t size) const
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

        void SortByThreshold(Box* boxes, size_t size) const
        {
            std::sort(boxes, boxes + size, [](const Box& a, const Box& b) {return a.score > b.score; });
        }
    };
}