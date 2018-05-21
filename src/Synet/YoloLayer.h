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
    template <class T> class YoloLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef Synet::Region<T> Region;
        typedef std::vector<Region> Regions;

        YoloLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const YoloParam & param = this->Param().yolo();
            _num = param.num();
            _total = param.total();
            _classes = param.classes();
            _anchors.resize(param.anchors().size());
            for (size_t i = 0; i < param.anchors().size(); ++i)
                _anchors[i] = param.anchors()[i];
            _mask.resize(param.mask().size());
            for (size_t i = 0; i < param.mask().size(); ++i)
                _mask[i] = param.mask()[i];
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Shape dstShape = src[0]->Shape();
            dstShape[1] = _num*(_classes + 4 + 1);
            dst[0]->Reshape(dstShape);
        }

        void GetRegions(const TensorPtrs & src, size_t width, size_t height, Type threshold, bool relative, bool letter, Regions & dst)
        {
            SYNET_PERF_FUNC();
            dst.clear();

            size_t b = 0;
            size_t w = src[0]->Axis(2);
            size_t h = src[0]->Axis(3);
            size_t W = 0;
            size_t H = 0;
            if (letter)
            {
                if (((float)w / width) < ((float)h / h))
                {
                    W = w;
                    H = (height * w) / width;
                }
                else
                {
                    W = (width * h) / height;
                    H = h;
                }
            }
            else
            {
                W = w;
                H = h;
            }
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    for (size_t n = 0; n < _num; ++n)
                    {
                        Type objectness = src[0]->CpuData({ b, n*(_classes + 5) + 4, y, x });
                        if (objectness > threshold)
                        {
                            Region region;
                            region.x = (x + src[0]->CpuData({ b, n*(_classes + 5) + 0, y, x })) / w;
                            region.y = (y + src[0]->CpuData({ b, n*(_classes + 5) + 1, y, x })) / h;
                            region.w = ::exp(src[0]->CpuData({ b, n*(_classes + 5) + 2, y, x }))*_anchors[2*_mask[n] + 0] / w;
                            region.h = ::exp(src[0]->CpuData({ b, n*(_classes + 5) + 3, y, x }))*_anchors[2*_mask[n] + 1] / h;
                            for (size_t i = 0; i < _classes; ++i)
                            {
                                region.id = i;
                                region.prob = objectness*src[0]->CpuData({ b, n*(_classes + 5) + 5 + i, y, x });
                                if (region.prob > threshold)
                                {
                                    region.x = (region.x - (netw - W) / 2. / w) / ((float)W / w);
                                    region.y = (region.y - (neth - H) / 2. / h) / ((float)H / h);
                                    region.w *= (float)w / W;
                                    region.h *= (float)h / H;
                                    if (!relative) 
                                    {
                                        region.x *= width;
                                        region.w *= width;
                                        region.y *= height;
                                        region.h *= height;
                                    }
                                    dst.push_back(region)
                                }
                            }
                        }
                    }
                }
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            size_t batch = src[0]->Axis(0);
            size_t area = src[0]->Axis(2)*src[0]->Axis(3);
            Index index(4, 0);
            for (index[0] = 0; index[0] < batch; ++index[0])
            {
                for (size_t n = 0; n < _num; ++n)
                {
                    index[1] = n*(_classes + 4 + 1);
                    CpuSigmoid(src[0]->CpuData(index), 2 * area, dst[0]->CpuData(index));
                    index[1] += 2;
                    CpuCopy(src[0]->CpuData(index), 2 * area, dst[0]->CpuData(index));
                    index[1] += 2;
                    CpuSigmoid(src[0]->CpuData(index), (_classes + 1) * area, dst[0]->CpuData(index));
                }
            }
        }

    private:
        typedef std::vector<Type> VectorF;
        typedef std::vector<size_t> VectorI;

        size_t _total, _num, _classes;
        VectorF _anchors;
        VectorI _mask;
    };
}