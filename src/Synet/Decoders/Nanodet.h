/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Network.h"

namespace Synet
{
    struct NanodetParam
    {
        CPL_PARAM_VALUE(int, classes, 80);
        CPL_PARAM_VALUE(int, regMax, 7);
        CPL_PARAM_VALUE(Ints, strides, Ints({ 8, 16, 32, 64 }));
    };

    class NanodetDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        NanodetDecoder()
            : _netW(0)
            , _netH(0)
            , _classes(0)
            , _regMax(-1)
        {
        }

        bool Init(size_t netW, size_t netH, const NanodetParam& param = NanodetParam())
        {
            _netW = netW;
            _netH = netH;
            _classes = param.classes();
            _regMax = param.regMax();
            _strides = param.strides();
            _anchors.clear();
            for (size_t s = 0; s < _strides.size(); ++s)
            {
                int stride = _strides[s];
                int featH = (int)::ceil(_netH / stride);
                int featW = (int)::ceil(_netW / stride);
                for (int h = 0; h < featH; ++h)
                {
                    for (int w = 0; w < featW; ++w)
                    {
                        _anchors.push_back(w * stride);
                        _anchors.push_back(h * stride);
                        _anchors.push_back(stride);
                    }
                }
            }
            return true;
        }

        bool Enable() const
        {
            return _classes != 0;
        }

        Regions GetRegions(const float* data, size_t size, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(_netW);
            float kY = float(imgH) / float(_netH);
            const float* anchors = _anchors.data();
            Regions regions;
            size_t predSize = _regMax + 1;
            for (size_t i = 0; i < size; ++i, data += _classes + 4 * predSize, anchors += 3)
            {
                float score = data[0];
                size_t index = 0;
                for (size_t c = 1; c < _classes; ++c)
                {
                    if (data[c] > score)
                    {
                        score = data[c];
                        index = c;
                    }
                }
                if (score < threshold)
                    continue;

                float cx = anchors[0] * kX, cy = anchors[1] * kY;
                float px = anchors[2] * kX, py = anchors[2] * kY;
                const float* pred = data + _classes;

                float pdx0 = PredDist(pred + 0 * predSize, predSize) * px;
                float pdy0 = PredDist(pred + 1 * predSize, predSize) * py;
                float pdx1 = PredDist(pred + 2 * predSize, predSize) * px;
                float pdy1 = PredDist(pred + 3 * predSize, predSize) * py;

                float xMin = std::max(cx - pdx0, .0f);
                float yMin = std::max(cy - pdy0, .0f);
                float xMax = std::min(cx + pdx1, (float)imgW);
                float yMax = std::min(cy + pdy1, (float)imgH);

                Region region;
                region.prob = score;
                region.id = (int)index;
                region.x = (xMin + xMax) / 2.0f;
                region.y = (yMin + yMax) / 2.0f;
                region.w = xMax - xMin;
                region.h = yMax - yMin;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = net.Dst()[0]->Data<float>();
                size_t size = net.Dst()[0]->Size(1, 2);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst[0].Data<float>();
                size_t size = dst[0].Size(1, 2);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        size_t _netW, _netH, _classes, _regMax;
        Ints _strides;
        Floats _anchors;

        inline float PredDist(const float* src, size_t count) const
        {
            float max = -FLT_MAX;
            for (size_t i = 0; i < count; ++i)
                max = std::max(max, src[i]);
            float sum = 0;
            for (size_t i = 0; i < count; ++i)
                sum += ::exp(src[i] - max);
            float dst = 0;
            for (size_t i = 0; i < count; ++i)
                dst += i * ::exp(src[i] - max) / sum;
            return dst;
        }
    };
}


