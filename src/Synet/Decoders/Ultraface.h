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

#include "Synet/Network.h"

namespace Synet
{
    struct UltrafaceParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "boxes", "scores" }));
    };

    class UltrafaceDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Network<float> Net;

        UltrafaceDecoder()
        {
        }

        bool Init(const UltrafaceParam& param = UltrafaceParam())
        {
            _names = param.names();
            return true;
        }

        bool Enable() const
        {
            return _names.size();
        }

        Regions GetRegions(const float* boxes, const float* scores, size_t size, size_t srcW, size_t srcH, float threshold, float overlap) const
        {
            Regions regions;
            for (size_t i = 0; i < size; ++i) 
            {
                float score = scores[2 * i + 1];
                if (score < threshold)
                    continue;
                float x1 = Min(boxes[4 * i], boxes[4 * i + 2]) * srcW;
                float x2 = Max(boxes[4 * i], boxes[4 * i + 2]) * srcW;
                float y1 = Min(boxes[4 * i + 1], boxes[4 * i + 3]) * srcH;
                float y2 = Max(boxes[4 * i + 1], boxes[4 * i + 3]) * srcH;
                float w = x2 - x1;
                float h = y2 - y1;
                float max = Max(w, h);
                float dx = (max - w) / 2;
                float dy = (max - h) / 2;
                x1 -= dx;
                x2 += dx;
                y1 -= dy;
                y2 += dy;
                Region region;
                region.prob = score;
                region.id = 2;
                region.x = (x1 + x2) * 0.5f;
                region.y = (y1 + y2) * 0.5f;
                region.w = x2 - x1;
                region.h = y2 - y1;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t srcW, size_t srcH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* boxes = net.Dst(_names[0])->CpuData(Shp(b, 0, 0));
                const float* scores = net.Dst(_names[1])->CpuData(Shp(b, 0, 0));
                size_t size = net.Dst(_names[0])->Size(1, 2);
                result[b] = GetRegions(boxes, scores, size, srcW, srcH, threshold, overlap);
            }
            return result;
        }

    private:
        Strings _names;
    };
}


