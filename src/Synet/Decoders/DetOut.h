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

#pragma once

#include "Synet/Network.h"

namespace Synet
{
    class DetOutDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef Synet::Network Net;

        DetOutDecoder()
            : _enable(false)
        {
        }

        bool Init()
        {
            _enable = true;
            return true;
        }

        bool Enable() const
        {
            return _enable;
        }

        Regions GetRegions(const float* data, size_t size, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW);
            float kY = float(imgH);
            Regions regions;
            for (size_t i = 0; i < size; ++i, data += 7) 
            {
                float score = data[2];
                if (score < threshold)
                    continue;
                Region region;
                region.prob = score;
                region.id = (int)data[1];
                region.x = (data[5] + data[3]) * kX / 2.0f;
                region.y = (data[6] + data[4]) * kY / 2.0f;
                region.w = (data[5] - data[3]) * kX;
                region.h = (data[6] - data[4]) * kY;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap, size_t thread = 0) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size() && net.Dst(thread)[0]->Size(); ++b)
            {
                const float* data = net.Dst(thread)[0]->Data<float>(Shp(b, 0, 0, 0));
                size_t size = net.Dst(thread)[0]->Axis(2);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensor& dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst.Axis(0));
            for (size_t b = 0; b < result.size() && dst.Size(); ++b)
            {
                const float* data = dst.Data<float>(Shp(b, 0, 0, 0));
                size_t size = dst.Axis(2);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        bool _enable;
    };
}


