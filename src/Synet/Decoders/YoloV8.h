/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
    class YoloV8Decoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Network Net;

        YoloV8Decoder()
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

        Regions GetRegions(const float* data, size_t size, size_t netW, size_t netH, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(netW);
            float kY = float(imgH) / float(netH);
            Regions regions;
            for (size_t i = 0; i < size; ++i, data++) 
            {
                float score = data[4 * size];
                if (score < threshold)
                    continue;
                Region region;
                region.prob = score;
                region.x = data[0 * size] * kX;
                region.y = data[1 * size] * kY;
                region.w = data[2 * size] * kX;
                region.h = data[3 * size] * kY;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            const Net::Tensor & dst = *net.Dst()[0];
            assert(dst.Count() == 3 && dst.Axis(0) == result.size() && dst.Axis(1) == 5);
            size_t size = dst.Axis(2);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst.Data<float>();
                result[b] = GetRegions(data, size, net.NchwShape()[3], net.NchwShape()[2], imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        bool _enable;
    };
}


