/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
    struct YoloV5Param
    {
        CPL_PARAM_VALUE(int, classes, 2);
        CPL_PARAM_VALUE(int, background, 1);
    };

    class YoloV5Decoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Network Net;

        YoloV5Decoder()
            : _classes(0)
            , _background(-1)
        {
        }

        bool Init(const YoloV5Param& param = YoloV5Param())
        {
            _classes = param.classes();
            _background = param.background();
            return true;
        }

        bool Enable() const
        {
            return _classes != 0;
        }

        Regions GetRegions(const float* data, size_t size, size_t netW, size_t netH, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(netW);
            float kY = float(imgH) / float(netH);
            Regions regions;
            for (size_t i = 0; i < size; ++i, data += _classes + 5) 
            {
                float score = data[4];
                if (score < threshold)
                    continue;
                Region region;
                region.prob = score;
                float classMax = 0;
                for (size_t c = 0; c < _classes; ++c)
                {
                    if (data[5 + c] > classMax)
                    {
                        classMax = data[5 + c];
                        region.id = c;
                    }
                }
                if (region.id == _background)
                    continue;
                float xMin = std::max(0.0f, (data[0] - data[2] / 2) * kX);
                float xMax = std::min((float)imgW, (data[0] + data[2] / 2) * kX);
                float yMin = std::max(0.0f, (data[1] - data[3] / 2) * kY);
                float yMax = std::min((float)imgH, (data[1] + data[3] / 2) * kY);
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
                result[b] = GetRegions(data, size, net.NchwShape()[3], net.NchwShape()[2], imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        size_t _classes, _background;
    };
}


