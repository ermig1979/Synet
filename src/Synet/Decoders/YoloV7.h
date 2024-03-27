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
    class YoloV7Decoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        YoloV7Decoder()
            : _netW(0)
            , _netH(0)
        {
        }

        bool Init(size_t netW, size_t netH)
        {
            _netW = netW;
            _netH = netH;
            return _netW && _netH;
        }

        bool Enable() const
        {
            return _netW && _netH;
        }

        Regions GetRegions(const float* data, size_t size, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(_netW);
            float kY = float(imgH) / float(_netH);
            Regions regions;
            for (size_t i = 0; i < size; ++i, data += 7) 
            {
                float score = data[6];
                if (score < threshold)
                    continue;
                Region region;
                region.prob = score;
                region.id = (int)data[5];
                region.x = (data[3] + data[1]) * kX / 2.0f;
                region.y = (data[4] + data[2]) * kY / 2.0f;
                region.w = (data[3] - data[1]) * kX;
                region.h = (data[4] - data[2]) * kY;
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
                size_t size = net.Dst()[0]->Size(0, 1);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(1);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst[0].Data<float>();
                size_t size = dst[0].Size(0, 1);
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        size_t _netW, _netH;
    };
}


