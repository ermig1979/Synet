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
    struct YoloV11Param
    {
        CPL_PARAM_VALUE(int, classes, 17);
    };

    class YoloV11Decoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        YoloV11Decoder()
            : _netW(0)
            , _netH(0)
            , _classes(0)
        {
        }

        bool Init(size_t netW, size_t netH, const YoloV11Param& param = YoloV11Param())
        {
            _netW = netW;
            _netH = netH;
            _classes = param.classes();
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
            for (size_t i = 0; i < size; ++i, data++) 
            {
                Region region;
                region.x = data[0 * size] * kX;
                region.y = data[1 * size] * kY;
                region.w = data[2 * size] * kX;
                region.h = data[3 * size] * kY;
                for (size_t c = 0; c < _classes; ++c)
                {
                    float score = data[(c + 4) * size];
                    if (score > region.prob)
                    {
                        region.prob = score;
                        region.id = c;
                    }
                }
                if (region.prob >= threshold)
                    regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            const Net::Tensor & dst = *net.Dst()[0];
            assert(dst.Count() == 3 && dst.Axis(0) == result.size());
            size_t size = dst.Axis(2);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst.Data<float>();
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            assert(dst[0].Count() == 3 && dst[0].Axis(0) == result.size());
            size_t size = dst[0].Axis(2);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst[0].Data<float>();
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        size_t _netW, _netH, _classes;

    };
}


