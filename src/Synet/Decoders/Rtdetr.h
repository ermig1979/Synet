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
    class RtdetrDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        RtdetrDecoder()
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
            Regions regions;
            for (size_t i = 0; i < size; ++i, data += 6) 
            {
                float score = data[4];
                if (score < threshold)
                    continue;
                Region region;
                region.prob = score;
                region.x = data[0] * imgW;
                region.y = data[1] * imgH;
                region.w = data[2] * imgW;
                region.h = data[3] * imgH;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap, size_t thread = 0) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            const Net::Tensor & dst = *net.Dst(thread)[0];
            assert(dst.Count() == 3 && dst.Axis(0) == result.size() && dst.Axis(2) == 6);
            size_t size = dst.Axis(1);
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
            assert(dst[0].Count() == 3 && dst[0].Axis(2) == 6);
            size_t size = dst[0].Axis(1);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* data = dst[0].Data<float>();
                result[b] = GetRegions(data, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        bool _enable;
    };
}


