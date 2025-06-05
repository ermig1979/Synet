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
    struct ScrfdV2Param
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "bboxes", "scores" }));
    };

    class ScrfdV2Decoder
    {
    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        ScrfdV2Decoder()
        {
        }

        bool Init(size_t netW, size_t netH, const ScrfdV2Param& param = ScrfdV2Param())
        {
            _netW = netW;
            _netH = netH;
            _names = param.names();
            return true;
        }

        bool Enable() const
        {
            return _names.size();
        }

        Regions GetRegions(const float* bboxes, const float* scores, size_t size, size_t  imgW, size_t  imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(_netW), kY = float(imgH) / float(_netH);
            Regions regions;
            for (size_t i = 0; i < size; ++i)
            {
                float score = scores[i];
                if (score < threshold)
                    continue;
                float x1 = Min(bboxes[4 * i], bboxes[4 * i + 2]) * kX;
                float x2 = Max(bboxes[4 * i], bboxes[4 * i + 2]) * kX;
                float y1 = Min(bboxes[4 * i + 1], bboxes[4 * i + 3]) * kY;
                float y2 = Max(bboxes[4 * i + 1], bboxes[4 * i + 3]) * kY;
                Region region;
                region.prob = score;
                region.id = 0;
                region.x = (x1 + x2) * 0.5f;
                region.y = (y1 + y2) * 0.5f;
                region.w = x2 - x1;
                region.h = y2 - y1;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t  imgW, size_t  imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* bboxes = net.Dst(_names[0])->Data<float>(Shp(b, 0, 0));
                const float* scores = net.Dst(_names[1])->Data<float>(Shp(b, 0, 0));
                size_t size = net.Dst(_names[0])->Size(1, 2);
                result[b] = GetRegions(bboxes, scores, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t  imgW, size_t  imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* bboxes = GetPtr(dst, _names[0], b);
                const float* scores = GetPtr(dst, _names[1], b);
                if (bboxes && scores)
                {
                    size_t size = dst[0].Size(1, 2);
                    result[b] = GetRegions(bboxes, scores, size, imgW, imgH, threshold, overlap);
                }
            }
            return result;
        }

    private:
        Strings _names;
        size_t _netW, _netH;

        SYNET_INLINE const float* GetPtr(const Tensors& dst, const String& name, size_t b) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return dst[d].Data<float>(Shp(b, 0, 0));
            return NULL;
        }
    };
}


