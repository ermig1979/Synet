/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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
    struct SsdParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "labels", "scores", "bboxes" }));
    };

    class SsdDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        SsdDecoder()
        {
        }

        bool Init(const SsdParam & param = SsdParam())
        {
            _names = param.names();
            return true;
        }

        bool Enable() const
        {
            return _names.size() > 0;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap, size_t thread = 0) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            const Net::Tensor & dst = *net.Dst(thread)[0];
            size_t size = dst.Axis(1);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const int64_t* labels = net.Dst(_names[0], thread)->Data<int64_t>(Shp(b, 0));
                const float* scores = net.Dst(_names[1], thread)->Data<float>(Shp(b, 0));
                const float* boxes = net.Dst(_names[2], thread)->Data<float>(Shp(b, 0, 0));
                if (labels && scores && boxes)
                    result[b] = GetRegions(labels, scores, boxes, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            size_t size = dst[0].Axis(1);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* labels = GetPtr(dst, _names[0], b);
                const float* scores = GetPtr(dst, _names[1], b);
                const float* boxes = GetPtr(dst, _names[2], b);
                if(labels && scores && boxes)
                    result[b] = GetRegions(labels, scores, boxes, size, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        Strings _names;

        template<class T> Regions GetRegions(const T* labels, const float* scores, const float* boxes, size_t size, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            Regions regions;
            for (size_t i = 0; i < size; ++i, boxes += 4) 
            {
                if (scores[i] < threshold)
                    continue;
                Region region;
                region.id = (int)labels[i];
                region.prob = scores[i];
                region.x = (boxes[2] + boxes[0]) * 0.5f * imgW;
                region.y = (boxes[3] + boxes[1]) * 0.5f * imgH;
                region.w = (boxes[2] - boxes[0]) * imgW;
                region.h = (boxes[3] - boxes[1]) * imgH;
                regions.push_back(region);
            }
            Synet::Filter(regions, overlap);
            return regions;
        }        
        
        SYNET_INLINE const float* GetPtr(const Tensors& dst, const String& name, size_t b) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name && dst[d].Count() == 3)
                    return dst[d].Data<float>(Shp(b, 0, 0));
                else if (dst[d].Name() == name && dst[d].Count() == 2)
                    return dst[d].Data<float>(Shp(b, 0));
            return NULL;
        }
    };
}


