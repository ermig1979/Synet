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
    struct ScrfdParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32" }));
        CPL_PARAM_VALUE(Shape, step, Shape({ 8, 16, 32 }));
        CPL_PARAM_VALUE(Shape, minSize, Shape({ 1, 2, 1, 2, 1, 2 }));
    };

    class ScrfdDecoder
    {
    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        ScrfdDecoder()
        {
        }

        bool Init(size_t netW, size_t netH, const ScrfdParam & param = ScrfdParam())
        {
            _netW = netW;
            _netH = netH;
            _names = param.names();
            _step = param.step();
            _size.resize(_step.size());
            _dstW.resize(_step.size());
            _dstH.resize(_step.size());
            size_t M = param.minSize().size() / _step.size();
            for (size_t s = 0; s < _step.size(); ++s)
            {
                size_t N = M;
                const size_t* minSize = param.minSize().data() + s * M;
                while (minSize[N - 1] == 0)
                    N--;
                _size[s] = N;
                _dstW[s] = _netW / _step[s];
                _dstH[s] = _netH / _step[s];
            }
            return true;
        }

        bool Enable() const
        {
            return _step.size() != 0;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t s = 0, n = _step.size(); s < n; ++s)
                {
                    const float* score = GetPtr(net, _names[0 * n + s], b);
                    const float* bbox = GetPtr(net, _names[1 * n + s], b);
                    if (score && bbox)
                        GetRegions(score, bbox, s, imgW, imgH, threshold, overlap, result[b]);
                }
                Synet::Filter(result[b], overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t s = 0, n = _step.size(); s < n; ++s)
                {
                    const float* score = GetPtr(dst, _names[0 * n + s], b);
                    const float* bbox = GetPtr(dst, _names[1 * n + s], b);
                    if (score && bbox)
                        GetRegions(score, bbox, s, imgW, imgH, threshold, overlap, result[b]);
                }
                Synet::Filter(result[b], overlap);
            }
            return result;
        }

    private:
        Strings _names;
        size_t _netW, _netH;
        Shape _step, _size, _dstW, _dstH;

        SYNET_INLINE const float* GetPtr(const Tensors& dst, const String& name, size_t b) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return dst[d].Data<float>(Shp(b, 0, 0));
            return NULL;
        }

        SYNET_INLINE const float* GetPtr(const Net& net, const String& name, size_t b) const
        {
            const Tensor* dst = net.Dst(name);
            if(dst)
                return dst->Data<float>(Shp(b, 0, 0));
            return NULL;
        }

        void GetRegions(const float* score, const float* bbox, size_t s, size_t imgW, size_t imgH, float threshold, float overlap, Regions& regions) const
        {
            size_t size = _size[s], dstW = _dstW[s], dstH = _dstH[s];
            float step = float(_step[s]), kX = float(imgW) / float(_netW), kY = float(imgH) / float(_netH);
            for (size_t y = 0; y < dstH; ++y)
            {
                for (size_t x = 0; x < dstW; ++x)
                {
                    for (size_t i = 0; i < size; ++i)
                    {
                        if (score[0] > threshold)
                        {
                            float xy[4];
                            DecodeXY(x * step, y * step, bbox, step, xy);

                            Synet::Region<float> region;
                            region.x = (xy[0] + xy[2]) * 0.5f * kX;
                            region.y = (xy[1] + xy[3]) * 0.5f * kY;
                            region.w = fabs(xy[2] - xy[0]) * kX;
                            region.h = fabs(xy[3] - xy[1]) * kY;
                            region.id = 0;
                            region.prob = score[0];
                            regions.push_back(region);
                        }
                        score += 1;
                        bbox += 4;
                    }
                }
            }
        }

        SYNET_INLINE void DecodeXY(float x, float y, const float* bbox, float step, float* xy) const
        {
            xy[0] = RestrictRange<float>(x - bbox[0] * step, 0, _netW);
            xy[1] = RestrictRange<float>(y - bbox[1] * step, 0, _netH);
            xy[2] = RestrictRange<float>(x + bbox[2] * step, 0, _netW);
            xy[3] = RestrictRange<float>(y + bbox[3] * step, 0, _netH);
        }
    };
}


