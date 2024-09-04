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
    class YoloDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        YoloDecoder()
            : _enable(false)
        {
        }

        bool Init(size_t netW, size_t netH, const std::vector<YoloParam> & param)
        {
            _netW = netW;
            _netH = netH;
            _output.resize(param.size());
            for (size_t o = 0; o < param.size(); ++o)
            {
                _output[o].name = param[o].name();
                _output[o].num = param[o].num();
                _output[o].total = param[o].total();
                _output[o].classes = param[o].classes();
                _output[o].anchors.resize(param[o].anchors().size());
                for (size_t i = 0; i < param[o].anchors().size(); ++i)
                    _output[o].anchors[i] = param[o].anchors()[i];
                _output[o].mask.resize(param[o].mask().size());
                for (size_t i = 0; i < param[o].mask().size(); ++i)
                    _output[o].mask[i] = param[o].mask()[i];
            }
            _enable = true;
            return true;
        }

        bool Enable() const
        {
            return _enable;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            const Net::Tensor & dst = *net.Dst()[0];
            assert(dst.Count() == 4 && dst.Axis(0) == result.size());
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t o = 0; o < _output.size(); ++o)
                {
                    const Tensor* dst = net.Dst(_output[o].name);
                    if (dst == NULL)
                        dst = net.GetInternalTensor(_output[o].name);
                    if (dst)
                        AppendRegions(_output[o], *dst, b, imgW, imgH, threshold, overlap, result[b]);
                }
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t o = 0; o < _output.size(); ++o)
                {
                    assert(dst[o].Count() == 4 && dst[o].Axis(0) == result.size());
                    const Tensor* pDst = GetTensor(dst, _output[o].name);
                    if (pDst)
                        AppendRegions(_output[o], *pDst, b, imgW, imgH, threshold, overlap, result[b]);
                }
            }
            return result;
        }

    private:
        bool _enable;
        size_t _netW, _netH;
        struct Output
        {
            String name;
            size_t total, num, classes;
            std::vector<float> anchors;
            std::vector<size_t> mask;
        };
        std::vector<Output> _output;

        void AppendRegions(const Output& out, const Tensor& dst, size_t b, size_t imgW, size_t imgH, float threshold, float overlap, Regions& regions) const
        {
            size_t dstH = dst.Axis(2), dstW = dst.Axis(3);
            float kX = float(imgW) / float(dstW);
            float kY = float(imgH) / float(dstH);
            float kW = float(imgW) / float(_netW);
            float kH = float(imgH) / float(_netH);
            for (size_t y = 0; y < dstH; ++y)
            {
                for (size_t x = 0; x < dstW; ++x)
                {
                    for (size_t n = 0; n < out.num; ++n)
                    {
                        float objectness = dst.Data<float>(Shp(b, n * (out.classes + 5) + 4, y, x))[0];
                        if (objectness > threshold)
                        {
                            Region region;
                            region.x = (x + dst.Data<float>(Shp(b, n * (out.classes + 5) + 0, y, x ))[0]) * kX;
                            region.y = (y + dst.Data<float>(Shp(b, n * (out.classes + 5) + 1, y, x ))[0]) * kY;
                            region.w = ::exp(dst.Data<float>(Shp(b, n * (out.classes + 5) + 2, y, x ))[0]) * out.anchors[2 * out.mask[n] + 0] * kW;
                            region.h = ::exp(dst.Data<float>(Shp(b, n * (out.classes + 5) + 3, y, x ))[0]) * out.anchors[2 * out.mask[n] + 1] * kH;
                            for (size_t i = 0; i < out.classes; ++i)
                            {
                                region.id = (int)i;
                                region.prob = objectness * dst.Data<float>(Shp(b, n * (out.classes + 5) + 5 + i, y, x ))[0];
                                if (region.prob > threshold)
                                {
                                    bool insert = true;
                                    for (size_t k = 0; k < regions.size(); ++k)
                                    {
                                        Region& r = regions[k];
                                        if (region.id == regions[k].id && Overlap(region, regions[k]) >= overlap)
                                        {
                                            if (region.prob > regions[k].prob)
                                                regions[k] = region;
                                            insert = false;
                                            break;
                                        }
                                    }
                                    if (insert)
                                        regions.push_back(region);
                                }
                            }
                        }
                    }
                }
            }
        }

        SYNET_INLINE const Tensor* GetTensor(const Tensors& dst, const String& name) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return &dst[d];
            return NULL;
        }
    };
}


