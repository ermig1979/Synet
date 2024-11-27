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
#include "Synet/Utils/Activation.h"

namespace Synet
{
    class RegionDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        RegionDecoder()
            : _enable(false)
        {
        }

        bool Init(const RegionParam & param)
        {
            _coords = param.coords();
            _classes = param.classes();
            _num = param.num();
            _softmax = param.softmax();
            _anchors.resize(param.anchors().size());
            for (size_t i = 0; i < param.anchors().size(); ++i)
                _anchors[i] = param.anchors()[i];
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
            size_t dstW = dst.Axis(3);
            size_t dstH = dst.Axis(2);
            assert(dst.Count() == 4 && dst.Axis(0) == result.size());
            for (size_t b = 0; b < result.size(); ++b)
                GetRegions(dst.Data<float>(Shp(b, 0, 0, 0)), dstW, dstH, imgW, imgH, threshold, overlap, result[b]);
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            size_t dstW = dst[0].Axis(3);
            size_t dstH = dst[0].Axis(2);
            assert(dst[0].Count() == 4 && dst[0].Axis(0) == result.size());
            for (size_t b = 0; b < result.size(); ++b)
                GetRegions(dst[0].Data<float>(Shp(b, 0, 0, 0)), dstW, dstH, imgW, imgH, threshold, overlap, result[b]);
            return result;
        }

    private:
        bool _enable, _softmax;
        size_t _coords, _classes, _num;
        Floats _anchors;

        void GetRegions(const float * dst, size_t dstW, size_t dstH, size_t imgW, size_t imgH, float threshold, float overlap, Regions& regions) const
        {
            regions.clear();
            float kY = float(imgH) / float(dstH);
            float kX = float(imgW) / float(dstW);
            for (size_t row = 0, i = 0; row < dstH; ++row)
            {
                for (size_t col = 0; col < dstW; ++col, ++i)
                {
                    for (size_t n = 0; n < _num; ++n)
                    {
                        size_t index = i * _num + n;
                        size_t predictIndex = index * (_classes + 5) + 4;
                        float scale = dst[predictIndex];
                        size_t regionIndex = index * (_classes + 5);
                        Region r;
                        r.x = (col + CpuSigmoid(dst[regionIndex + 0])) * kX;
                        r.y = (row + CpuSigmoid(dst[regionIndex + 1])) * kY;
                        r.w = ::exp(dst[regionIndex + 2]) * _anchors[2 * n] * kX;
                        r.h = ::exp(dst[regionIndex + 3]) * _anchors[2 * n + 1] * kY;
                        size_t classIndex = index * (_classes + 5) + 5;
                        for (size_t id = 0; id < _classes; ++id)
                        {
                            float prob = scale * dst[classIndex + id];
                            if (prob > threshold)
                            {
                                r.prob = prob;
                                r.id = (int)id;
                                regions.push_back(r);
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


