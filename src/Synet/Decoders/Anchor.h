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
    struct AnchorParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "conformance", "landmarks", "location" }));
        CPL_PARAM_VALUE(Floats, variance, Floats({ 0.1f, 0.2f }));
        CPL_PARAM_VALUE(Shape, step, Shape());
        CPL_PARAM_VALUE(Shape, minSize, Shape());
        CPL_PARAM_VALUE(bool, clip, false);
    };

    class AnchorDecoder
    {
    public: 
        struct Box
        {
            float conf, rect[4], lms[10];
        };
        typedef std::vector<Box> Boxes;

        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        AnchorDecoder()
        {
        }

        bool Init(size_t netW, size_t netH, const AnchorParam& param = AnchorParam())
        {
            _names = param.names();
            _variance = param.variance();
            _prior.clear();
            float inW = float((int)netW);
            float inH = float((int)netH);
            size_t M = param.minSize().size() / param.step().size();
            for (size_t s = 0; s < param.step().size(); ++s)
            {
                size_t step = param.step()[s];
                size_t N = M;
                const size_t* minSize = param.minSize().data() + s * M;
                while (minSize[N - 1] == 0)
                    N--;
                size_t H = (int)ceil(inH / step);
                size_t W = (int)ceil(inW / step);
                size_t offset = _prior.size();
                _prior.resize(offset + H * W * N * 4);
                for (size_t y = 0; y < H; y++)
                {
                    float shiftY = (y + 0.5f) * step / inH;
                    for (size_t x = 0; x < W; x++)
                    {
                        float shiftX = (x + 0.5f) * step / inW;
                        for (size_t n = 0; n < N; n++)
                        {
                            size_t offs = offset + ((y * W + x) * N + n) * 4;
                            _prior[offs + 0] = shiftX;
                            _prior[offs + 1] = shiftY;
                            _prior[offs + 2] = float(minSize[n]) / inW;
                            _prior[offs + 3] = float(minSize[n]) / inH;
                            if (param.clip())
                                for (size_t i = 0; i < 4; ++i)
                                    _prior[offs + i] = RestrictRange(_prior[offs + i], 0.0f, 1.0f);
                        }
                    }
                }
            }
            return true;
        }

        bool Init(const Net& net, const AnchorParam& param = AnchorParam())
        {
            return Init(net.NchwShape()[3], net.NchwShape()[2], param);
        }

        bool Enable() const
        {
            return _prior.size() != 0;
        }

        Boxes GetBoxes(const float* conf, const float* lms, const float* loc, float threshold, float overlap) const
        {
            Boxes boxes;
            const float* var = _variance.data(), * prior = _prior.data();
            for (size_t i = 0, n = _prior.size() / 4; i < n; ++i)
            {
                Box box;
                box.conf = conf[1];
                if (box.conf > threshold)
                {
                    Decode(prior, lms, loc, box);
                    boxes.push_back(box);
                }
                prior += 4;
                conf += 2;
                loc += 4;
                lms += 10;
            }
            Filter(boxes, overlap);
            return boxes;
        }

        Regions GetRegions(const float* conf, const float* lms, const float* loc, size_t srcW, size_t srcH, float threshold, float overlap) const
        {
            Regions regions;
            Boxes boxes = GetBoxes(conf, lms, loc, threshold, overlap);
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                Region region = Convert(boxes[i], (int)srcW, (int)srcH);
                if (region.w && region.h)
                    regions.push_back(region);
            }
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t srcW, size_t srcH, float threshold, float overlap, size_t thread = 0) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* conf = net.Dst(_names[0], thread)->Data<float>(Shp(b, 0, 0));
                const float* lms = net.Dst(_names[1], thread)->Data<float>(Shp(b, 0, 0));
                const float* loc = net.Dst(_names[2], thread)->Data<float>(Shp(b, 0, 0));
                result[b] = GetRegions(conf, lms, loc, srcW, srcH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t srcW, size_t srcH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                const float* conf = GetPtr(dst, _names[0], b);
                const float* lms = GetPtr(dst, _names[1], b);
                const float* loc = GetPtr(dst, _names[2], b);
                if (conf && lms && loc)
                    result[b] = GetRegions(conf, lms, loc, srcW, srcH, threshold, overlap);
            }
            return result;
        }

    private:
        Strings _names;
        Floats _prior, _variance;

        inline Region Convert(const Box& b, int w, int h) const
        {
            Region r;
            r.prob = b.conf;
            float r0 = RestrictRange(b.rect[0], 0.0f, 1.0f);
            float r1 = RestrictRange(b.rect[1], 0.0f, 1.0f);
            float r2 = RestrictRange(b.rect[2], 0.0f, 1.0f);
            float r3 = RestrictRange(b.rect[3], 0.0f, 1.0f);
            r.x = (r0 + r2) / 2.0f * w;
            r.y = (r1 + r3) / 2.0f * h;
            r.w = (Max(r0, r2) - Min(r0, r2)) * w;
            r.h = (Max(r1, r3) - Min(r1, r3)) * h;
            return r;
        }

        inline void Decode(const float* prior, const float* lms, const float* loc, Box & box) const
        {
            float cx = prior[0] + loc[0] * _variance[0] * prior[2];
            float cy = prior[1] + loc[1] * _variance[0] * prior[3];
            float sx = prior[2] * ::exp(loc[2] * _variance[1]);
            float sy = prior[3] * ::exp(loc[3] * _variance[1]);
            box.rect[0] = cx - sx / 2;
            box.rect[1] = cy - sy / 2;
            box.rect[2] = cx + sx / 2;
            box.rect[3] = cy + sy / 2;
            for (size_t l = 0; l < 5; ++l)
            {
                box.lms[l * 2 + 0] = prior[0] + lms[l * 2 + 0] * _variance[0] * prior[2];
                box.lms[l * 2 + 1] = prior[1] + lms[l * 2 + 1] * _variance[0] * prior[3];
            }
        }

        inline float Overlap(const Box& a, const Box& b) const
        {
            float aArea = (a.rect[2] - a.rect[0]) * (a.rect[3] - a.rect[1]);
            float bArea = (b.rect[2] - b.rect[0]) * (b.rect[3] - b.rect[1]);
            float xMin = Max(a.rect[0], b.rect[0]);
            float yMin = Max(a.rect[1], b.rect[1]);
            float xMax = Min(a.rect[2], b.rect[2]);
            float yMax = Min(a.rect[3], b.rect[3]);
            float abInt = Max(0.0f, xMax - xMin) * Max(0.0f, yMax - yMin);
            return abInt / (aArea + bArea - abInt);
        }

        void Filter(Boxes& src, float threshold) const
        {
            Boxes dst;
            dst.reserve(src.size());
            for (size_t s = 0; s < src.size(); ++s)
            {
                bool orig = true;
                for (size_t d = 0; d < dst.size(); ++d)
                {
                    if (Overlap(src[s], dst[d]) > threshold)
                    {
                        if (src[s].conf > dst[d].conf)
                            dst[d] = src[s];
                        orig = false;
                        break;
                    }
                }
                if (orig)
                    dst.push_back(src[s]);
            }
            src.swap(dst);
        }

        SYNET_INLINE const float* GetPtr(const Tensors& dst, const String& name, size_t b) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return dst[d].Data<float>(Shp(b, 0, 0));
            return NULL;
        }
    };

    //---------------------------------------------------------------------------------------------

    AnchorParam GetEpsilonParam()
    {
        AnchorParam epsilon;
        epsilon.names() = Strings({ "conf", "lmks", "loc" });
        epsilon.step() = Shape({ 8, 16, 32, 64 });
        epsilon.minSize() = Shape({ 10, 16, 24, 32, 48, 0, 64, 96, 0, 128, 192, 256 });
        return epsilon;
    }

    AnchorParam GetRetinaParam()
    {
        AnchorParam retina;
        retina.names() = Strings({ "classifications", "ldm_regressions", "bbox_regressions" });
        retina.step() = Shape({ 8, 16, 32 });
        retina.minSize() = Shape({ 10, 20, 32, 64, 128, 256 });
        return retina;
    }
}
