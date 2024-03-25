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
    struct ScrfdParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" }));
        CPL_PARAM_VALUE(Floats, variance, Floats({ 0.1f, 0.2f }));
        CPL_PARAM_VALUE(Shape, step, Shape({ 8, 16, 32 }));
        CPL_PARAM_VALUE(Shape, minSize, Shape({ 1, 2, 0, 1, 2, 0, 1, 2, 0 }));
        CPL_PARAM_VALUE(bool, clip, false);
    };

    class ScrfdDecoder
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

        ScrfdDecoder()
        {
        }

        bool Init(size_t netW, size_t netH, const ScrfdParam & param = ScrfdParam())
        {
            _names = param.names();
            _variance = param.variance();
            _prior.clear();
            float inW = float((int)netW);
            float inH = float((int)netH);
            size_t M = param.minSize().size() / param.step().size();
            _prior.resize(param.step().size());
            for (size_t s = 0; s < param.step().size(); ++s)
            {
                size_t step = param.step()[s];
                size_t N = M;
                const size_t* minSize = param.minSize().data() + s * M;
                while (minSize[N - 1] == 0)
                    N--;
                size_t H = (int)ceil(inH / step);
                size_t W = (int)ceil(inW / step);
                _prior[s].resize(H * W * N * 4);
                float* prior = _prior[s].data();
                for (size_t y = 0; y < H; y++)
                {
                    float shiftY = (y + 0.5f) * step / inH;
                    for (size_t x = 0; x < W; x++)
                    {
                        float shiftX = (x + 0.5f) * step / inW;
                        for (size_t n = 0; n < N; n++)
                        {
                            prior[0] = shiftX;
                            prior[1] = shiftY;
                            prior[2] = float(minSize[n]) / inW;
                            prior[3] = float(minSize[n]) / inH;
                            if (param.clip())
                                for (size_t i = 0; i < 4; ++i)
                                    prior[i] = RestrictRange(prior[i], 0.0f, 1.0f);
                            prior += 4;
                        }
                    }
                }
            }
            return true;
        }

        bool Enable() const
        {
            return _prior.size() != 0;
        }

        Boxes GetBoxes(const float *prior, const float* conf, const float* lms, const float* loc, size_t size, float threshold, float overlap) const
        {
            Boxes boxes;
            for (size_t i = 0; i < size; ++i)
            {
                Box box;
                box.conf = conf[0];
                if (box.conf > threshold)
                {
                    Decode(prior, lms, loc, box);
                    boxes.push_back(box);
                }
                prior += 4;
                conf += 1;
                loc += 4;
                lms += 10;
            }
            Filter(boxes, overlap);
            return boxes;
        }

        void GetRegions(const float* prior, const float* conf, const float* lms, const float* loc, size_t size, size_t imgW, size_t imgH, float threshold, float overlap, Regions & regions) const
        {
            Boxes boxes = GetBoxes(prior, conf, lms, loc, size, threshold, overlap);
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                Region region = Convert(boxes[i], (int)imgW, (int)imgH);
                if (region.w && region.h)
                    regions.push_back(region);
            }
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t p = 0, n = _prior.size(); p < n; ++p)
                {
                    const float* conf = GetPtr(net, _names[0 * n + p], b);
                    const float* loc = GetPtr(net, _names[1 * n + p], b);
                    const float* lms = GetPtr(net, _names[2 * n + p], b);
                    if (conf && lms && loc)
                        GetRegions(_prior[p].data(), conf, lms, loc, _prior[p].size() / 4, imgW, imgH, threshold, overlap, result[b]);
                }
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t p = 0, n = _prior.size(); p < n; ++p)
                {
                    const float* conf = GetPtr(dst, _names[0 * n + p], b);
                    const float* loc = GetPtr(dst, _names[1 * n + p], b);
                    const float* lms = GetPtr(dst, _names[2 * n + p], b);
                    if (conf && lms && loc)
                        GetRegions(_prior[p].data(), conf, lms, loc, _prior[p].size() / 4, imgW, imgH, threshold, overlap, result[b]);
                }
            }
            return result;
        }

    private:
        Strings _names;
        std::vector<Floats> _prior;
        Floats _variance;

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

        inline void Decode(const float* prior, const float* lms, const float* loc, Box& box) const
        {
            float cx = prior[0] + loc[0] * _variance[0] * prior[2];
            float cy = prior[1] + loc[1] * _variance[0] * prior[3];
            float sx = prior[2] * ::expf(loc[2] * _variance[1]);
            float sy = prior[3] * ::expf(loc[3] * _variance[1]);
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

        SYNET_INLINE const float* GetPtr(const Net& net, const String& name, size_t b) const
        {
            const Tensor* dst = net.Dst(name);
            if(dst)
                return dst->Data<float>(Shp(b, 0, 0));
            return NULL;
        }
    };
}


