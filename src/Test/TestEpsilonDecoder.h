/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

#include "TestNetwork.h"

namespace Test
{
    class EpsilonDecoder
    {
    public: 
        struct Box
        {
            float conf, rect[4], lms[10];
        };
        typedef std::vector<Box> Boxes;

        EpsilonDecoder()
        {
        }

        bool Init(const Size & size, const EpsilonParam& epsilon)
        {
            if (!epsilon.enable())
                return false;
            _variance = epsilon.variance();
            _prior.clear();
            float inW = float((int)size.x);
            float inH = float((int)size.y);
            size_t M = epsilon.minSize().size() / epsilon.step().size();
            for (size_t s = 0; s < epsilon.step().size(); ++s)
            {
                size_t step = epsilon.step()[s];
                size_t N = M;
                const size_t* minSize = epsilon.minSize().data() + s * M;
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
                            if (epsilon.clip())
                                for (size_t i = 0; i < 4; ++i)
                                    _prior[offs + i] = Synet::RestrictRange(_prior[offs + i], 0.0f, 1.0f);
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

        Regions GetRegions(const float* conf, const float* lms, const float* loc, const Size& size, float threshold, float overlap) const
        {
            Regions regions;
            Boxes boxes = GetBoxes(conf, lms, loc, threshold, overlap);
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                Region region = Convert(boxes[i], (int)size.x, (int)size.y);
                if (region.w && region.h)
                    regions.push_back(region);
            }
            return regions;
        }

    private:
        Floats _prior, _variance;

        inline Region Convert(const Box& b, int w, int h) const
        {
            Region r;
            r.prob = b.conf;
            float r0 = Synet::RestrictRange(b.rect[0], 0.0f, 1.0f);
            float r1 = Synet::RestrictRange(b.rect[1], 0.0f, 1.0f);
            float r2 = Synet::RestrictRange(b.rect[2], 0.0f, 1.0f);
            float r3 = Synet::RestrictRange(b.rect[3], 0.0f, 1.0f);
            r.x = Synet::Round((r0 + r2) / 2.0f * w);
            r.y = Synet::Round((r1 + r3) / 2.0f * h);
            r.w = Synet::Round((Synet::Max(r0, r2) - Synet::Min(r0, r2))*w);
            r.h = Synet::Round((Synet::Max(r1, r3) - Synet::Min(r1, r3))*h);
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
            float xMin = Synet::Max(a.rect[0], b.rect[0]);
            float yMin = Synet::Max(a.rect[1], b.rect[1]);
            float xMax = Synet::Min(a.rect[2], b.rect[2]);
            float yMax = Synet::Min(a.rect[3], b.rect[3]);
            float abInt = Synet::Max(0.0f, xMax - xMin) * Synet::Max(0.0f, yMax - yMin);
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

    };
}


