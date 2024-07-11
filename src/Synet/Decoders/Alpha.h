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
    struct AlphaParam
    {
        CPL_PARAM_VALUE(Strings, names, Strings({ "broadcast_div0", "conv8_3_bbox", "broadcast_div1", "conv10_3_bbox", "broadcast_div2", "conv13_3_bbox", "broadcast_div3", "conv15_3_bbox", 
            "broadcast_div4", "conv18_3_bbox", "broadcast_div5", "conv21_3_bbox", "broadcast_div6", "conv23_3_bbox", "broadcast_div7", "conv25_3_bbox" }));
        CPL_PARAM_VALUE(Floats, rfSize, Floats( {7.5, 10.0, 20.0, 35.0, 55.0, 125.0, 200.0, 280.0}));
        CPL_PARAM_VALUE(Ints, rfStride, Ints({ 4, 4, 8, 8, 16, 32, 32, 32 }));
        CPL_PARAM_VALUE(Ints, isDivide, Ints({ 0, 0, 1, 0, 1, 1, 0, 0 }));
    };

    class AlphaDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        AlphaDecoder()
            : _netW(0)
            , _netH(0)
        {
        }

        bool Init(size_t netW, size_t netH, const AlphaParam& param = AlphaParam())
        {
            _netW = netW;
            _netH = netH;
            _rfSize = param.rfSize();
            _count = _rfSize.size();
            if (param.rfStride().size() != _count || param.rfStride().size() != _count || param.names().size() != _count * 2)
                SYNET_ERROR("Check alpha decoder parameters!");
            _names = param.names();
            _rfStride = param.rfStride();
            _isDivide = param.isDivide();
            _rfW.resize(_count);
            _rfH.resize(_count);

            size_t currW = (_netW - 1) / 4, currH = (_netH - 1) / 4;
            for (size_t i = 0; i < _count; ++i)
            {
                if (_isDivide[i]) 
                {
                    currW = (currW - 1) / 2;
                    currH = (currH - 1) / 2;
                }
                _rfW[i] = (int)currW;
                _rfH[i] = (int)currH;
            }

            return _netW && _netH;
        }

        bool Enable() const
        {
            return _netW && _netH;
        }

        Regions GetRegions(const float *const* ptrs, TensorFormat format, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            float kX = float(imgW) / float(_netW);
            float kY = float(imgH) / float(_netH);
            Regions regions;
            for (size_t i = 0; i < _count; ++i)
            {
                size_t rfH = _rfH[i], rfW = _rfW[i], stride = _rfStride[i], size = rfH * rfW;
                float rfSize = _rfSize[i];
                const float *scores = ptrs[i * 2 + 0];
                const float* bboxes = ptrs[i * 2 + 1];
                for (size_t h = 0; h < rfH; ++h)
                {
                    for (size_t w = 0; w < rfW; ++w)
                    {
                        size_t offs = rfW * h + w;
                        float score, bbox0, bbox1, bbox2, bbox3;
                        if (format == TensorFormatNchw)
                        {
                            score = scores[offs + 0 * size];
                            bbox0 = bboxes[offs + 0 * size];
                            bbox1 = bboxes[offs + 1 * size];
                            bbox2 = bboxes[offs + 2 * size];
                            bbox3 = bboxes[offs + 3 * size];
                        }
                        else
                        {
                            score = scores[offs * 2 + 0];
                            bbox0 = bboxes[offs * 4 + 0];
                            bbox1 = bboxes[offs * 4 + 1];
                            bbox2 = bboxes[offs * 4 + 2];
                            bbox3 = bboxes[offs * 4 + 3];
                        }
                        if (score > threshold)
                        {                
                            int left = int(stride * (w + 1) - 1 - bbox0 * rfSize);
                            int top = int(stride * (h + 1) - 1 - bbox1 * rfSize);
                            int right = int(stride * (w + 1) - 1 - bbox2 * rfSize);
                            int bottom = int(stride * (h + 1) - 1 - bbox3 * rfSize);
                            if (left < 0)
                            {
                                right -= left;
                                left = 0;
                            }
                            if (top < 0)
                            {
                                bottom -= top;
                                top = 0;
                            }
                            if (right > imgW)
                            {
                                left -= right - imgW;
                                right = imgW;
                            }
                            if (bottom > imgH)
                            {
                                top -= bottom - imgH;
                                bottom = imgH;
                            }
                            Region region;
                            region.prob = score;
                            region.x = (left + right) * 0.5f * kX;
                            region.y = (top + bottom) * 0.5f * kY;
                            region.w = (right - left) * kX;
                            region.h = (bottom - top) * kY;
                            regions.push_back(region);
                        }
                    }
                }
            }
            Synet::Filter(regions, overlap);
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(net.NchwShape()[0]);
            assert(net.Dst().size() == _count * 2);
            bool nhwc = net.Format() == TensorFormatNhwc;
            size_t C = nhwc ? 3 : 1, H = nhwc ? 1 : 2, W = nhwc ? 2 : 3;
            for (size_t i = 0; i < _count; ++i)
            {
                const Tensor& score = *net.Dst()[i * 2 + 0];
                assert(score.Count() == 4 && score.Axis(C) == 2 && score.Axis(H) == _rfH[i] && score.Axis(W) == _rfW[i]);
                const Tensor& bbox = *net.Dst()[i * 2 + 1];
                assert(bbox.Count() == 4 && bbox.Axis(C) == 4 && bbox.Axis(H) == _rfH[i] && bbox.Axis(W) == _rfW[i]);
            }
            FloatPtrs ptrs(_count * 2, NULL);
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t i = 0; i < _count * 2; ++i)
                    ptrs[i] = (float*)net.Dst(_names[i])->Data<float>(Shp(b, 0, 0, 0));
                result[b] = GetRegions(ptrs.data(), net.Format(), imgW, imgH, threshold, overlap);
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            std::vector<Regions> result(dst[0].Axis(0));
            assert(dst.size() == _count * 2);
            for (size_t i = 0; i < _count; ++i)
            {
                const Tensor& score = dst[i * 2 + 0];
                assert(score.Count() == 4 && score.Axis(1) == 2 && score.Axis(2) == _rfH[i] && score.Axis(3) == _rfW[i]);
                const Tensor& bbox = dst[i * 2 + 1];
                assert(bbox.Count() == 4 && bbox.Axis(1) == 4 && bbox.Axis(2) == _rfH[i] && bbox.Axis(3) == _rfW[i]);
            }
            FloatPtrs ptrs(_count * 2, NULL);
            for (size_t b = 0; b < result.size(); ++b)
            {
                for (size_t i = 0; i < _count * 2; ++i)
                    ptrs[i] = (float*)GetTensor(dst, _names[i])->Data<float>(Shp(b, 0, 0, 0));
                result[b] = GetRegions(ptrs.data(), TensorFormatNchw, imgW, imgH, threshold, overlap);
            }
            return result;
        }

    private:
        Strings _names;
        Ints _rfW, _rfH, _rfStride, _isDivide;
        Floats _rfSize;
        size_t _netW, _netH, _count;

        SYNET_INLINE const Tensor* GetTensor(const Tensors& dst, const String& name) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return dst.data() + d;
            return NULL;
        }
    };
}


