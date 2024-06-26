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

#include <stack>

namespace Synet
{
    struct IimParam
    {
        CPL_PARAM_VALUE(String, name, String("binary_map"));
        CPL_PARAM_VALUE(float, threshold, 0.5f);
        CPL_PARAM_VALUE(int, minArea, 3);
    };

    class IimDecoder
    {
    public: 
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        IimDecoder()
            : _netW(0)
            , _netH(0)
        {
        }

        bool Init(size_t netW, size_t netH, const IimParam& param = IimParam())
        {
            _netW = netW;
            _netH = netH;
            _name = param.name();
            _threshold = param.threshold();
            _minArea = param.minArea();
            _neighbours[0] = Point(-1, 0);
            _neighbours[1] = Point(0, -1);
            _neighbours[2] = Point(1, 0);
            _neighbours[3] = Point(0, 1);
            return true;
        }

        bool Enable() const
        {
            return !_name.empty();
        }

        const String & Name() const
        {
            return _name;
        }

        Regions GetRegions(const float* bin, size_t imgW, size_t imgH) const
        {
            //SYNET_PERF_FUNC();
            View mask(_netW, _netH, View::Int32);
            InitMask(bin, mask);

            float kX = float(imgW) / float(_netW);
            float kY = float(imgH) / float(_netH);
            Regions regions;
            int index = Start;
            for (ptrdiff_t y = 1; y < (ptrdiff_t)mask.height - 1; ++y)
            {
                for (ptrdiff_t x = 1; x < (ptrdiff_t)mask.width - 1; ++x)
                {
                    int area = 0;
                    if (mask.At<int>(x, y) == Seed)
                    {
                        std::stack<Point> stack;
                        stack.push(Point(x, y));
                        Rect rect;
                        while (!stack.empty())
                        {
                            Point current = stack.top();
                            stack.pop();
                            mask.At<int>(current) = index;
                            rect |= current;
                            area++;
                            for (size_t n = 0; n < 4; ++n)
                            {
                                Point neighbour = current + _neighbours[n];
                                if (mask.At<int>(neighbour) == Seed)
                                    stack.push(neighbour);
                            }
                        }
                        if (area >= _minArea)
                        {
                            Region region;
                            region.prob = 1.0f;
                            region.id = 1;
                            region.x = rect.Center().x * kX;
                            region.y = rect.Center().y * kY;
                            region.w = rect.Width() * kX;
                            region.h = rect.Height() * kY;
                            regions.push_back(region);
                        }
                        index++;
                    }
                }
            }
            return regions;
        }

        std::vector<Regions> GetRegions(const Net& net, size_t imgW, size_t imgH) const
        {
            const Tensor* dst = net.Dst(_name);
            const float* bin = dst->Data<float>();
            std::vector<Regions> result(dst->Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                result[b] = GetRegions(bin, imgW, imgH);
                bin += _netH * _netW;
            }
            return result;
        }

        std::vector<Regions> GetRegions(const Tensors& dst, size_t imgW, size_t imgH) const
        {
            const float* bin = GetPtr(dst, _name, 0);
            std::vector<Regions> result(dst[0].Axis(0));
            for (size_t b = 0; b < result.size(); ++b)
            {
                result[b] = GetRegions(bin, imgW, imgH);
                bin += _netH * _netW;
            }
            return result;
        }

    private:
        typedef Simd::Rectangle<ptrdiff_t> Rect;
        typedef Simd::Point<ptrdiff_t> Point;
        enum
        {
            Zero,
            Seed,
            Start,
        };

        String _name;
        float _threshold;
        int _minArea;
        Point _neighbours[4];
        size_t _netW, _netH;

        void InitMask(const float* bin, View & mask) const
        {
            for (size_t y = 0; y < _netH; ++y)
            {
                for (size_t x = 0; x < _netW; ++x, bin++)
                {
                    mask.At<int>(x, y) = Zero;
                    if (bin[0] > _threshold)
                        mask.At<int>(x, y) = Seed;
                }
            }
            Simd::FillFrame(mask, Rect(1, 1, mask.width - 1, mask.height - 1), Zero);
        }

        SYNET_INLINE const float* GetPtr(const Tensors& dst, const String& name, size_t b) const
        {
            for (size_t d = 0; d < dst.size(); d++)
                if (dst[d].Name() == name)
                    return dst[d].Data<float>(Shp(b, 0, 0, 0));
            return NULL;
        }
    };
}


