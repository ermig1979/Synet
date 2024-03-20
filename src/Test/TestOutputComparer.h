/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

#include "TestParams.h"
#include "TestOptions.h"
#include "TestRegionDecoder.h"

#include "Synet/Utils/Difference.h"
#include "Synet/Utils/DebugPrint.h"

namespace Test
{
    class OutputComparer
    {
        const Options & _options;
        const TestParam& _param;
        RegionDecoder _regionDecoder;

    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Difference<float> Difference;

        OutputComparer(const Options& options, const TestParam& param, const Shape & src, const Tensors& dst)
            : _options(options)
            , _param(param)
        {
            if (_param.detection().decoder() != "")
            {
                Strings names;
                for (size_t i = 0; i < dst.size(); ++i)
                    names.push_back(dst[i].Name());
                _regionDecoder.Init(src, names, _param);
            }
        }

        bool Compare(const Tensors& first, const Tensors& second, const String & failed) const
        {
            using Synet::Detail::DebugPrint;

            if (first.size() != second.size())
                SYNET_ERROR(failed << std::endl << "Dst count : " << first.size() << " != " << second.size());
            if (_regionDecoder.Enable())
                return CompareRegions(first, second, failed);
            for (size_t d = 0; d < first.size(); ++d)
            {
                String compType = _param.output().size() ? _param.output()[d].compare() : "";
                String decType = _param.detection().decoder();
                if (compType == "0" || compType == "false" || compType == "skip")
                    continue;
                const Tensor& f = first[d];
                const Tensor& s = second[d];

                if (f.Shape() != s.Shape())
                    SYNET_ERROR(failed << std::endl << "Dst[" << d << "] shape : " << DebugPrint(f.Shape()) << " != " << DebugPrint(s.Shape()));
                Difference difference(f, s);
                if (difference.Valid() && 0)
                {
                    if (!difference.Estimate(_options.compareThreshold, _options.compareQuantile))
                    {
                        PrintError(difference, d, failed, std::cout);
                        PrintMaxErrorNeighbours(f, s, difference, 2, std::cout);
                        return false;
                    }
                    continue;
                }
                switch (f.Count())
                {
                case 1:
                    if (!Compare1d(f, s, d, failed))
                        return false;
                    break;
                case 2:
                    if (!Compare2d(f, s, d, failed))
                        return false;
                    break;
                case 3:
                    if (!Compare3d(f, s, d, failed))
                        return false;
                    break;
                case 4:
                    if (!Compare4d(f, s, d, failed))
                        return false;
                    break;
                default:
                    SYNET_ERROR("Error! Dst has unsupported shape " << Synet::Detail::DebugPrint(f.Shape()));
                }
            }
            return true;
        }

        private:
        bool Compare(float a, float b, float t, float& e) const
        {
            float d = ::fabs(a - b);
            e = std::min(d, d / std::max(::fabs(a), ::fabs(b)));
            return e <= t;
        }

        bool Compare(const Tensor& f, const Tensor& s, const Shape& i, size_t d, const String& m) const
        {
            using Synet::Detail::DebugPrint;
            float _f = f.CpuData(i)[0], _s = s.CpuData(i)[0], _t = _options.compareThreshold, _e = 0;
            if (!Compare(_f, _s, _t, _e))
                SYNET_ERROR(m << std::endl << std::fixed << "Dst[" << d << "]" << DebugPrint(f.Shape()) << " at " << DebugPrint(i) << " : " << _f << " != " << _s << " ( " << _e << " > " << _t << " )");
            return true;
        }

        bool Compare1d(const Tensor& f, const Tensor& s, size_t d, const String& failed) const
        {
            for (size_t n = 0; n < f.Axis(0); ++n)
                if (!Compare(f, s, Shp(n), d, failed))
                    return false;
            return true;
        }

        bool Compare2d(const Tensor& f, const Tensor& s, size_t d, const String& failed) const
        {
            using Synet::Detail::DebugPrint;
            const String & compType = _param.output().size() ? _param.output()[d].compare() : "";
            if (compType == "cos_dist" && (_options.bf16 || !_options.comparePrecise))
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                {
                    float cd;
                    SimdCosineDistance32f(f.Data<float>(Shp(n, 0)), s.Data<float>(Shp(n, 0)), f.Axis(1), &cd);
                    if (cd > _options.compareThreshold)
                        SYNET_ERROR(failed << std::endl << std::fixed << "Dst[" << d << "] " << DebugPrint(f.Shape()) << " at " << DebugPrint(Shp(n, 0)) << " : cosine distance " << cd << " > " << _options.compareThreshold);
                }
            }
            else
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        if (!Compare(f, s, Shp(n, c), d, failed))
                            return false;
            }
            return true;
        }

        bool Compare3d(const Tensor& f, const Tensor& s, size_t d, const String& failed) const
        {
            String decType = _param.detection().decoder();
            if (decType == "yoloV8" && (_options.bf16 || !_options.comparePrecise))
            {
                float threshold = _param.detection().confidence();
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t y = 0; y < f.Axis(2); ++y)
                    {
                        float score = f.Data<float>(Shp(n, 4, y))[0];
                        if (score >= threshold)
                        {
                            for (size_t c = 0; c < 5; ++c)
                                if (!Compare(f, s, Shp(n, c, y), d, failed))
                                    return false;
                        }
                    }
            }
            else
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        for (size_t y = 0; y < f.Axis(2); ++y)
                            if (!Compare(f, s, Shp(n, c, y), d, failed))
                                return false;
            }
            return true;
        }

        bool Compare4d(const Tensor& f, const Tensor& s, size_t d, const String& failed) const
        {
            for (size_t n = 0; n < f.Axis(0); ++n)
                for (size_t c = 0; c < f.Axis(1); ++c)
                    for (size_t y = 0; y < f.Axis(2); ++y)
                        for (size_t x = 0; x < f.Axis(3); ++x)
                            if (!Compare(f, s, Shp(n, c, y, x), d, failed))
                                return false;
            return true;
        }

        void PrintError(const Difference& d, size_t i, const String& msg, std::ostream& os) const
        {
            using Synet::Detail::DebugPrint;
            const Difference::Statistics& s = d.GetStatistics();
            const Difference::Specific& e = s.exceed;
            const Difference::Specific& m = s.max;
            os << msg << std::endl << std::fixed;
            os << "Dst[" << i << "]" << DebugPrint(d.GetShape()) << " at ";
            os << DebugPrint(e.index) << " : diff = " << e.diff << " (" << e.first << " != " << e.second << ")";
            os << ", num = " << double(e.count) / s.count << "(" << e.count << ")";
            os << ", avg = " << s.mean << ", std = " << s.sdev << ", abs = " << s.adev;
            os << ", max " << DebugPrint(m.index) << " diff = " << s.max.diff << " (" << m.first << " != " << m.second << ")" << std::endl;
        }

        void PrintNeighbours(const Tensor& t, const Shape& m, int n, std::ostream& os) const
        {
            for (int y = Synet::Max(0, (int)m[2] - n), h = (int)Synet::Min(t.Axis(2), m[2] + n + 1); y < h; y++)
            {
                for (int x = Synet::Max(0, (int)m[3] - n), w = (int)Synet::Min(t.Axis(3), m[3] + n + 1); x < w; x++)
                    os << ExpandLeft(ToString(t.CpuData(Shp(m[0], m[1], y, x))[0], 3), 8) << " ";
                os << std::endl;
            }
            os << std::endl;
        }

        void PrintMaxErrorNeighbours(const Tensor& f, const Tensor& s, const Difference& d, int n, std::ostream& os) const
        {
            if (f.Shape() != s.Shape() || f.Count() != 4)
                return;
            const Shape& m = d.GetStatistics().max.index;
            PrintNeighbours(f, m, n, os);
            PrintNeighbours(s, m, n, os);
        }

        bool CompareRegions(const Tensors& f, const Tensors& s, const String& failed) const
        {
            Regions rf = _regionDecoder.GetRegions(f, Size(1, 1), _param.detection().confidence(), _options.regionOverlap);
            Regions rs = _regionDecoder.GetRegions(s, Size(1, 1), _param.detection().confidence(), _options.regionOverlap);
            SortRegionsByProb(rf);
            SortRegionsByProb(rs);
            size_t n = std::min(rf.size(), rs.size());
            //std::cout << "CompareRegions: " << rf.size() << " and " << rs.size() << std::endl;
            for (size_t fi = 0; fi < n; ++fi)
            {
                const Region& _f = rf[fi];
                size_t indexMax = 0;
                float overlapMax = 0;
                for (size_t si = 0; si < n; ++si)
                {
                    float overlap = Synet::Overlap(_f, rs[si]);
                    if (_f.id == rs[si].id && overlap > overlapMax )
                    {
                        indexMax = si;
                        overlapMax = overlap;
                    }
                }
                const Region& _s = rs[indexMax];
                //std::cout << "CompareRegions: " << ToStr(_f) << " and " << ToStr(_s) << std::endl;
                float d = std::max(std::max(::abs(_f.x - _s.x), ::abs(_f.y - _s.y)), std::max(abs(_f.w - _s.w), ::abs(_f.h - _s.h)));
                if (_f.id != _s.id || d > _options.compareThreshold)
                {
                    SYNET_ERROR(failed << std::endl << " Region[" << fi << "] from " << n << " : " << ToStr(_f) << " != " << ToStr(_s) << " : difference " << d << " > " << _options.compareThreshold);
                }
            }
            return true;
        }

        void SortRegionsByProb(Regions& regions) const
        {
            const float eps = 0.01;
            std::sort(regions.begin(), regions.end(), [eps](const Region& r1, const Region& r2) 
                {
                    return r1.prob > r2.prob;
                });
        }
    };
}

