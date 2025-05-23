/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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
            if (_param.output().size() != first.size() && _param.output().size() != 0)
                SYNET_ERROR(failed << std::endl << "Check output parameter size!" << _param.output().size() << " != " << first.size() << " !");
            if (_regionDecoder.Enable())
                return CompareRegions(first, second, failed);
            for (size_t d = 0; d < first.size(); ++d)
            {
                String compType = _param.output().size() ? _param.output()[d].compare() : "";
                String decType = _param.detection().decoder();
                if (compType == "0" || compType == "false" || compType == "skip")
                    continue;
                float compareThreshold = _options.compareThreshold;
                if (_options.bf16)
                {
                    if (_param.output().size() && _param.output()[d].bf16Threshold() != 0.0f)
                        compareThreshold = _param.output()[d].bf16Threshold();
                }
                else
                {
                    if (_param.output().size() && _param.output()[d].fp32Threshold() != 0.0f)
                        compareThreshold = _param.output()[d].fp32Threshold();
                }
                const Tensor& f = first[d];
                const Tensor& s = second[d];

                if (f.Shape() != s.Shape())
                    SYNET_ERROR(failed << std::endl << "Dst[" << d << "] " << s.Name() << " shape : " << DebugPrint(f.Shape()) << " != " << DebugPrint(s.Shape()));
                Difference difference(f, s);
                if (difference.Valid() && 0)
                {
                    if (!difference.Estimate(compareThreshold, _options.compareQuantile))
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
                    if (!Compare1d(f, s, d, failed, compType, compareThreshold))
                        return false;
                    break;
                case 2:
                    if (!Compare2d(f, s, d, failed, compType, compareThreshold))
                        return false;
                    break;
                case 3:
                    if (!Compare3d(f, s, d, failed, compType, compareThreshold))
                        return false;
                    break;
                case 4:
                    if (!Compare4d(f, s, d, failed, compType, compareThreshold))
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

        bool Compare(const Tensor& f, const Tensor& s, const Shape& i, size_t d, const String& m, const String & c, float t) const
        {
            using Synet::Detail::DebugPrint;
            float _f = f.Data<float>(i)[0], _s = s.Data<float>(i)[0], _e = 0;
            if (!Compare(_f, _s, t, _e))
                SYNET_ERROR(m << std::endl << std::fixed << "Dst[" << d << "] " << s.Name() << " " << DebugPrint(f.Shape()) << " at " << DebugPrint(i) << " : " << _f << " != " << _s << " ( " << _e << " > " << t << " ) " << c);
            return true;
        }

        bool Compare1d(const Tensor& f, const Tensor& s, size_t d, const String& failed, String compType, float compareThreshold) const
        {
            for (size_t n = 0; n < f.Axis(0); ++n)
                if (!Compare(f, s, Shp(n), d, failed, "", compareThreshold))
                    return false;
            return true;
        }

        bool Compare2d(const Tensor& f, const Tensor& s, size_t d, const String& failed, String compType, float compareThreshold) const
        {
            using Synet::Detail::DebugPrint;
            if (compType == "cos_dist" && (_options.bf16 || !_options.comparePrecise))
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                {
                    float cd;
                    SimdCosineDistance32f(f.Data<float>(Shp(n, 0)), s.Data<float>(Shp(n, 0)), f.Axis(1), &cd);
                    if (cd > compareThreshold)
                        SYNET_ERROR(failed << std::endl << std::fixed << "Dst[" << d << "] " << s.Name() << " " << DebugPrint(f.Shape()) << " at " << DebugPrint(Shp(n, 0)) << " : cosine distance " << cd << " > " << compareThreshold);
                }
            }
            else if (compType == "softmax" && (_options.bf16 || !_options.comparePrecise))
            {
                Tensor _f, _s;
                _f.Clone(f);
                _s.Clone(s);
                SimdSynetSoftmaxLayerForward(f.Data<float>(), f.Axis(0), f.Axis(1), 1, _f.Data<float>());
                SimdSynetSoftmaxLayerForward(s.Data<float>(), s.Axis(0), s.Axis(1), 1, _s.Data<float>());
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        if (!Compare(_f, _s, Shp(n, c), d, failed, compType, compareThreshold))
                            return false;
            }
            else if (compType == "sigmoid" && (_options.bf16 || !_options.comparePrecise))
            {
                Tensor _f, _s;
                _f.Clone(f);
                _s.Clone(s);
                float _1 = 1.0;
                SimdSynetSigmoid32f(f.Data<float>(), f.Size(), &_1, _f.Data<float>());
                SimdSynetSigmoid32f(s.Data<float>(), s.Size(), &_1, _s.Data<float>());
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        if (!Compare(_f, _s, Shp(n, c), d, failed, compType, compareThreshold))
                            return false;
            }
            else
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        if (!Compare(f, s, Shp(n, c), d, failed, "", compareThreshold))
                            return false;
            }
            return true;
        }

        bool Compare3d(const Tensor& f, const Tensor& s, size_t d, const String& failed, String compType, float compareThreshold) const
        {
            using Synet::Detail::DebugPrint;
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
                                if (!Compare(f, s, Shp(n, c, y), d, failed, "", compareThreshold))
                                    return false;
                        }
                    }
            }
            else if ((compType == "softmax-1" || compType == "softmax-2") && (_options.bf16 || !_options.comparePrecise))
            {
                Tensor _f, _s;
                _f.Clone(f);
                _s.Clone(s);
                if (compType == "softmax-1")
                {
                    SimdSynetSoftmaxLayerForward(f.Data<float>(), f.Axis(0), f.Axis(1), f.Axis(2), _f.Data<float>());
                    SimdSynetSoftmaxLayerForward(s.Data<float>(), s.Axis(0), s.Axis(1), s.Axis(2), _s.Data<float>());
                }
                else if (compType == "softmax-2")
                {
                    SimdSynetSoftmaxLayerForward(f.Data<float>(), f.Size(0, 2), f.Axis(2), 1, _f.Data<float>());
                    SimdSynetSoftmaxLayerForward(s.Data<float>(), s.Size(0, 2), s.Axis(2), 1, _s.Data<float>());
                }
                for (size_t n = 0; n < _f.Axis(0); ++n)
                    for (size_t c = 0; c < _f.Axis(1); ++c)
                        for (size_t y = 0; y < _f.Axis(2); ++y)
                            if (!Compare(_f, _s, Shp(n, c, y), d, failed, compType, compareThreshold))
                                return false;
            }
            else if (compType == "sigmoid" && (_options.bf16 || !_options.comparePrecise))
            {
                Tensor _f, _s;
                _f.Clone(f);
                _s.Clone(s);
                float _1 = 1.0;
                SimdSynetSigmoid32f(f.Data<float>(), f.Size(), &_1, _f.Data<float>());
                SimdSynetSigmoid32f(s.Data<float>(), s.Size(), &_1, _s.Data<float>());
                for (size_t n = 0; n < _f.Axis(0); ++n)
                    for (size_t c = 0; c < _f.Axis(1); ++c)
                        for (size_t y = 0; y < _f.Axis(2); ++y)
                            if (!Compare(_f, _s, Shp(n, c, y), d, failed, compType, compareThreshold))
                                return false;
            }
            else if (compType == "cos_dist-12" && (_options.bf16 || !_options.comparePrecise))
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                {
                    float cd;
                    SimdCosineDistance32f(f.Data<float>(Shp(n, 0, 0)), s.Data<float>(Shp(n, 0, 0)), f.Axis(1) * f.Axis(2), &cd);
                    if (cd > compareThreshold)
                        SYNET_ERROR(failed << std::endl << std::fixed << "Dst[" << d << "] " << s.Name() << " " << DebugPrint(f.Shape()) << " at " << DebugPrint(Shp(n, 0)) << " : cosine distance " << cd << " > " << compareThreshold);
                }
            }
            else
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        for (size_t y = 0; y < f.Axis(2); ++y)
                            if (!Compare(f, s, Shp(n, c, y), d, failed, "", compareThreshold))
                                return false;
            }
            return true;
        }

        bool Compare4d(const Tensor& f, const Tensor& s, size_t d, const String& failed, String compType, float compareThreshold) const
        {
            using Synet::Detail::DebugPrint;
            if (compType == "cos_dist" && (_options.bf16 || !_options.comparePrecise) && f.Axis(2) == 1 && f.Axis(3) == 1)
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                {
                    float cd;
                    SimdCosineDistance32f(f.Data<float>(Shp(n, 0, 0, 0)), s.Data<float>(Shp(n, 0, 0, 0)), f.Axis(1), &cd);
                    if (cd > _options.compareThreshold)
                        SYNET_ERROR(failed << std::endl << std::fixed << "Dst[" << d << "] " << s.Name() << " " << DebugPrint(f.Shape()) << " at " << DebugPrint(Shp(n, 0, 0, 0)) << " : cosine distance " << cd << " > " << compareThreshold);
                }
            }
            else if (compType == "avg2x2" || compType == "avg4x4" || compType == "avg8x8" || compType == "avg16x16")
            {
                size_t kX = 1, kY = 1;
                if (compType == "avg2x2")
                    kX = 2, kY = 2;
                else if (compType == "avg4x4")
                    kX = 4, kY = 4;
                else if (compType == "avg8x8")
                    kX = 8, kY = 8;                
                else if (compType == "avg16x16")
                    kX = 16, kY = 16;
                Shape shape = Shp(f.Axis(0), f.Axis(1), f.Axis(2) / kY, f.Axis(3) / kX);
                Tensor _f, _s;
                _f.Reshape(Synet::TensorType32f, shape, Synet::TensorFormatNchw);
                _s.Reshape(Synet::TensorType32f, shape, Synet::TensorFormatNchw);
                SimdSynetPoolingAverage(f.Data<float>(), f.Axis(0) * f.Axis(1), f.Axis(2), f.Axis(3), kY, kX, kY, kX, 0, 0, _f.Data<float>(), _f.Axis(2), _f.Axis(3), SimdFalse, SimdTensorFormatNchw);
                SimdSynetPoolingAverage(s.Data<float>(), s.Axis(0) * s.Axis(1), s.Axis(2), s.Axis(3), kY, kX, kY, kX, 0, 0, _s.Data<float>(), _s.Axis(2), _s.Axis(3), SimdFalse, SimdTensorFormatNchw);
                for (size_t n = 0; n < _f.Axis(0); ++n)
                    for (size_t c = 0; c < _f.Axis(1); ++c)
                        for (size_t y = 0; y < _f.Axis(2); ++y)
                            for (size_t x = 0; x < _f.Axis(3); ++x)
                                if (!Compare(_f, _s, Shp(n, c, y, x), d, failed, compType, compareThreshold))
                                    return false;
            }
            else
            {
                for (size_t n = 0; n < f.Axis(0); ++n)
                    for (size_t c = 0; c < f.Axis(1); ++c)
                        for (size_t y = 0; y < f.Axis(2); ++y)
                            for (size_t x = 0; x < f.Axis(3); ++x)
                                if (!Compare(f, s, Shp(n, c, y, x), d, failed, "", compareThreshold))
                                    return false;
            }
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
                    os << ExpandLeft(ToString(t.Data<float>(Shp(m[0], m[1], y, x))[0], 3), 8) << " ";
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
            float compareThreshold = _options.compareThreshold;
            if (_options.bf16)
            {
                if (_param.output().size() && _param.output()[0].bf16Threshold() != 0.0f)
                    compareThreshold = _param.output()[0].bf16Threshold();
            }
            else
            {
                if (_param.output().size() && _param.output()[0].fp32Threshold() != 0.0f)
                    compareThreshold = _param.output()[0].fp32Threshold();
            }
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
                    if (_f.id == rs[si].id && overlap > overlapMax)
                    {
                        indexMax = si;
                        overlapMax = overlap;
                    }
                }
                const Region& _s = rs[indexMax];
                //std::cout << "CompareRegions: " << ToStr(_f) << " and " << ToStr(_s) << std::endl;
                float d = std::max(std::max(::abs(_f.x - _s.x), ::abs(_f.y - _s.y)), std::max(abs(_f.w - _s.w), ::abs(_f.h - _s.h)));
                if (_f.id != _s.id || d > compareThreshold)
                {
                    SYNET_ERROR(failed << std::endl << " Region[" << fi << "] from " << n << " : " << ToStr(_f) << " != " << ToStr(_s) << " : difference " << d << " > " << compareThreshold);
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

