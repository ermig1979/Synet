/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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

#include "Synet/Utils/Statistics.h"

namespace Synet
{
    template <typename T> class Difference
    {
    public:
        typedef T Type;
        typedef Synet::Tensor<T> Tensor;
        typedef std::vector<T> Vector;

        struct Specific
        {
            double diff;
            Type first, second;
            Shape index;
            Specific(double d = 0, Type f = 0, Type s = 0, const Shape& i = Shape())
                : diff(d), first(f), second(s), index(i) {}
        };
        typedef std::vector<Specific> Specifics;
        typedef std::vector<int> Histogram;

        struct Statistics
        {
            double vsum, asum, ssum, mean, sdev, adev;
            size_t count;
            Specific max;
            Specifics spec;
            Histogram hist;

            Statistics()
                : vsum(0), asum(0), ssum(0), count(0)
            {
                hist.resize(32, 0);
            }
        };

        Difference()
        {
        }

        bool Estimate(const Tensor & first, const Tensor & second)
        {
            if (first.Format() != second.Format() || first.Shape() != second.Shape())
                return false;

            if (!Init(first))
                return false;

            SetNorm(first.CpuData());

            CollectStatistics(first.CpuData(), second.CpuData());

            SetResume();

            return true;
        }

        const Statistics & GetStatistics() const
        {
            return _statistics;
        }

        const Shape & GetShape() const
        {
            return _shape;
        }

        const Specific* Exceed(double threshold, double quantile = 0)
        {
            size_t count = size_t(_statistics.count * quantile), exceed = 0;
            for (size_t i = 0; i < _statistics.spec.size(); ++i)
            {
                if (_statistics.spec[i].diff > threshold)
                {
                    exceed += _statistics.hist[i];
                    if(exceed >= count)
                        return _statistics.spec.data() + i;
                }
            }
            if (_statistics.max.diff > threshold && exceed >= count)
                return &_statistics.max;
            return NULL;
        }

    private:

        TensorFormat _format;
        size_t _batch, _channels, _height, _width;
        Shape _shape;
        Vector _norm;
        Statistics _statistics;

        bool Init(const Tensor& tensor)
        {
            _format = tensor.Format();
            _shape = tensor.Shape();
            if (_shape.size() != 4)
                return false;
            if (_format == TensorFormatNchw)
            {
                _batch = _shape[0];
                _channels = _shape[1];
                _height = _shape[2];
                _width = _shape[3];
            }
            else if (TensorFormatNhwc)
            {
                _batch = _shape[0];
                _height = _shape[1];
                _width = _shape[2];
                _channels = _shape[3];
            }
            else
                return false;
            return true;
        }

        void SetNorm(const T * src)
        {
            Vector min, max;
            min.resize(_channels, std::numeric_limits<Type>::max());
            max.resize(_channels, std::numeric_limits<Type>::lowest());
            Detail::UpdateChannelsMinMax(src, _batch, _channels, _height, _width, _format, min.data(), max.data());
            _norm.resize(_channels);
            for (size_t c = 0; c < _channels; ++c)
                _norm[c] = Type(1) / std::max(Type(1), max[c] - min[c]);
        }

        void CollectStatistics(const Type * first, const Type * second)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        for (size_t w = 0; w < _width; ++w)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                                UpdateStatistics(first[c], second[c], _norm[c], Shp(b, c, h, w));
                            first += _channels, second += _channels;
                        }
                    }
                }
                else if (_format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        for (size_t h = 0; h < _height; ++h)
                        {
                            for (size_t w = 0; w < _width; ++w)
                                UpdateStatistics(first[w], second[w], _norm[c], Shp(b, c, h, w));
                            first += _width, second += _width;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        void UpdateStatistics(Type first, Type second, Type norm, const Shape& index)
        {
            double diff = (first - second) * norm;
            double absd = std::abs(diff);
            _statistics.count += 1;
            _statistics.vsum += diff;
            _statistics.asum += absd;
            _statistics.ssum += diff * diff;
            if (absd > _statistics.max.diff)
                _statistics.max = Specific(absd, first, second, index);
            int idx = int(10.0 * log10(std::max(1.0, std::min(1.0, absd) * 1000.0)));
            if (_statistics.hist[idx] == 0)
                _statistics.spec.push_back(Specific(absd, first, second, index));
            _statistics.hist[idx]++;
        }

        void SetResume()
        {
            _statistics.mean = _statistics.vsum / _statistics.count;
            _statistics.adev = _statistics.asum / _statistics.count;
            _statistics.sdev = sqrt(_statistics.ssum / _statistics.count - _statistics.mean * _statistics.mean);
            std::sort(_statistics.spec.begin(), _statistics.spec.end(), [](const Specific& a, const Specific& b) { return a.diff < b.diff; });
        }
    };
}