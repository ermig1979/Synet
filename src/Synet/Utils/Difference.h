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
            size_t count;
            Specific(double d = 0, Type f = 0, Type s = 0, const Shape& i = Shape(), size_t c = 0)
                : diff(d), first(f), second(s), index(i), count(c) {}
        };
        typedef std::vector<int> Histogram;

        struct Statistics
        {
            double vsum, asum, ssum, mean, sdev, adev;
            size_t count;
            Specific max, exceed;
            Histogram hist;

            Statistics()
                : vsum(0), asum(0), ssum(0), count(0)
            {
                hist.resize(32, 0);
            }
        };

        Difference(const Tensor& first, const Tensor& second)
            : _valid(false)
        {
            _valid = Init(first, second);
        }

        bool Valid() const
        {
            return _valid;
        }

        bool Estimate(double threshold, double quantile = 0)
        {
            SetNorm(_first);

            CollectStatistics(threshold);

            return Resume(quantile);
        }

        const Statistics & GetStatistics() const
        {
            return _statistics;
        }

        const Shape & GetShape() const
        {
            return _shape;
        }

    private:

        bool _valid;
        const Type* _first, * _second;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width;
        Shape _shape;
        Vector _norm;
        Statistics _statistics;

        bool Init(const Tensor& first, const Tensor& second)
        {
            if (first.Format() != second.Format() || first.Shape() != second.Shape())
                return false;
            _format = first.Format();
            _shape = first.Shape();
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
            _first = first.CpuData();
            _second = second.CpuData();
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

        void CollectStatistics(double threshold)
        {
            const Type* first = _first;
            const Type* second = _second;
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        for (size_t w = 0; w < _width; ++w)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                                UpdateStatistics(first[c], second[c], _norm[c], Shp(b, c, h, w), threshold);
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
                                UpdateStatistics(first[w], second[w], _norm[c], Shp(b, c, h, w), threshold);
                            first += _width, second += _width;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        void UpdateStatistics(Type first, Type second, Type norm, const Shape& index, double threshold)
        {
            double diff = (first - second) * norm;
            double absd = std::abs(diff);
            _statistics.count += 1;
            _statistics.vsum += diff;
            _statistics.asum += absd;
            _statistics.ssum += diff * diff;
            if (absd > _statistics.max.diff)
                _statistics.max = Specific(absd, first, second, index, 1);
            int idx = int(10.0 * log10(std::max(1.0, std::min(1.0, absd) * 1000.0)));
            _statistics.hist[idx]++;
            if (absd >= threshold)
            {
                if (_statistics.exceed.count == 0)
                    _statistics.exceed = Specific(absd, first, second, index, 1);
                else
                    _statistics.exceed.count++;
            }
        }

        bool Resume(double quantile)
        {
            _statistics.mean = _statistics.vsum / _statistics.count;
            _statistics.adev = _statistics.asum / _statistics.count;
            _statistics.sdev = sqrt(_statistics.ssum / _statistics.count - _statistics.mean * _statistics.mean);
            return int(quantile*_statistics.count) >= _statistics.exceed.count;
        }
    };
}