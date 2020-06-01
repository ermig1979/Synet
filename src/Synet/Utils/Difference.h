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

            _statistics.Resume();

            return true;
        }

    private:

        TensorFormat _format;
        size_t _batch, _channels, _height, _width;
        Vector _norm;

        struct Specific
        {
            String name;
            double diff;
            Type first, second;
            Shape index;
            Specific()
                : diff(0)
            {}
        };
        typedef std::vector<Specific> Specifics;        
        
        struct Statistics
        {
            double sum, sqsum, average, sigma;
            size_t count;
            Specific absMax;

            Statistics()
                : sum(0), sqsum(0), count(0)
            {
            }

            void Update(Type first, Type second, Type norm, const Shape& index)
            {
                double diff = (first - second) * norm;
                count += 1;
                sum += diff;
                sqsum += diff * diff;
                double abs = std::abs(diff);
                if (abs > absMax.diff)
                {
                    absMax.diff = abs;
                    absMax.first = first;
                    absMax.second = second;
                    absMax.index = index;
                }
            }

            void Resume()
            {
                average = sum / count;
                sigma = sqrt(sqsum / count - average * average);
            }
        } _statistics;        

        bool Init(const Tensor& tensor)
        {
            _format = tensor.Format();
            if (tensor.Count() != 4)
                return false;
            if (_format == TensorFormatNchw)
            {
                _batch = tensor.Axis(0);
                _channels = tensor.Axis(1);
                _height = tensor.Axis(2);
                _width = tensor.Axis(3);
            }
            else if (TensorFormatNhwc)
            {
                _batch = tensor.Axis(0);
                _height = tensor.Axis(1);
                _width = tensor.Axis(2);
                _channels = tensor.Axis(3);
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
                                _statistics.Update(first[c], second[c], _norm[c], Shp(b, c, h, w));
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
                                _statistics.Update(first[w], second[w], _norm[c], Shp(b, c, h, w));
                            first += _width, second += _width;
                        }
                    }
                }
                else
                    assert(0);
            }
        }
    };
}