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

#include "Synet/Tensor.h"

namespace Synet
{
    template<class T> struct DifferenceStatistics
    {
        T max, min, sum, sqsum;
        size_t count;
        Shape indexMin, indexMax;
        DifferenceStatistics()
            : max(0), min(0), sum(0), sqsum(0), count(0)
        {
        }

        void Update(T val, const Shape& idx)
        {
            count += 1;
            sum += val;
            sqsum += val * val;
            if (val > max)
            {
                max = val;
                indexMax = idx;
            }
            if (val < min)
            {
                min = val;
                indexMin = idx;
            }
        }
    };

	namespace Detail
	{
        template <typename T> void UpdateChannelsMaxAbs(const T* src, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format, T* max)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                            {
                                float abs = std::abs(src[c]);
                                max[c] = std::max(max[c], abs);
                            }
                            src += channels;
                        }
                    }
                }
                else if (format == TensorFormatNchw)
                {
					for (size_t c = 0; c < channels; ++c)
					{
                        float _max = max[c];
						for (size_t h = 0; h < height; ++h)
						{
							for (size_t w = 0; w < width; ++w)
							{
								float abs = std::abs(src[w]);
                                _max = std::max(_max, abs);
							}
							src += width;
						}
                        max[c] = _max;
					}
                }
                else
                    assert(0);
            }
        }

        template <typename T, typename D> void SetDifferenceStatistics(const T* first, const T* second, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format,
            const T* norm, Synet::DifferenceStatistics<D>& diff)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                diff.Update((first[c] - second[c]) / norm[c], Shp(b, c, h, w));
                            first += channels, second += channels;
                        }
                    }
                }
                else if (format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                diff.Update((first[w] - second[w]) / norm[c], Shp(b, c, h, w));
                            first += width, second += width;
                        }
                    }
                }
                else
                    assert(0);
            }
        }
	}

    template <typename T> inline void UpdateChannelsMaxAbs(const Tensor<T>& tensor, T* max)
    {
        assert(tensor.Count() == 4);
        if (tensor.Format() == TensorFormatNhwc)
            Detail::UpdateChannelsMaxAbs(tensor.CpuData(), tensor.Axis(0), tensor.Axis(3), tensor.Axis(1), tensor.Axis(2), tensor.Format(), max);
        else if (tensor.Format() == TensorFormatNchw)
            Detail::UpdateChannelsMaxAbs(tensor.CpuData(), tensor.Axis(0), tensor.Axis(1), tensor.Axis(2), tensor.Axis(3), tensor.Format(), max);
        else
            assert(0);
    }

    template <typename T, typename D> inline void SetDifferenceStatistics(const Tensor<T>& a, const Tensor<T>& b, const T* n, DifferenceStatistics<D>& d)
    {
        assert(a.Count() == 4);
        if (a.Format() == TensorFormatNhwc)
            Detail::SetDifferenceStatistics(a.CpuData(), b.CpuData(), a.Axis(0), a.Axis(3), a.Axis(1), a.Axis(2), a.Format(), n, d);
        else if (a.Format() == TensorFormatNchw)
            Detail::SetDifferenceStatistics(a.CpuData(), b.CpuData(), a.Axis(0), a.Axis(1), a.Axis(2), a.Axis(3), a.Format(), n, d);
        else
            assert(0);
    }

	template <typename T, typename D> inline void EstimateDifference(const Tensor<T>& a, const Tensor<T>& b, DifferenceStatistics<D>& d)
	{
        assert(a.Count() == 4 && a.Shape() == b.Shape());
        size_t channels = a.Format() == TensorFormatNhwc ? a.Axis(3) : a.Axis(1);
        std::vector<T> max(channels, 1.0);
        UpdateChannelsMaxAbs(a, max.data());
        UpdateChannelsMaxAbs(b, max.data());
        SetDifferenceStatistics(a, b, max.data(), d);
	}
}