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
    namespace Detail
    {
        template <typename T> void UpdateChannelsMinMax(const T* src, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format, T* min, T* max)
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
                                float value = src[c];
                                min[c] = std::min(min[c], value);
                                max[c] = std::max(max[c], value);
                            }
                            src += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        template <typename T> void UpdateChannelsHistogram(const T* src, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format, const T* min, const T* max, int32_t* histogram, size_t size)
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
                                T value = (src[c] - min[c]) / (max[c] - min[c]);
                                int32_t index = int32_t(value * size);
                                histogram[c * size + index]++;
                            }
                            src += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        template <typename T> void ValidateMinMax(const T * min, T * max, size_t size, T epsilon)
        {
            for (size_t i = 0; i < size; ++i)
                max[i] = min[i] + std::max(max[i] - min[i], epsilon);
        }
    }

    template <typename T> void UpdateChannelsMinMax(const Tensor<T> & tensor, T* min, T* max)
    {
        assert(tensor.Count() == 4);
        if (tensor.Format() == TensorFormatNhwc)
            Detail::UpdateChannelsMinMax(tensor.CpuData(), tensor.Axis(0), tensor.Axis(3), tensor.Axis(1), tensor.Axis(2), tensor.Format(), min, max);
        else
            assert(0);
    }

    template <typename T> void UpdateChannelsHistogram(const Tensor<T>& tensor, const T* min, const T* max)
    {
        assert(tensor.Count() == 4);
        if (tensor.Format() == TensorFormatNhwc)
            Detail::UpdateChannelsHistogram(tensor.CpuData(), tensor.Axis(0), tensor.Axis(3), tensor.Axis(1), tensor.Axis(2), tensor.Format(), min, max);
        else
            assert(0);
    }

    template <typename T> void UpdateChannelsQuantile(const Tensor<T>& tensor, T quntile, T epsilon, T * lower, T * upper)
    {
        assert(tensor.Count() == 4 && tensor.Format() == TensorFormatNhwc);

        if (quntile == T(0))
        {
            UpdateChannelsMinMax(tensor, lower, upper);
            return;
        }
        size_t channels = tensor.Axis(3);
        std::vector<T> min(channels, std::numeric_limits<T>::max());
        std::vector<T> max(channels, std::numeric_limits<T>::lowest());
        UpdateChannelsMinMax(tensor, min.data(), max.data());
        Detail::ValidateMinMax(min.data(), max.data(), min.size(), epsilon);
    }
}