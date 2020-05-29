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

	template <typename T> inline void EstimateDifference(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& d)
	{
        assert(a.Count() == 4 && a.Shape() == b.Shape());
        size_t channels = a.Format() == TensorFormatNhwc ? a.Axis(3) : a.Axis(1);
        std::vector<T> max(channels, 1.0);
        UpdateChannelsMaxAbs(a, max.data());
        UpdateChannelsMaxAbs(b, max.data());
	}
}