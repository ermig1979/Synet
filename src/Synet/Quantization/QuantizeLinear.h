/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Utils/Math.h"

namespace Synet
{
    SYNET_INLINE int QuantizeLinear(float value, float scale, int zero, int min, int max)
    {
        return RestrictRange(Round(value * scale) + zero, min, max);
    }

    //--------------------------------------------------------------------------------------------------

    SYNET_INLINE int QuantizeSumLinear(int sum, int bias, float norm, int zero, int min, int max)
    {
        return RestrictRange(Round(float(sum + bias) * norm) + zero, min, max);
    }

    SYNET_INLINE void QuantizeSumLinear(const int32_t* sum, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format, const int32_t* bias, const float* norm, const uint8_t* zero, uint8_t* dst)
    {
        int min = std::numeric_limits<uint8_t>::min();
        int max = std::numeric_limits<uint8_t>::max();
        for (size_t b = 0; b < batch; ++b)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    int32_t _bias = bias[c];
                    float _norm = norm[c];
                    int32_t _zero = zero[c];
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                            dst[w] = (uint8_t)QuantizeSumLinear(sum[w], _bias, _norm, _zero, min, max);
                        sum += width;
                        dst += width;
                    }
                }
            }
            else if (format == TensorFormatNhwc)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = (uint8_t)QuantizeSumLinear(sum[w], bias[c], norm[c], zero[c], min, max);
                        sum += channels;
                        dst += channels;
                    }
                }
            }
            else
                assert(0);
        }
    }
}