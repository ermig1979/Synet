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

#include "Synet/Params.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    SYNET_INLINE float DequantizeLinear(int value, int zero, float norm)
    {
        return float(value - zero) * norm;
    }

    SYNET_INLINE void DequantizeLinear(const int32_t *src, size_t batch, size_t channels, size_t height, size_t width, TensorFormat format, const int32_t* zero, const float *norm, float * dst)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    int32_t _zero = zero[c];
                    float _norm = norm[c];
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                            dst[w] = DequantizeLinear(src[w], _zero, _norm);
                        src += width;
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
                            dst[c] = DequantizeLinear(src[c], zero[c], norm[c]);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
                assert(0);
        }
    }
}