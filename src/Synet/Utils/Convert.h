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

#include "Synet/Utils/Math.h"
#include "Synet/Params.h"

namespace Synet
{
    namespace Detail
    {
        inline void Convert32fTo8uNchw(const float * src, size_t channels, size_t spatial, const float * scale, const float * shift, uint8_t * dst)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                float _scale = scale[c];
                float _shift = shift[c];
                for (size_t s = 0; s < spatial; ++s)
                    dst[s] = (uint8_t)Synet::Round(src[s] * _scale + _shift);
                src += spatial;
                dst += spatial;
            }
        }

        inline void Convert8uTo32fNchw(const uint8_t * src, size_t channels, size_t spatial, const float * scale, const float * shift, float * dst)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                float _scale = scale[c];
                float _shift = shift[c];
                for (size_t s = 0; s < spatial; ++s)
                    dst[s] = float(src[s]) * _scale + _shift;
                src += spatial;
                dst += spatial;
            }
        }
    }

    inline void Convert32fTo8u(const float * src, size_t channels, size_t spatial, TensorFormat format, const float * scale, const float * shift, uint8_t * dst)
    {
        if (format == TensorFormatNchw)
            Detail::Convert32fTo8uNchw(src, channels, spatial, scale, shift, dst);
        else
            assert(0);
    }

    inline void Convert8uTo32f(const uint8_t * src, size_t channels, size_t spatial, TensorFormat format, const float * scale, const float * shift, float * dst)
    {
        if (format == TensorFormatNchw)
            Detail::Convert8uTo32fNchw(src, channels, spatial, scale, shift, dst);
        else
            assert(0);
    }
}