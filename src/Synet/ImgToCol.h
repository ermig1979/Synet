/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

#include "Synet/Common.h"

namespace Synet
{
    template <typename T> void ImgToCol(const T * src, size_t channels, size_t srcY, size_t srcX, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, T * dst)
    {
        SYNET_PERF_FUNC();

        size_t dstY = (srcY + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        size_t dstX = (srcX + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        size_t channelSize = srcX * srcY;
        if (dilationX == 1 && dilationY == 1 && strideX == 2 && strideY == 2 && padX == 0 && padY == 0 && kernelX == 1 && kernelY == 1)
        {
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t dy = 0; dy < dstY; ++dy)
                {
                    const T * psrc = src + 2*dy*srcX;
                    for (size_t dx = 0, sx = 0; dx < dstX; ++dx, sx += 2)
                        *(dst++) = psrc[sx];
                }
                src += channelSize;
            }
        }
        else if (dilationX*dilationY*strideX*strideY != 1)
        {
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ky++)
                {
                    for (size_t kx = 0; kx < kernelX; kx++)
                    {
                        size_t sy = ky * dilationY - padY;
                        for (size_t dy = 0; dy < dstY; ++dy)
                        {
                            if (sy < srcY)
                            {
                                size_t sx = kx * dilationX - padX;
                                for (size_t dx = 0; dx < dstX; ++dx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = src[sy * srcX + sx];
                                    else
                                        *(dst++) = 0;
                                    sx += strideX;
                                }
                            }
                            else
                            {
                                for (size_t dx = 0; dx < dstX; ++dx)
                                    *(dst++) = 0;
                            }
                            sy += strideY;
                        }
                    }
                }
                src += channelSize;
            }        
        }
        else
        {
            const ptrdiff_t bodySize = dstX - padX * 2;
            for (size_t channel = 0; channel < channels; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < kernelX; ++kx)
                    {
                        size_t sy = ky - padY;
                        for (size_t dy = 0; dy < dstY; ++dy, ++sy)
                        {
                            if (sy < srcY)
                            {
                                size_t sx = kx - padX, dx = 0;
                                const T * psrc = src + sy*srcX;
                                for (; dx < padX; ++dx, ++sx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = psrc[sx];
                                    else
                                        *(dst++) = 0;
                                }
                                if (bodySize > 0)
                                {
                                    memcpy(dst, psrc + sx, bodySize * sizeof(T));
                                    dst += bodySize;
                                    dx += bodySize;
                                    sx += bodySize;
                                }
                                for (; dx < dstX; ++dx, ++sx)
                                {
                                    if (sx < srcX)
                                        *(dst++) = psrc[sx];
                                    else
                                        *(dst++) = 0;
                                }
                            }
                            else
                            {
                                memset(dst, 0, dstX * sizeof(T));
                                dst += dstX;
                            }
                        }
                    }
                }
                src += channelSize;
            }
        }
    }
}