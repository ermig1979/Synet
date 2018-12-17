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
    template <typename T> void ImgToCol(const T * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t padH, size_t padW, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, T * dst)
    {
        SYNET_PERF_FUNC();

        size_t dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        size_t dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        size_t srcSize = srcW * srcH;
        if (dilationX == 1 && dilationY == 1 && strideX == 2 && strideY == 2 && padX == 0 && padY == 0 && padW == 0 && padH == 0 && kernelX == 1 && kernelY == 1)
        {
            for (size_t channel = 0; channel < srcC; ++channel)
            {
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    const T * psrc = src + 2*dy*srcW;
                    for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += 2)
                        *(dst++) = psrc[sx];
                }
                src += srcSize;
            }
        }
        else if (dilationX*dilationY*strideX*strideY != 1)
        {
            for (size_t channel = 0; channel < srcC; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ky++)
                {
                    for (size_t kx = 0; kx < kernelX; kx++)
                    {
                        size_t sy = ky * dilationY - padY;
                        for (size_t dy = 0; dy < dstH; ++dy)
                        {
                            if (sy < srcH)
                            {
                                size_t sx = kx * dilationX - padX;
                                for (size_t dx = 0; dx < dstW; ++dx)
                                {
                                    if (sx < srcW)
                                        *(dst++) = src[sy * srcW + sx];
                                    else
                                        *(dst++) = 0;
                                    sx += strideX;
                                }
                            }
                            else
                            {
                                for (size_t dx = 0; dx < dstW; ++dx)
                                    *(dst++) = 0;
                            }
                            sy += strideY;
                        }
                    }
                }
                src += srcSize;
            }        
        }
        else
        {
            const ptrdiff_t bodySize = dstW - padX - padW;
            for (size_t channel = 0; channel < srcC; ++channel)
            {
                for (size_t ky = 0; ky < kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < kernelX; ++kx)
                    {
                        size_t sy = ky - padY;
                        for (size_t dy = 0; dy < dstH; ++dy, ++sy)
                        {
                            if (sy < srcH)
                            {
                                size_t sx = kx - padX, dx = 0;
                                const T * psrc = src + sy*srcW;
                                for (; dx < padX; ++dx, ++sx)
                                {
                                    if (sx < srcW)
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
                                for (; dx < dstW; ++dx, ++sx)
                                {
                                    if (sx < srcW)
                                        *(dst++) = psrc[sx];
                                    else
                                        *(dst++) = 0;
                                }
                            }
                            else
                            {
                                memset(dst, 0, dstW * sizeof(T));
                                dst += dstW;
                            }
                        }
                    }
                }
                src += srcSize;
            }
        }
    }

    template <typename T> void ImgToRow(const T * src, size_t srcH, size_t srcW, size_t srcC, size_t kernelY, size_t kernelX,
        size_t padY, size_t padX, size_t padH, size_t padW, size_t strideY, size_t strideX, size_t dilationY, size_t dilationX, size_t group, T * dst)
    {
        SYNET_PERF_FUNC();

        size_t dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        size_t dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;

        size_t size = srcC / group;
        for (size_t g = 0; g < group; ++g)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t ky = 0; ky < kernelY; ky++)
                    {
                        size_t sy = dy*strideY + ky * dilationY - padY;
                        if (sy < srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; kx++)
                            {
                                size_t sx = dx*strideX + kx * dilationX - padX;
                                if (sx < srcW)
                                {
                                    memcpy(dst, src + (sy * srcW + sx)*srcC, size * sizeof(float));
                                    dst += size;
                                }
                                else
                                {
                                    memset(dst, 0, size * sizeof(float));
                                    dst += size;
                                }
                            }
                        }
                        else
                        {
                            memset(dst, 0, kernelX * size * sizeof(float));
                            dst += kernelX*size;
                        }
                    }
                }
            }
            src += size;
        }
    }
}