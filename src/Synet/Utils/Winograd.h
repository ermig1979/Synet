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

#include "Synet/Common.h"
#include "Synet/Utils/Gemm.h"

namespace Synet
{

    namespace Winograd2x3i
    {
        template <class T> void SetFilter(const T * src, size_t size, T * dst)
        {
            const T r2 = T(1.0 / 2);
            const T r4 = T(1.0 / 4);
            for (size_t i = 0; i < size; ++i, src += 9, dst += 16)
            {
                dst[0] = src[0];
                dst[1] = (src[0] + src[2] + src[1])*r2;
                dst[2] = (src[0] + src[2] - src[1])*r2;
                dst[3] = src[2];
                dst[4] = (src[0] + src[6] + src[3])*r2;
                dst[5] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) + (src[1] + src[7] + src[4]))*r4;
                dst[6] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) - (src[1] + src[7] + src[4]))*r4;
                dst[7] = (src[2] + src[8] + src[5])*r2;
                dst[8] = (src[0] + src[6] - src[3])*r2;
                dst[9] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) + (src[1] + src[7] - src[4]))*r4;
                dst[10] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) - (src[1] + src[7] - src[4]))*r4;
                dst[11] = (src[2] + src[8] - src[5])*r2;
                dst[12] = src[6];
                dst[13] = (src[6] + src[8] + src[7])*r2;
                dst[14] = (src[6] + src[8] - src[7])*r2;
                dst[15] = src[8];
            }
        }

        template <class T> void SetInput1(const T * src, size_t srcStride, T * dst)
        {
            T tmp[16];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[0 * srcStride + 3];

            tmp[4] = src[1 * srcStride + 0];
            tmp[5] = src[1 * srcStride + 1];
            tmp[6] = src[1 * srcStride + 2];
            tmp[7] = src[1 * srcStride + 3];

            tmp[8] = src[2 * srcStride + 0];
            tmp[9] = src[2 * srcStride + 1];
            tmp[10] = src[2 * srcStride + 2];
            tmp[11] = src[2 * srcStride + 3];

            tmp[12] = src[3 * srcStride + 0];
            tmp[13] = src[3 * srcStride + 1];
            tmp[14] = src[3 * srcStride + 2];
            tmp[15] = src[3 * srcStride + 3];

            dst[0] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
            dst[1] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
            dst[2] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
            dst[3] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
            dst[4] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
            dst[5] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
            dst[6] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
            dst[7] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
            dst[8] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
            dst[9] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
            dst[10] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
            dst[11] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
            dst[12] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
            dst[13] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
            dst[14] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
            dst[15] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);
        }

        template <class T> void SetInput1p(const T * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, T * dst)
        {
            T tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            SetInput1(tmp, 4, dst);
        }

        template <class T> void SetInput(const T * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, T * dst, bool pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = dstHeight / 2 * 2;
            size_t dstWidthFull = dstWidth / 2 * 2;
            size_t noseW = std::min<size_t>(4, dstWidth + 1);
            size_t noseH = std::min<size_t>(4, dstHeight + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst), dst += 16;
                    if (col < dstWidth)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst), dst += 16;
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1(src + row * srcWidth + col, srcWidth, dst), dst += 16;
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst), dst += 16;
                }
                if (row < dstHeight)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst), dst += 16;
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst), dst += 16;
                }
                src += srcWidth*srcHeight;
            }
        }

        template <class T> void Gemm(size_t M, size_t N, size_t K, const T * A, const T * B, T * C)
        {
            std::vector<T> pB(K * 16);
            for (size_t j = 0; j < N; ++j)
            {
                for (size_t i = 0; i < M; ++i)
                {
                    T * c = C + (i * N + j) * 16;
                    for (size_t l = 0; l < 16; ++l)
                        c[l] = 0;
                    if (i == 0)
                    {
                        for (size_t k = 0; k < K; ++k)
                        {
                            const T * b = B + (k * N + j) * 16;
                            T * pb = pB.data() + k * 16;
                            for (size_t l = 0; l < 16; ++l)
                                pb[l] = b[l];
                        }
                    }
                    const T * b = pB.data();
                    const T * a = A + (i * K) * 16;
                    for (size_t k = 0; k < K; ++k)
                    {
                        c[0] += a[0] * b[0];
                        c[1] += a[1] * b[1];
                        c[2] += a[2] * b[2];
                        c[3] += a[3] * b[3];
                        c[4] += a[4] * b[4];
                        c[5] += a[5] * b[5];
                        c[6] += a[6] * b[6];
                        c[7] += a[7] * b[7];
                        c[8] += a[8] * b[8];
                        c[9] += a[9] * b[9];
                        c[10] += a[10] * b[10];
                        c[11] += a[11] * b[11];
                        c[12] += a[12] * b[12];
                        c[13] += a[13] * b[13];
                        c[14] += a[14] * b[14];
                        c[15] += a[15] * b[15];
                        b += 16;
                        a += 16;
                    }
                }
            }
        }

        template <class T> void SetOutput1(const T * src, T * dst, size_t dstStride)
        {
            T tmp[8];
            tmp[0] = src[0] + src[1] + src[2];
            tmp[1] = src[1] - src[2] - src[3];
            tmp[2] = src[4] + src[5] + src[6];
            tmp[3] = src[5] - src[6] - src[7];
            tmp[4] = src[8] + src[9] + src[10];
            tmp[5] = src[9] - src[10] - src[11];
            tmp[6] = src[12] + src[13] + src[14];
            tmp[7] = src[13] - src[14] - src[15];

            dst[0 * dstStride + 0] = tmp[0] + tmp[2] + tmp[4];
            dst[0 * dstStride + 1] = tmp[1] + tmp[3] + tmp[5];
            dst[1 * dstStride + 0] = tmp[2] - tmp[4] - tmp[6];
            dst[1 * dstStride + 1] = tmp[3] - tmp[5] - tmp[7];
        }

        template <class T> void SetOutput1p(const T * src, T * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            T tmp[2 * 2];
            SetOutput1(src, tmp, 2);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 2 + col];
        }

        template <class T> void SetOutput(const T * src, T * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t dstHeightFull = dstHeight / 2 * 2;
            size_t dstWidthFull = dstWidth / 2 * 2;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        SetOutput1(src, dst + row*dstWidth + col, dstWidth), src += 16;
                    if (col < dstWidth)
                        SetOutput1p(src, dst + row*dstWidth + col, dstWidth, 2, dstWidth - col), src += 16;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        SetOutput1p(src, dst + row*dstWidth + col, dstWidth, dstHeight - row, 2), src += 16;
                    if (col < dstWidth)
                        SetOutput1p(src, dst + row*dstWidth + col, dstWidth, dstHeight - row, dstWidth - col), src += 16;
                }
                dst += dstHeight * dstWidth;
            }
        }
    }

    namespace Winograd2x3p
    {
        template <class T> void SetFilter(const T * src, size_t size, T * dst)
        {
            const T r2 = T(1.0 / 2);
            const T r4 = T(1.0 / 4);
            for (size_t i = 0; i < size; ++i, src += 9, dst += 1)
            {
                dst[0 * size] = src[0];
                dst[1 * size] = (src[0] + src[2] + src[1])*r2;
                dst[2 * size] = (src[0] + src[2] - src[1])*r2;
                dst[3 * size] = src[2];
                dst[4 * size] = (src[0] + src[6] + src[3])*r2;
                dst[5 * size] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) + (src[1] + src[7] + src[4]))*r4;
                dst[6 * size] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) - (src[1] + src[7] + src[4]))*r4;
                dst[7 * size] = (src[2] + src[8] + src[5])*r2;
                dst[8 * size] = (src[0] + src[6] - src[3])*r2;
                dst[9 * size] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) + (src[1] + src[7] - src[4]))*r4;
                dst[10 * size] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) - (src[1] + src[7] - src[4]))*r4;
                dst[11 * size] = (src[2] + src[8] - src[5])*r2;
                dst[12 * size] = src[6];
                dst[13 * size] = (src[6] + src[8] + src[7])*r2;
                dst[14 * size] = (src[6] + src[8] - src[7])*r2;
                dst[15 * size] = src[8];
            }
        }

        template <class T> void SetInput1(const T * src, size_t srcStride, T * dst, size_t dstStride)
        {
            T tmp[16];
            tmp[0] = src[0*srcStride + 0];
            tmp[1] = src[0*srcStride + 1];
            tmp[2] = src[0*srcStride + 2];
            tmp[3] = src[0*srcStride + 3];

            tmp[4] = src[1*srcStride + 0];
            tmp[5] = src[1*srcStride + 1];
            tmp[6] = src[1*srcStride + 2];
            tmp[7] = src[1*srcStride + 3];

            tmp[8] = src[2*srcStride + 0];
            tmp[9] = src[2*srcStride + 1];
            tmp[10] = src[2*srcStride + 2];
            tmp[11] = src[2*srcStride + 3];

            tmp[12] = src[3*srcStride + 0];
            tmp[13] = src[3*srcStride + 1];
            tmp[14] = src[3*srcStride + 2];
            tmp[15] = src[3*srcStride + 3];

            dst[0 * dstStride] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
            dst[1 * dstStride] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
            dst[2 * dstStride] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
            dst[3 * dstStride] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
            dst[4 * dstStride] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
            dst[5 * dstStride] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
            dst[6 * dstStride] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
            dst[7 * dstStride] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
            dst[8 * dstStride] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
            dst[9 * dstStride] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
            dst[10 * dstStride] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
            dst[11 * dstStride] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
            dst[12 * dstStride] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
            dst[13 * dstStride] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
            dst[14 * dstStride] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
            dst[15 * dstStride] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);
        }

        template <class T> void SetInput1p(const T * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, T * dst, size_t dstStride)
        {
            T tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            SetInput1(tmp, 4, dst, dstStride);
        }

        template <class T> void SetInput(const T * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, T * dst, bool pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstStride = ((dstHeight + 1) / 2) * ((dstWidth + 1) / 2)*srcChannels;
            size_t dstHeightFull = dstHeight / 2 * 2;
            size_t dstWidthFull = dstWidth / 2 * 2;
            size_t noseW = std::min<size_t>(4, dstWidth + 1);
            size_t noseH = std::min<size_t>(4, dstHeight + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth*srcHeight;
            }
        }

        template <class T> void SetOutput1(const T * src, size_t srcStride, T * dst, size_t dstStride)
        {
            T c1[16];
            c1[0] = src[0 * srcStride];
            c1[1] = src[1 * srcStride];
            c1[2] = src[2 * srcStride];
            c1[3] = src[3 * srcStride];
            c1[4] = src[4 * srcStride];
            c1[5] = src[5 * srcStride];
            c1[6] = src[6 * srcStride];
            c1[7] = src[7 * srcStride];
            c1[8] = src[8 * srcStride];
            c1[9] = src[9 * srcStride];
            c1[10] = src[10 * srcStride];
            c1[11] = src[11 * srcStride];
            c1[12] = src[12 * srcStride];
            c1[13] = src[13 * srcStride];
            c1[14] = src[14 * srcStride];
            c1[15] = src[15 * srcStride];

            T tmp[8];
            tmp[0] = c1[0] + c1[1] + c1[2];
            tmp[1] = c1[1] - c1[2] - c1[3];
            tmp[2] = c1[4] + c1[5] + c1[6];
            tmp[3] = c1[5] - c1[6] - c1[7];
            tmp[4] = c1[8] + c1[9] + c1[10];
            tmp[5] = c1[9] - c1[10] - c1[11];
            tmp[6] = c1[12] + c1[13] + c1[14];
            tmp[7] = c1[13] - c1[14] - c1[15];

            dst[0*dstStride + 0] = tmp[0] + tmp[2] + tmp[4];
            dst[0*dstStride + 1] = tmp[1] + tmp[3] + tmp[5];
            dst[1*dstStride + 0] = tmp[2] - tmp[4] - tmp[6];
            dst[1*dstStride + 1] = tmp[3] - tmp[5] - tmp[7];
        }

        template <class T> void SetOutput1p(const T * src, size_t srcStride, T * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            T tmp[2 * 2];
            SetOutput1(src, srcStride, tmp, 2);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 2 + col];
        }

        template <class T> void SetOutput(const T * src, T * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t srcStride = ((dstHeight + 1) / 2) * ((dstWidth + 1) / 2)*dstChannels;
            size_t dstHeightFull = dstHeight / 2 * 2;
            size_t dstWidthFull = dstWidth / 2 * 2;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        SetOutput1(src++, srcStride, dst + row*dstWidth + col, dstWidth);
                    if (col < dstWidth)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, 2, dstWidth - col);
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, dstHeight - row, 2);
                    if (col < dstWidth)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                }
                dst += dstHeight * dstWidth;
            }
        }
    }

    namespace Winograd4x3p
    {
        template <class T> void SetFilter(const T * src, size_t size, T * dst)
        {
            const T r4 = T(1.0 / 4);
            const T r6 = T(1.0 / 6);
            const T r12 = T(1.0 / 12);
            const T r24 = T(1.0 / 24);
            for (size_t i = 0; i < size; ++i, src += 9, dst += 1)
            {
                T t[18];
                t[0] = r4 * src[0];
                t[1] = r4 * src[1];
                t[2] = r4 * src[2];
                t[3] = -r6 * (src[0] + src[3] + src[6]);
                t[4] = -r6 * (src[1] + src[4] + src[7]);
                t[5] = -r6 * (src[2] + src[5] + src[8]);
                t[6] = -r6 * (src[0] - src[3] + src[6]);
                t[7] = -r6 * (src[1] - src[4] + src[7]);
                t[8] = -r6 * (src[2] - src[5] + src[8]);
                t[9] = r24 * src[0] + r12 * src[3] + r6 * src[6];
                t[10] = r24 * src[1] + r12 * src[4] + r6 * src[7];
                t[11] = r24 * src[2] + r12 * src[5] + r6 * src[8];
                t[12] = r24 * src[0] - r12 * src[3] + r6 * src[6];
                t[13] = r24 * src[1] - r12 * src[4] + r6 * src[7];
                t[14] = r24 * src[2] - r12 * src[5] + r6 * src[8];
                t[15] = src[6];
                t[16] = src[7];
                t[17] = src[8];

                dst[size * 0] = r4 * t[0];
                dst[size * 1] = -r6 * (t[0] + t[1] + t[2]);
                dst[size * 2] = -r6 * (t[0] - t[1] + t[2]);
                dst[size * 3] = r24 * t[0] + r12 * t[1] + r6 * t[2];
                dst[size * 4] = r24 * t[0] - r12 * t[1] + r6 * t[2];
                dst[size * 5] = t[2];

                dst[size * 6] = r4 * t[3];
                dst[size * 7] = -r6 * (t[3] + t[4] + t[5]);
                dst[size * 8] = -r6 * (t[3] - t[4] + t[5]);
                dst[size * 9] = r24 * t[3] + r12 * t[4] + r6 * t[5];
                dst[size * 10] = r24 * t[3] - r12 * t[4] + r6 * t[5];
                dst[size * 11] = t[5];

                dst[size * 12] = r4 * t[6];
                dst[size * 13] = -r6 * (t[6] + t[7] + t[8]);
                dst[size * 14] = -r6 * (t[6] - t[7] + t[8]);
                dst[size * 15] = r24 * t[6] + r12 * t[7] + r6 * t[8];
                dst[size * 16] = r24 * t[6] - r12 * t[7] + r6 * t[8];
                dst[size * 17] = t[8];

                dst[size * 18] = r4 * t[9];
                dst[size * 19] = -r6 * (t[9] + t[10] + t[11]);
                dst[size * 20] = -r6 * (t[9] - t[10] + t[11]);
                dst[size * 21] = r24 * t[9] + r12 * t[10] + r6 * t[11];
                dst[size * 22] = r24 * t[9] - r12 * t[10] + r6 * t[11];
                dst[size * 23] = t[11];

                dst[size * 24] = r4 * t[12];
                dst[size * 25] = -r6 * (t[12] + t[13] + t[14]);
                dst[size * 26] = -r6 * (t[12] - t[13] + t[14]);
                dst[size * 27] = r24 * t[12] + r12 * t[13] + r6 * t[14];
                dst[size * 28] = r24 * t[12] - r12 * t[13] + r6 * t[14];
                dst[size * 29] = t[14];

                dst[size * 30] = r4 * t[15];
                dst[size * 31] = -r6 * (t[15] + t[16] + t[17]);
                dst[size * 32] = -r6 * (t[15] - t[16] + t[17]);
                dst[size * 33] = r24 * t[15] + r12 * t[16] + r6 * t[17];
                dst[size * 34] = r24 * t[15] - r12 * t[16] + r6 * t[17];
                dst[size * 35] = t[17];
            }
        }

        template <class T> void SetInput1(const T * src, size_t srcStride, T * dst, size_t dstStride)
        {
            T tmp1[36];
            tmp1[0] = src[0 * srcStride + 0];
            tmp1[1] = src[0 * srcStride + 1];
            tmp1[2] = src[0 * srcStride + 2];
            tmp1[3] = src[0 * srcStride + 3];
            tmp1[4] = src[0 * srcStride + 4];
            tmp1[5] = src[0 * srcStride + 5];
            tmp1[6] = src[1 * srcStride + 0];
            tmp1[7] = src[1 * srcStride + 1];
            tmp1[8] = src[1 * srcStride + 2];
            tmp1[9] = src[1 * srcStride + 3];
            tmp1[10] = src[1 * srcStride + 4];
            tmp1[11] = src[1 * srcStride + 5];
            tmp1[12] = src[2 * srcStride + 0];
            tmp1[13] = src[2 * srcStride + 1];
            tmp1[14] = src[2 * srcStride + 2];
            tmp1[15] = src[2 * srcStride + 3];
            tmp1[16] = src[2 * srcStride + 4];
            tmp1[17] = src[2 * srcStride + 5];
            tmp1[18] = src[3 * srcStride + 0];
            tmp1[19] = src[3 * srcStride + 1];
            tmp1[20] = src[3 * srcStride + 2];
            tmp1[21] = src[3 * srcStride + 3];
            tmp1[22] = src[3 * srcStride + 4];
            tmp1[23] = src[3 * srcStride + 5];
            tmp1[24] = src[4 * srcStride + 0];
            tmp1[25] = src[4 * srcStride + 1];
            tmp1[26] = src[4 * srcStride + 2];
            tmp1[27] = src[4 * srcStride + 3];
            tmp1[28] = src[4 * srcStride + 4];
            tmp1[29] = src[4 * srcStride + 5];
            tmp1[30] = src[5 * srcStride + 0];
            tmp1[31] = src[5 * srcStride + 1];
            tmp1[32] = src[5 * srcStride + 2];
            tmp1[33] = src[5 * srcStride + 3];
            tmp1[34] = src[5 * srcStride + 4];
            tmp1[35] = src[5 * srcStride + 5];

            T tmp2[36];
            tmp2[0] = 4 * tmp1[0] - 5 * tmp1[12] + tmp1[24];
            tmp2[1] = 4 * tmp1[1] - 5 * tmp1[13] + tmp1[25];
            tmp2[2] = 4 * tmp1[2] - 5 * tmp1[14] + tmp1[26];
            tmp2[3] = 4 * tmp1[3] - 5 * tmp1[15] + tmp1[27];
            tmp2[4] = 4 * tmp1[4] - 5 * tmp1[16] + tmp1[28];
            tmp2[5] = 4 * tmp1[5] - 5 * tmp1[17] + tmp1[29];
            tmp2[6] = -4 * tmp1[6] - 4 * tmp1[12] + tmp1[18] + tmp1[24];
            tmp2[7] = -4 * tmp1[7] - 4 * tmp1[13] + tmp1[19] + tmp1[25];
            tmp2[8] = -4 * tmp1[8] - 4 * tmp1[14] + tmp1[20] + tmp1[26];
            tmp2[9] = -4 * tmp1[9] - 4 * tmp1[15] + tmp1[21] + tmp1[27];
            tmp2[10] = -4 * tmp1[10] - 4 * tmp1[16] + tmp1[22] + tmp1[28];
            tmp2[11] = -4 * tmp1[11] - 4 * tmp1[17] + tmp1[23] + tmp1[29];
            tmp2[12] = 4 * tmp1[6] - 4 * tmp1[12] - tmp1[18] + tmp1[24];
            tmp2[13] = 4 * tmp1[7] - 4 * tmp1[13] - tmp1[19] + tmp1[25];
            tmp2[14] = 4 * tmp1[8] - 4 * tmp1[14] - tmp1[20] + tmp1[26];
            tmp2[15] = 4 * tmp1[9] - 4 * tmp1[15] - tmp1[21] + tmp1[27];
            tmp2[16] = 4 * tmp1[10] - 4 * tmp1[16] - tmp1[22] + tmp1[28];
            tmp2[17] = 4 * tmp1[11] - 4 * tmp1[17] - tmp1[23] + tmp1[29];
            tmp2[18] = -2 * tmp1[6] - tmp1[12] + 2 * tmp1[18] + tmp1[24];
            tmp2[19] = -2 * tmp1[7] - tmp1[13] + 2 * tmp1[19] + tmp1[25];
            tmp2[20] = -2 * tmp1[8] - tmp1[14] + 2 * tmp1[20] + tmp1[26];
            tmp2[21] = -2 * tmp1[9] - tmp1[15] + 2 * tmp1[21] + tmp1[27];
            tmp2[22] = -2 * tmp1[10] - tmp1[16] + 2 * tmp1[22] + tmp1[28];
            tmp2[23] = -2 * tmp1[11] - tmp1[17] + 2 * tmp1[23] + tmp1[29];
            tmp2[24] = 2 * tmp1[6] - tmp1[12] - 2 * tmp1[18] + tmp1[24];
            tmp2[25] = 2 * tmp1[7] - tmp1[13] - 2 * tmp1[19] + tmp1[25];
            tmp2[26] = 2 * tmp1[8] - tmp1[14] - 2 * tmp1[20] + tmp1[26];
            tmp2[27] = 2 * tmp1[9] - tmp1[15] - 2 * tmp1[21] + tmp1[27];
            tmp2[28] = 2 * tmp1[10] - tmp1[16] - 2 * tmp1[22] + tmp1[28];
            tmp2[29] = 2 * tmp1[11] - tmp1[17] - 2 * tmp1[23] + tmp1[29];
            tmp2[30] = 4 * tmp1[6] - 5 * tmp1[18] + tmp1[30];
            tmp2[31] = 4 * tmp1[7] - 5 * tmp1[19] + tmp1[31];
            tmp2[32] = 4 * tmp1[8] - 5 * tmp1[20] + tmp1[32];
            tmp2[33] = 4 * tmp1[9] - 5 * tmp1[21] + tmp1[33];
            tmp2[34] = 4 * tmp1[10] - 5 * tmp1[22] + tmp1[34];
            tmp2[35] = 4 * tmp1[11] - 5 * tmp1[23] + tmp1[35];

            dst[0 * dstStride] = tmp2[0] * 4 - tmp2[2] * 5 + tmp2[4];
            dst[1 * dstStride] = -tmp2[1] * 4 - tmp2[2] * 4 + tmp2[3] + tmp2[4];
            dst[2 * dstStride] = tmp2[1] * 4 - tmp2[2] * 4 - tmp2[3] + tmp2[4];
            dst[3 * dstStride] = -tmp2[1] * 2 - tmp2[2] + tmp2[3] * 2 + tmp2[4];
            dst[4 * dstStride] = tmp2[1] * 2 - tmp2[2] - tmp2[3] * 2 + tmp2[4];
            dst[5 * dstStride] = tmp2[1] * 4 - tmp2[3] * 5 + tmp2[5];
            dst[6 * dstStride] = tmp2[6] * 4 - tmp2[8] * 5 + tmp2[10];
            dst[7 * dstStride] = -tmp2[7] * 4 - tmp2[8] * 4 + tmp2[9] + tmp2[10];
            dst[8 * dstStride] = tmp2[7] * 4 - tmp2[8] * 4 - tmp2[9] + tmp2[10];
            dst[9 * dstStride] = -tmp2[7] * 2 - tmp2[8] + tmp2[9] * 2 + tmp2[10];
            dst[10 * dstStride] = tmp2[7] * 2 - tmp2[8] - tmp2[9] * 2 + tmp2[10];
            dst[11 * dstStride] = tmp2[7] * 4 - tmp2[9] * 5 + tmp2[11];
            dst[12 * dstStride] = tmp2[12] * 4 - tmp2[14] * 5 + tmp2[16];
            dst[13 * dstStride] = -tmp2[13] * 4 - tmp2[14] * 4 + tmp2[15] + tmp2[16];
            dst[14 * dstStride] = tmp2[13] * 4 - tmp2[14] * 4 - tmp2[15] + tmp2[16];
            dst[15 * dstStride] = -tmp2[13] * 2 - tmp2[14] + tmp2[15] * 2 + tmp2[16];
            dst[16 * dstStride] = tmp2[13] * 2 - tmp2[14] - tmp2[15] * 2 + tmp2[16];
            dst[17 * dstStride] = tmp2[13] * 4 - tmp2[15] * 5 + tmp2[17];
            dst[18 * dstStride] = tmp2[18] * 4 - tmp2[20] * 5 + tmp2[22];
            dst[19 * dstStride] = -tmp2[19] * 4 - tmp2[20] * 4 + tmp2[21] + tmp2[22];
            dst[20 * dstStride] = tmp2[19] * 4 - tmp2[20] * 4 - tmp2[21] + tmp2[22];
            dst[21 * dstStride] = -tmp2[19] * 2 - tmp2[20] + tmp2[21] * 2 + tmp2[22];
            dst[22 * dstStride] = tmp2[19] * 2 - tmp2[20] - tmp2[21] * 2 + tmp2[22];
            dst[23 * dstStride] = tmp2[19] * 4 - tmp2[21] * 5 + tmp2[23];
            dst[24 * dstStride] = tmp2[24] * 4 - tmp2[26] * 5 + tmp2[28];
            dst[25 * dstStride] = -tmp2[25] * 4 - tmp2[26] * 4 + tmp2[27] + tmp2[28];
            dst[26 * dstStride] = tmp2[25] * 4 - tmp2[26] * 4 - tmp2[27] + tmp2[28];
            dst[27 * dstStride] = -tmp2[25] * 2 - tmp2[26] + tmp2[27] * 2 + tmp2[28];
            dst[28 * dstStride] = tmp2[25] * 2 - tmp2[26] - tmp2[27] * 2 + tmp2[28];
            dst[29 * dstStride] = tmp2[25] * 4 - tmp2[27] * 5 + tmp2[29];
            dst[30 * dstStride] = tmp2[30] * 4 - tmp2[32] * 5 + tmp2[34];
            dst[31 * dstStride] = -tmp2[31] * 4 - tmp2[32] * 4 + tmp2[33] + tmp2[34];
            dst[32 * dstStride] = tmp2[31] * 4 - tmp2[32] * 4 - tmp2[33] + tmp2[34];
            dst[33 * dstStride] = -tmp2[31] * 2 - tmp2[32] + tmp2[33] * 2 + tmp2[34];
            dst[34 * dstStride] = tmp2[31] * 2 - tmp2[32] - tmp2[33] * 2 + tmp2[34];
            dst[35 * dstStride] = tmp2[31] * 4 - tmp2[33] * 5 + tmp2[35];
        }

        template <class T> void SetInput1p(const T * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, T * dst, size_t dstStride)
        {
            T tmp[6 * 6] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 6 + col] = src[row * srcStride + col];
            SetInput1(tmp, 6, dst, dstStride);
        }

        template <class T> void SetInput(const T * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, T * dst, bool pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstStride = ((dstHeight + 3) / 4) * ((dstWidth + 3) / 4)*srcChannels;
            size_t dstHeightFull = dstHeight / 4 * 4;
            size_t dstWidthFull = dstWidth / 4 * 4;
            size_t noseW = std::min<size_t>(6, dstWidth + 1);
            size_t noseH = std::min<size_t>(6, dstHeight + 1);
            size_t start = pad ? 4 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 4;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 4;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, 6, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 4)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, 6, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        SetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, 6, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 6, dst++, dstStride);
                    if (col < dstWidth)
                        SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth*srcHeight;
            }
        }

        template <class T> void SetOutput1(const T * src, size_t srcStride, T * dst, size_t dstStride)
        {
            T c1[36];
            c1[0] = src[0 * srcStride];
            c1[1] = src[1 * srcStride];
            c1[2] = src[2 * srcStride];
            c1[3] = src[3 * srcStride];
            c1[4] = src[4 * srcStride];
            c1[5] = src[5 * srcStride];
            c1[6] = src[6 * srcStride];
            c1[7] = src[7 * srcStride];
            c1[8] = src[8 * srcStride];
            c1[9] = src[9 * srcStride];
            c1[10] = src[10 * srcStride];
            c1[11] = src[11 * srcStride];
            c1[12] = src[12 * srcStride];
            c1[13] = src[13 * srcStride];
            c1[14] = src[14 * srcStride];
            c1[15] = src[15 * srcStride];
            c1[16] = src[16 * srcStride];
            c1[17] = src[17 * srcStride];
            c1[18] = src[18 * srcStride];
            c1[19] = src[19 * srcStride];
            c1[20] = src[20 * srcStride];
            c1[21] = src[21 * srcStride];
            c1[22] = src[22 * srcStride];
            c1[23] = src[23 * srcStride];
            c1[24] = src[24 * srcStride];
            c1[25] = src[25 * srcStride];
            c1[26] = src[26 * srcStride];
            c1[27] = src[27 * srcStride];
            c1[28] = src[28 * srcStride];
            c1[29] = src[29 * srcStride];
            c1[30] = src[30 * srcStride];
            c1[31] = src[31 * srcStride];
            c1[32] = src[32 * srcStride];
            c1[33] = src[33 * srcStride];
            c1[34] = src[34 * srcStride];
            c1[35] = src[35 * srcStride];

            T tmp[24];
            tmp[0] = c1[0] + c1[6] + c1[12] + c1[18] + c1[24];
            tmp[1] = c1[1] + c1[7] + c1[13] + c1[19] + c1[25];
            tmp[2] = c1[2] + c1[8] + c1[14] + c1[20] + c1[26];
            tmp[3] = c1[3] + c1[9] + c1[15] + c1[21] + c1[27];
            tmp[4] = c1[4] + c1[10] + c1[16] + c1[22] + c1[28];
            tmp[5] = c1[5] + c1[11] + c1[17] + c1[23] + c1[29];
            tmp[6] = c1[6] - c1[12] + 2 * c1[18] - 2 * c1[24];
            tmp[7] = c1[7] - c1[13] + 2 * c1[19] - 2 * c1[25];
            tmp[8] = c1[8] - c1[14] + 2 * c1[20] - 2 * c1[26];
            tmp[9] = c1[9] - c1[15] + 2 * c1[21] - 2 * c1[27];
            tmp[10] = c1[10] - c1[16] + 2 * c1[22] - 2 * c1[28];
            tmp[11] = c1[11] - c1[17] + 2 * c1[23] - 2 * c1[29];
            tmp[12] = c1[6] + c1[12] + 4 * c1[18] + 4 * c1[24];
            tmp[13] = c1[7] + c1[13] + 4 * c1[19] + 4 * c1[25];
            tmp[14] = c1[8] + c1[14] + 4 * c1[20] + 4 * c1[26];
            tmp[15] = c1[9] + c1[15] + 4 * c1[21] + 4 * c1[27];
            tmp[16] = c1[10] + c1[16] + 4 * c1[22] + 4 * c1[28];
            tmp[17] = c1[11] + c1[17] + 4 * c1[23] + 4 * c1[29];
            tmp[18] = c1[6] - c1[12] + 8 * c1[18] - 8 * c1[24] + c1[30];
            tmp[19] = c1[7] - c1[13] + 8 * c1[19] - 8 * c1[25] + c1[31];
            tmp[20] = c1[8] - c1[14] + 8 * c1[20] - 8 * c1[26] + c1[32];
            tmp[21] = c1[9] - c1[15] + 8 * c1[21] - 8 * c1[27] + c1[33];
            tmp[22] = c1[10] - c1[16] + 8 * c1[22] - 8 * c1[28] + c1[34];
            tmp[23] = c1[11] - c1[17] + 8 * c1[23] - 8 * c1[29] + c1[35];

            dst[0 * dstStride + 0] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
            dst[0 * dstStride + 1] = tmp[1] - tmp[2] + 2 * tmp[3] - 2 * tmp[4];
            dst[0 * dstStride + 2] = tmp[1] + tmp[2] + 4 * tmp[3] + 4 * tmp[4];
            dst[0 * dstStride + 3] = tmp[1] - tmp[2] + 8 * tmp[3] - 8 * tmp[4] + tmp[5];
            dst[1 * dstStride + 0] = tmp[6] + tmp[7] + tmp[8] + tmp[9] + tmp[10];
            dst[1 * dstStride + 1] = tmp[7] - tmp[8] + 2 * tmp[9] - 2 * tmp[10];
            dst[1 * dstStride + 2] = tmp[7] + tmp[8] + 4 * tmp[9] + 4 * tmp[10];
            dst[1 * dstStride + 3] = tmp[7] - tmp[8] + 8 * tmp[9] - 8 * tmp[10] + tmp[11];
            dst[2 * dstStride + 0] = tmp[12] + tmp[13] + tmp[14] + tmp[15] + tmp[16];
            dst[2 * dstStride + 1] = tmp[13] - tmp[14] + 2 * tmp[15] - 2 * tmp[16];
            dst[2 * dstStride + 2] = tmp[13] + tmp[14] + 4 * tmp[15] + 4 * tmp[16];
            dst[2 * dstStride + 3] = tmp[13] - tmp[14] + 8 * tmp[15] - 8 * tmp[16] + tmp[17];
            dst[3 * dstStride + 0] = tmp[18] + tmp[19] + tmp[20] + tmp[21] + tmp[22];
            dst[3 * dstStride + 1] = tmp[19] - tmp[20] + 2 * tmp[21] - 2 * tmp[22];
            dst[3 * dstStride + 2] = tmp[19] + tmp[20] + 4 * tmp[21] + 4 * tmp[22];
            dst[3 * dstStride + 3] = tmp[19] - tmp[20] + 8 * tmp[21] - 8 * tmp[22] + tmp[23];
        }

        template <class T> void SetOutput1p(const T * src, size_t srcStride, T * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            T tmp[4 * 4];
            SetOutput1(src, srcStride, tmp, 4);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 4 + col];
        }

        template <class T> void SetOutput(const T * src, T * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t srcStride = ((dstHeight + 3) / 4) * ((dstWidth + 3) / 4)*dstChannels;
            size_t dstHeightFull = dstHeight / 4 * 4;
            size_t dstWidthFull = dstWidth / 4 * 4;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 4)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        SetOutput1(src++, srcStride, dst + row*dstWidth + col, dstWidth);
                    if (col < dstWidth)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, 4, dstWidth - col);
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, dstHeight - row, 4);
                    if (col < dstWidth)
                        SetOutput1p(src++, srcStride, dst + row*dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                }
                dst += dstHeight * dstWidth;
            }
        }
    }

    template <class T> class Winograd
    {
    public:
        Winograd()
            : _type(Winograd::WinogradNone)
        {
        }

        void Init(Shape src, size_t dst, Shape kernel, Shape stride, Shape dilation, Shape pad, size_t group)
        {
            assert(src.size() == 3 && kernel.size() == 2 && stride.size() == 2 && dilation.size() == 2 && pad.size() == 4);
            if (stride[0] != 1 || stride[1] != 1 || dilation[0] != 1 || dilation[1] != 1)
                return;
            if (!((pad[0] == 0 && pad[1] == 0) || (pad[0] == 1 && pad[1] == 1)))
                return;
            if (group != 1)
                return;
            if (kernel[0] == 3 && kernel[1] == 3)
            {
                _srcC = src[0];
                _srcH = src[1];
                _srcW = src[2];
                _dstC = dst;
                if (pad[0] == 1 && pad[1] == 1)
                {
                    _pad = true;
                    _dstH = _srcH;
                    _dstW = _srcW;
                }
                else
                {
                    _pad = false;
                    _dstH = _srcH - 2;
                    _dstW = _srcW - 2;
                }

                if (src[0] < 16)
                    return;
                else

                if (1)
                {
                    _block = 2;
                    _count = 16;
                    _type = Winograd::Winograd2x3p;
                }
                else
                {
                    _block = 4;
                    _count = 36;
                    _type = Winograd::Winograd4x3p;
                }

                _tileH = (_dstH + _block - 1) / _block;
                _tileW = (_dstW + _block - 1) / _block;
                _strideF = _srcC * _dstC;
                _strideS = _srcC * _tileH * _tileW;
                _strideD = _dstC * _tileH * _tileW;
            }
        }

        bool Enable()
        {
            return _type != Winograd::WinogradNone;
        }

        void SetFilter(const T * src)
        {
            SYNET_PERF_FUNC();

            _filter.Reshape({ _count, _strideF }, 0);
            switch (_type)
            {
            case Winograd::Winograd2x3i:
                Winograd2x3i::SetFilter(src, _srcC*_dstC, _filter.CpuData());
                break;
            case Winograd::Winograd2x3p:
                Winograd2x3p::SetFilter(src, _srcC*_dstC, _filter.CpuData());
                break;
            case Winograd::Winograd4x3p:
                Winograd4x3p::SetFilter(src, _srcC*_dstC, _filter.CpuData());
                break;
            default:
                assert(0);
            }
        }

        size_t SrcBufSize()
        {
                return _strideS*_count;
        }

        size_t DstBufSize()
        {
            return _strideD*_count;
        }

        void Convolution(const T * src, T * srcBuf, T * dstBuf, T * dst)
        {
            SYNET_PERF_FUNC();

            SetInput(src, srcBuf);

            RunGemm(srcBuf, dstBuf);

            SetOutput(dstBuf, dst);
        }

    private:
        typedef Synet::Tensor<T> Tensor;

        enum WinogradType
        {
            WinogradNone,
            Winograd2x3i,
            Winograd2x3p,
            Winograd4x3p,
        } _type;  

        bool _pad;
        size_t _srcC, _srcW, _srcH, _dstC, _dstH, _dstW;
        size_t _count, _block, _tileH, _tileW, _strideF, _strideS, _strideD;
        size_t _group, _wStep, _dStep;
        
        Tensor _filter;
        const T * _weight;

        void SetInput(const T * src, T * dst)
        {
            SYNET_PERF_FUNC();

            switch (_type)
            {
            case Winograd::Winograd2x3i:
                Winograd2x3i::SetInput(src, _srcC, _srcH, _srcW, dst, _pad);
                break;
            case Winograd::Winograd2x3p:
                Winograd2x3p::SetInput(src, _srcC, _srcH, _srcW, dst, _pad);
                break;
            case Winograd::Winograd4x3p:
                Winograd4x3p::SetInput(src, _srcC, _srcH, _srcW, dst, _pad);
                break;
            default:
                assert(0);
            }
        }

        void RunGemm(const T * src, T * dst)
        {
            SYNET_PERF_FUNC();

            const size_t M = _dstC;
            const size_t N = _tileW*_tileH;
            const size_t K = _srcC;            
            switch (_type)
            {
            case Winograd::Winograd2x3i:
                Winograd2x3i::Gemm(M, N, K, _filter.CpuData(), src, dst);
                break;
            case Winograd::Winograd2x3p:
            case Winograd::Winograd4x3p:
            {

                for (size_t i = 0; i < _count; ++i)
                {
                    const T * a = _filter.CpuData() + i * _strideF;
                    const T * b = src + i * _strideS;
                    T * c = dst + i * _strideD;
                    CpuGemm(CblasNoTrans, CblasNoTrans, M, N, K, T(1.0), a, b, T(0.0), c);
                }
                break;
            }
            default:
                assert(0);
            }
        }

        void SetOutput(const T * src, T * dst)
        {
            SYNET_PERF_FUNC();

            switch (_type)
            {
            case Winograd::Winograd2x3i:
                Winograd2x3i::SetOutput(src, dst, _dstC, _dstH, _dstW);
                break;
            case Winograd::Winograd2x3p:
                Winograd2x3p::SetOutput(src, dst, _dstC, _dstH, _dstW);
                break;
            case Winograd::Winograd4x3p:
                Winograd4x3p::SetOutput(src, dst, _dstC, _dstH, _dstW);
                break;
            default:
                assert(0);
            }
        }
    };
}