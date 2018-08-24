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
#include "Synet/Gemm.h"

namespace Synet
{
    template <class T> void WinogradTransformFilter4x3(const T * src, size_t srcChannels, size_t dstChannels, T * dst, size_t dstStride)
    {
        const T r4 = T(1.0 / 4);
        const T r6 = T(1.0 / 6);
        const T r12 = T(1.0 / 12);
        const T r24 = T(1.0 / 24);
        for (size_t m = 0; m < dstChannels; ++m)
        {
            for (size_t n = 0; n < srcChannels; ++n)
            {
                T c1[18];
                const T * F = src + 9*(n  + m * srcChannels);
                c1[0] = r4 * F[0];
                c1[1] = r4 * F[1];
                c1[2] = r4 * F[2];
                c1[3] = -r6 * (F[0] + F[3] + F[6]);
                c1[4] = -r6 * (F[1] + F[4] + F[7]);
                c1[5] = -r6 * (F[2] + F[5] + F[8]);
                c1[6] = -r6 * (F[0] - F[3] + F[6]);
                c1[7] = -r6 * (F[1] - F[4] + F[7]);
                c1[8] = -r6 * (F[2] - F[5] + F[8]);
                c1[9] = r24 * F[0] + r12 * F[3] + r6 * F[6];
                c1[10] = r24 * F[1] + r12 * F[4] + r6 * F[7];
                c1[11] = r24 * F[2] + r12 * F[5] + r6 * F[8];
                c1[12] = r24 * F[0] - r12 * F[3] + r6 * F[6];
                c1[13] = r24 * F[1] - r12 * F[4] + r6 * F[7];
                c1[14] = r24 * F[2] - r12 * F[5] + r6 * F[8];
                c1[15] = F[6];
                c1[16] = F[7];
                c1[17] = F[8];

                T c2[36];
                c2[0] = r4 * c1[0];
                c2[1] = -r6 * (c1[0] + c1[1] + c1[2]);
                c2[2] = -r6 * (c1[0] - c1[1] + c1[2]);
                c2[3] = r24 * c1[0] + r12 * c1[1] + r6 * c1[2];
                c2[4] = r24 * c1[0] - r12 * c1[1] + r6 * c1[2];
                c2[5] = c1[2];

                c2[6] = r4 * c1[3];
                c2[7] = -r6 * (c1[3] + c1[4] + c1[5]);
                c2[8] = -r6 * (c1[3] - c1[4] + c1[5]);
                c2[9] = r24 * c1[3] + r12 * c1[4] + r6 * c1[5];
                c2[10] = r24 * c1[3] - r12 * c1[4] + r6 * c1[5];
                c2[11] = c1[5];

                c2[12] = r4 * c1[6];
                c2[13] = -r6 * (c1[6] + c1[7] + c1[8]);
                c2[14] = -r6 * (c1[6] - c1[7] + c1[8]);
                c2[15] = r24 * c1[6] + r12 * c1[7] + r6 * c1[8];
                c2[16] = r24 * c1[6] - r12 * c1[7] + r6 * c1[8];
                c2[17] = c1[8];

                c2[18] = r4 * c1[9];
                c2[19] = -r6 * (c1[9] + c1[10] + c1[11]);
                c2[20] = -r6 * (c1[9] - c1[10] + c1[11]);
                c2[21] = r24 * c1[9] + r12 * c1[10] + r6 * c1[11];
                c2[22] = r24 * c1[9] - r12 * c1[10] + r6 * c1[11];
                c2[23] = c1[11];

                c2[24] = r4 * c1[12];
                c2[25] = -r6 * (c1[12] + c1[13] + c1[14]);
                c2[26] = -r6 * (c1[12] - c1[13] + c1[14]);
                c2[27] = r24 * c1[12] + r12 * c1[13] + r6 * c1[14];
                c2[28] = r24 * c1[12] - r12 * c1[13] + r6 * c1[14];
                c2[29] = c1[14];

                c2[30] = r4 * c1[15];
                c2[31] = -r6 * (c1[15] + c1[16] + c1[17]);
                c2[32] = -r6 * (c1[15] - c1[16] + c1[17]);
                c2[33] = r24 * c1[15] + r12 * c1[16] + r6 * c1[17];
                c2[34] = r24 * c1[15] - r12 * c1[16] + r6 * c1[17];
                c2[35] = c1[17];

                for (size_t x = 0; x < 36; ++x)
                    dst[x * dstStride + m * srcChannels + n] = c2[x];
            }
        }
    }

    template <class T> void WinogradGetTiles4x3x1(const T * src, size_t srcStride, size_t x, size_t y, T * dst, size_t dstStride, size_t *counter)
    {
        size_t coter = *counter;
        T tmp1[36];

        tmp1[0] = src[(x + 0) * srcStride + y + 0];
        tmp1[1] = src[(x + 0) * srcStride + y + 1];
        tmp1[2] = src[(x + 0) * srcStride + y + 2];
        tmp1[3] = src[(x + 0) * srcStride + y + 3];
        tmp1[4] = src[(x + 0) * srcStride + y + 4];
        tmp1[5] = src[(x + 0) * srcStride + y + 5];
        tmp1[6] = src[(x + 1) * srcStride + y + 0];
        tmp1[7] = src[(x + 1) * srcStride + y + 1];
        tmp1[8] = src[(x + 1) * srcStride + y + 2];
        tmp1[9] = src[(x + 1) * srcStride + y + 3];
        tmp1[10] = src[(x + 1) * srcStride + y + 4];
        tmp1[11] = src[(x + 1) * srcStride + y + 5];
        tmp1[12] = src[(x + 2) * srcStride + y + 0];
        tmp1[13] = src[(x + 2) * srcStride + y + 1];
        tmp1[14] = src[(x + 2) * srcStride + y + 2];
        tmp1[15] = src[(x + 2) * srcStride + y + 3];
        tmp1[16] = src[(x + 2) * srcStride + y + 4];
        tmp1[17] = src[(x + 2) * srcStride + y + 5];
        tmp1[18] = src[(x + 3) * srcStride + y + 0];
        tmp1[19] = src[(x + 3) * srcStride + y + 1];
        tmp1[20] = src[(x + 3) * srcStride + y + 2];
        tmp1[21] = src[(x + 3) * srcStride + y + 3];
        tmp1[22] = src[(x + 3) * srcStride + y + 4];
        tmp1[23] = src[(x + 3) * srcStride + y + 5];
        tmp1[24] = src[(x + 4) * srcStride + y + 0];
        tmp1[25] = src[(x + 4) * srcStride + y + 1];
        tmp1[26] = src[(x + 4) * srcStride + y + 2];
        tmp1[27] = src[(x + 4) * srcStride + y + 3];
        tmp1[28] = src[(x + 4) * srcStride + y + 4];
        tmp1[29] = src[(x + 4) * srcStride + y + 5];
        tmp1[30] = src[(x + 5) * srcStride + y + 0];
        tmp1[31] = src[(x + 5) * srcStride + y + 1];
        tmp1[32] = src[(x + 5) * srcStride + y + 2];
        tmp1[33] = src[(x + 5) * srcStride + y + 3];
        tmp1[34] = src[(x + 5) * srcStride + y + 4];
        tmp1[35] = src[(x + 5) * srcStride + y + 5];

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

        dst[0 * dstStride + coter] = tmp2[0] * 4 - tmp2[2] * 5 + tmp2[4];
        dst[1 * dstStride + coter] = -tmp2[1] * 4 - tmp2[2] * 4 + tmp2[3] + tmp2[4];
        dst[2 * dstStride + coter] = tmp2[1] * 4 - tmp2[2] * 4 - tmp2[3] + tmp2[4];
        dst[3 * dstStride + coter] = -tmp2[1] * 2 - tmp2[2] + tmp2[3] * 2 + tmp2[4];
        dst[4 * dstStride + coter] = tmp2[1] * 2 - tmp2[2] - tmp2[3] * 2 + tmp2[4];
        dst[5 * dstStride + coter] = tmp2[1] * 4 - tmp2[3] * 5 + tmp2[5];
        dst[6 * dstStride + coter] = tmp2[6] * 4 - tmp2[8] * 5 + tmp2[10];
        dst[7 * dstStride + coter] = -tmp2[7] * 4 - tmp2[8] * 4 + tmp2[9] + tmp2[10];
        dst[8 * dstStride + coter] = tmp2[7] * 4 - tmp2[8] * 4 - tmp2[9] + tmp2[10];
        dst[9 * dstStride + coter] = -tmp2[7] * 2 - tmp2[8] + tmp2[9] * 2 + tmp2[10];
        dst[10 * dstStride + coter] = tmp2[7] * 2 - tmp2[8] - tmp2[9] * 2 + tmp2[10];
        dst[11 * dstStride + coter] = tmp2[7] * 4 - tmp2[9] * 5 + tmp2[11];
        dst[12 * dstStride + coter] = tmp2[12] * 4 - tmp2[14] * 5 + tmp2[16];
        dst[13 * dstStride + coter] = -tmp2[13] * 4 - tmp2[14] * 4 + tmp2[15] + tmp2[16];
        dst[14 * dstStride + coter] = tmp2[13] * 4 - tmp2[14] * 4 - tmp2[15] + tmp2[16];
        dst[15 * dstStride + coter] = -tmp2[13] * 2 - tmp2[14] + tmp2[15] * 2 + tmp2[16];
        dst[16 * dstStride + coter] = tmp2[13] * 2 - tmp2[14] - tmp2[15] * 2 + tmp2[16];
        dst[17 * dstStride + coter] = tmp2[13] * 4 - tmp2[15] * 5 + tmp2[17];
        dst[18 * dstStride + coter] = tmp2[18] * 4 - tmp2[20] * 5 + tmp2[22];
        dst[19 * dstStride + coter] = -tmp2[19] * 4 - tmp2[20] * 4 + tmp2[21] + tmp2[22];
        dst[20 * dstStride + coter] = tmp2[19] * 4 - tmp2[20] * 4 - tmp2[21] + tmp2[22];
        dst[21 * dstStride + coter] = -tmp2[19] * 2 - tmp2[20] + tmp2[21] * 2 + tmp2[22];
        dst[22 * dstStride + coter] = tmp2[19] * 2 - tmp2[20] - tmp2[21] * 2 + tmp2[22];
        dst[23 * dstStride + coter] = tmp2[19] * 4 - tmp2[21] * 5 + tmp2[23];
        dst[24 * dstStride + coter] = tmp2[24] * 4 - tmp2[26] * 5 + tmp2[28];
        dst[25 * dstStride + coter] = -tmp2[25] * 4 - tmp2[26] * 4 + tmp2[27] + tmp2[28];
        dst[26 * dstStride + coter] = tmp2[25] * 4 - tmp2[26] * 4 - tmp2[27] + tmp2[28];
        dst[27 * dstStride + coter] = -tmp2[25] * 2 - tmp2[26] + tmp2[27] * 2 + tmp2[28];
        dst[28 * dstStride + coter] = tmp2[25] * 2 - tmp2[26] - tmp2[27] * 2 + tmp2[28];
        dst[29 * dstStride + coter] = tmp2[25] * 4 - tmp2[27] * 5 + tmp2[29];
        dst[30 * dstStride + coter] = tmp2[30] * 4 - tmp2[32] * 5 + tmp2[34];
        dst[31 * dstStride + coter] = -tmp2[31] * 4 - tmp2[32] * 4 + tmp2[33] + tmp2[34];
        dst[32 * dstStride + coter] = tmp2[31] * 4 - tmp2[32] * 4 - tmp2[33] + tmp2[34];
        dst[33 * dstStride + coter] = -tmp2[31] * 2 - tmp2[32] + tmp2[33] * 2 + tmp2[34];
        dst[34 * dstStride + coter] = tmp2[31] * 2 - tmp2[32] - tmp2[33] * 2 + tmp2[34];
        dst[35 * dstStride + coter] = tmp2[31] * 4 - tmp2[33] * 5 + tmp2[35];
        (*counter)++;
    }

    template <class T> void WinogradGetTiles4x3x1pad(const T * src, size_t srcStride, size_t x, size_t y, size_t lenX, size_t lenY, T * dst, size_t dstStride, size_t *counter)
    {
        float tmp[6 * 6] = { 0 };
        if (2 == lenX || 2 == lenY)
            return;
        int i, j;
        for (i = 0; i < lenX; ++i)
        {
            for (j = 0; j < lenY; ++j)
            {
                tmp[i * 6 + j] = src[(x + i) * srcStride + y + j];
            }
            for (; j < 6; ++j)
            {
                tmp[i * 6 + j] = 0;
            }
        }
        for (; i < 6; ++i)
        {
            for (j = 0; j < 6; ++j)
            {
                tmp[i * 6 + j] = 0;
            }
        }
        WinogradGetTiles4x3x1(tmp, 6, 0, 0, dst, dstStride, counter);
    }

    template <class T> void WinogradGetTiles4x3(const T * src, size_t srcStride, size_t irows, size_t icols, size_t sizeI, size_t C, T * dst, size_t dstStride, size_t N, size_t ntiles, size_t M)
    {
        size_t outHeight = irows - 2;
        size_t outWidth = icols - 2;
        size_t fullOutHeight = outHeight / 4 * 4;
        size_t fullOutWidth = outWidth / 4 * 4;

        for (size_t t = 0; t < N * C; ++t)
        {
            size_t i, j;

            size_t t1 = t / (C * M);
            size_t t2 = (t % (C * M)) / M;
            size_t t3 = t % M;

            const T * data = src + (t1 * M * C + t3 * C + t2) * sizeI;
            size_t tile_count = t * ntiles;

            for (i = 0; i < fullOutHeight; i += 4)
            {
                for (j = 0; j < fullOutWidth; j += 4)
                    WinogradGetTiles4x3x1(data, srcStride, i, j, dst, dstStride, &tile_count);
                WinogradGetTiles4x3x1pad(data, srcStride, i, j, 6, outWidth - fullOutWidth + 2, dst, dstStride, &tile_count);
            }
            for (j = 0; j < fullOutWidth; j += 4)
                WinogradGetTiles4x3x1pad(data, srcStride, i, j, outHeight - fullOutHeight + 2, 6, dst, dstStride, &tile_count);
            WinogradGetTiles4x3x1pad(data, srcStride, i, j, outHeight - fullOutHeight + 2, outWidth - fullOutWidth + 2, dst, dstStride, &tile_count);
        }
    }

    template <class T> void WinogradGemm4x3(const T * src, size_t srcStride, size_t irows, size_t icols, const T * filter, size_t filterStride, size_t frows, size_t fcols, T * dst, size_t dstStride, size_t batch)
    {
        const T alpha = T(1.0);
        const T beta = T(0.0);
        const size_t lda = fcols;
        const size_t ldb = icols;
        const size_t ldc = icols;
        const size_t M = frows;
        const size_t N = icols;
        const size_t K = fcols;
        for (size_t i = 0; i < 36; ++i)
        {
            for (size_t t = 0; t < batch; ++t)
            {
                const T * a = filter + i * filterStride;
                const T * b = src + i * srcStride + t * irows * icols;
                T * c = dst + i * dstStride + t * irows * fcols;
                CpuGemm(M, N, K, &alpha, a, lda, b, ldb, &beta, c, ldc);
            }
        }
    }
}