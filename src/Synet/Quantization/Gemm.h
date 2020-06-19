/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
    inline void CpuGemm8iNN(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t ldS, const int8_t* weight, size_t ldW, int32_t* dst, size_t ldD, bool overflow16i)
    {
        const size_t C2 = overflow16i ? C / 2 * 2 : 0;
        for (size_t i = 0; i < S; ++i)
        {
            for (size_t j = 0; j < D; ++j)
                dst[i * ldD + j] = 0;
            for (size_t k = 0, o = 0; k < K; k++)
            {
                size_t c = 0;
                for (; c < C2; c += 2, o += 2)
                {
                    int32_t s0 = src[i * ldS + o + 0];
                    int32_t s1 = src[i * ldS + o + 1];
                    const int8_t* w0 = weight + (o + 0) * ldW;
                    const int8_t* w1 = weight + (o + 1) * ldW;
                    int32_t* d = dst + i * ldD;
                    for (size_t j = 0; j < D; ++j)
                        d[j] += RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
                }
                for (; c < C; ++c, ++o)
                {
                    int32_t s0 = src[i * ldS + o];
                    const int8_t* w0 = weight + o * ldW;
                    int32_t* d = dst + i * ldD;
                    for (size_t j = 0; j < D; ++j)
                        d[j] += s0 * w0[j];
                }
            }
        }
    }

    inline void CpuGemm8iNN(size_t D, size_t S, size_t C, size_t K, const int8_t* weight, size_t ldW, const uint8_t* src, size_t ldS, int32_t* dst, size_t ldD, bool overflow16i)
    {
        const size_t C2 = overflow16i ? C / 2 * 2 : 0;
        for (size_t i = 0; i < D; ++i)
        {
            for (size_t j = 0; j < S; ++j)
                dst[i * ldD + j] = 0;
            size_t c = 0;
            for (; c < C2; c += 2)
            {
                for (size_t k = 0; k < K; k++)
                {
                    int32_t w0 = weight[i * ldW + (c + 0) * K + k];
                    int32_t w1 = weight[i * ldW + (c + 1) * K + k];
                    const uint8_t* s0 = src + ((c + 0) * K + k) * ldS;
                    const uint8_t* s1 = src + ((c + 1) * K + k) * ldS;
                    int32_t* d = dst + i * ldD;
                    for (size_t j = 0; j < S; ++j)
                        d[j] += RestrictRange(s0[j] * w0 + s1[j] * w1, SHRT_MIN, SHRT_MAX);
                }
            }
            for (; c < C; ++c)
            {
                for (size_t k = 0; k < K; k++)
                {
                    int32_t w0 = weight[i * ldW + (c + 0) * K + k];
                    const uint8_t* s0 = src + ((c + 0) * K + k) * ldS;
                    int32_t* d = dst + i * ldD;
                    for (size_t j = 0; j < S; ++j)
                        d[j] += s0[j] * w0;
                }
            }
        }
    }

    inline void CpuGemm8iNT(size_t M, size_t N, size_t K, const uint8_t* src, size_t ldS, const int8_t* weight, size_t ldW, int32_t* dst, size_t ldD, bool overflow16i)
    {
        const size_t K2 = overflow16i ? K / 2 * 2 : 0;
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                const int8_t * w = weight + j * ldW;
                size_t k = 0;
                int32_t d = 0;
                for (; k < K2; k += 2)
                    d += RestrictRange(int(src[k + 0]) * int(w[k + 0]) + int(src[k + 1]) * int(w[k + 1]), SHRT_MIN, SHRT_MAX);
                for (; k < K; ++k)
                    d += int(src[k + 0]) * int(w[k + 0]);
                dst[j] = d;
            }
            src += ldS;
            dst += ldD;
        }
    }
}