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
#include "Synet/Utils/Math.h"

namespace Synet
{
    struct Stat
    {
        String name;
        Floats min;
        Floats max;

        Floats scale32fTo8u, shift32fTo8u, scale8uTo32f, shift8uTo32f;
        Bytes zero8u;
        bool negative, channels;

        Stat()
            : negative(true)
            , channels(true)
        {
        }

        void Init8u()
        {
            assert(min.size() == max.size());
            size_t n = min.size();
            if (zero8u.size() == n)
                return;
            scale32fTo8u.resize(n);
            shift32fTo8u.resize(n);
            scale8uTo32f.resize(n);
            shift8uTo32f.resize(n);
            zero8u.resize(n);

            negative = false;
            for (size_t i = 0; i < n && !negative; ++i)
                if (min[i] < 0.0f)
                    negative = true;

            if (SYNET_INT8_IE_COMPATIBLE)
            {
                float absMax = 0;
                if (!channels)
                {
                    for (size_t i = 0; i < n; ++i)
                        absMax = std::max(absMax, std::max(::abs(min[i]), ::abs(max[i])));
                }
                for (size_t i = 0; i < n; ++i)
                {
                    float _abs = channels ? std::max(::abs(min[i]), ::abs(max[i])) : absMax;
                    float scale = (negative ? 127.0f : 255.0f) / _abs;
                    zero8u[i] = (negative ? 128 : 0);
                    scale32fTo8u[i] = scale;
                    scale8uTo32f[i] = 1.0f / scale;
                    shift32fTo8u[i] = float(zero8u[i]);
                    shift8uTo32f[i] = -float(zero8u[i]) * scale;
                }
            }
            else
            {
                for (size_t i = 0; i < n; ++i)
                {
                    float _min = std::min(0.0f, min[i]);
                    float _max = std::max(0.0f, max[i]);
                    float scale = 255.0f / (_max - _min);
                    zero8u[i] = (uint8_t)Quantize(0.0f - _min * scale);
                    scale32fTo8u[i] = scale;
                    scale8uTo32f[i] = 1.0f / scale;
                    if (SYNET_INT8_SAFE_ZERO)
                    {
                        shift32fTo8u[i] = float(zero8u[i]);
                        shift8uTo32f[i] = -float(zero8u[i]) * scale;
                    }
                    else
                    {
                        shift32fTo8u[i] = -_min * scale;
                        shift8uTo32f[i] = _min;
                    }
                }
            }
        }
    };

    typedef Stat * StatPtr;
    typedef std::vector<StatPtr> StatPtrs;
    typedef std::shared_ptr<Stat> StatSharedPtr;
    typedef std::vector<StatSharedPtr> StatSharedPtrs;
}