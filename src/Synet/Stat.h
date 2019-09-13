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
#include "Synet/Params.h"
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

        Stat(const StatisticParam & param)
            : negative(false)
            , channels(true)
        {
            name = param.name();
            min = param.min();
            max = param.max();
            assert(min.size() == max.size());
            for (size_t i = 0; i < min.size(); ++i)
            {
                assert(min[i] <= max[i]);
                if (min[i] < 0.0f)
                    negative = true;
            }
        }

        void Unify()
        {
            float _max = max[0], _min = min[0];
            for (size_t i = 1; i < min.size(); ++i)
            {
                _min = std::min(_min, min[i]);
                _max = std::max(_max, max[i]);
            }
            for (size_t i = 0; i < min.size(); ++i)
            {
                min[i] = _min;
                max[i] = _max;
            }
            channels = false;
        }

        void UnifyAs(const Stat & stat)
        {
            assert(min.size() == stat.min.size() && !stat.channels);
            for (size_t i = 0; i < min.size(); ++i)
            {
                assert(min[i] >= stat.min[0] && max[i] <= stat.max[0]);
                min[i] = stat.min[0];
                max[i] = stat.max[0];
            }
            //negative = stat.negative;
            channels = false;
        }

        void Init8u()
        {
            size_t n = min.size();
            if (zero8u.size() == n)
                return;
            scale32fTo8u.resize(n);
            shift32fTo8u.resize(n);
            scale8uTo32f.resize(n);
            shift8uTo32f.resize(n);
            zero8u.resize(n);

            if (SYNET_INT8_IE_COMPATIBLE)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    float absMax = ::fmax(::fabs(min[i]), ::fabs(max[i]));
                    float invScale = absMax / (negative ? 127.0f : 255.0f);
                    if (fabs(invScale) < 1e-7)
                        invScale = 1.0f;
                    zero8u[i] = (negative ? 128 : 0);
                    scale32fTo8u[i] = 1.0f / invScale;
                    scale8uTo32f[i] = invScale;
                    shift32fTo8u[i] = float(zero8u[i]);
                    shift8uTo32f[i] = -float(zero8u[i]) / invScale;
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