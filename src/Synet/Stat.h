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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Utils/Math.h"
#include "Synet/Quantization/Const.h"

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
                _min = Min(_min, min[i]);
                _max = Max(_max, max[i]);
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
            assert(min.size() == stat.max.size() && !stat.channels);
            for (size_t i = 0; i < min.size(); ++i)
            {
                assert(min[i] >= stat.min[0] && max[i] <= stat.max[0]);
                min[i] = stat.min[0];
                max[i] = stat.max[0];
            }
            channels = false;
        }

        void UnifyAs(const Stat * const * stats, size_t size)
        {
            size_t total = 0;
            for (size_t s = 0; s < size; ++s)
            {
                total += stats[s]->min.size();
                assert(!stats[s]->channels);
            }
            assert(min.size() == total);
            for (size_t s = 0, o = 0; s < size; ++s)
            {
                for (size_t i = 0; i < stats[s]->min.size(); ++i, ++o)
                {
                    assert(min[o] >= stats[s]->min[0] && max[o] <= stats[s]->max[0]);
                    min[o] = stats[s]->min[0];
                    max[o] = stats[s]->max[0];
                }
            }
            channels = false;
        }

        void Init8u(QuantizationMethod method)
        {
            size_t n = min.size();
            if (zero8u.size() == n)
                return;
            scale32fTo8u.resize(n);
            shift32fTo8u.resize(n);
            scale8uTo32f.resize(n);
            shift8uTo32f.resize(n);
            zero8u.resize(n);

            if (method == QuantizationMethodIECompatible)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    float absMax = ::fmax(::fabs(min[i]), ::fabs(max[i]));
                    float invScale = absMax / (negative ? QUANT_IE_COMP_SRC_I8_MAX : QUANT_IE_COMP_SRC_U8_MAX);
                    if (fabs(invScale) < 1e-7)
                        invScale = 1.0f;
                    zero8u[i] = (negative ? -QUANT_IE_COMP_SRC_I8_MIN : QUANT_IE_COMP_SRC_U8_MIN);
                    scale32fTo8u[i] = float(1.0 / invScale);
                    scale8uTo32f[i] = invScale;
                    shift32fTo8u[i] = float(zero8u[i]);
                    shift8uTo32f[i] = -float(zero8u[i]) * invScale;
                }
            }
            else if (method == QuantizationMethodSymmetricNarrowed)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    float absMax = ::fmax(::fabs(min[i]), ::fabs(max[i]));
                    float invScale = absMax / (negative ? QUANT_SYMM_NARR_SRC_I8_MAX : QUANT_SYMM_NARR_SRC_U8_MAX);
                    if (fabs(invScale) < 1e-7)
                        invScale = 1.0f;
                    zero8u[i] = (negative ? -QUANT_SYMM_NARR_SRC_I8_MIN : QUANT_SYMM_NARR_SRC_U8_MIN);
                    scale32fTo8u[i] = float(1.0 / invScale);
                    scale8uTo32f[i] = invScale;
                    shift32fTo8u[i] = float(zero8u[i]);
                    shift8uTo32f[i] = -float(zero8u[i]) * invScale;
                }
            }
            else
            {
                for (size_t i = 0; i < n; ++i)
                {
                    float _min = Min(0.0f, min[i]);
                    float _max = Max(0.0f, max[i]);
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

        SYNET_INLINE size_t MemoryUsage() const
        {
            return (min.size() + max.size() + scale32fTo8u.size() + shift32fTo8u.size() + 
                scale8uTo32f.size() + shift8uTo32f.size())*sizeof(float) + zero8u.size();
        }
    };

    typedef Stat * StatPtr;
    typedef std::vector<StatPtr> StatPtrs;
    typedef std::shared_ptr<Stat> StatSharedPtr;
    typedef std::vector<StatSharedPtr> StatSharedPtrs;
}