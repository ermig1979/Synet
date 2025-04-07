/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Utils/Shape.h"
#include "Synet/Params.h"

namespace Synet
{
    SYNET_INLINE bool IsCompatible(const Shape& a, const Shape& b)
    {
        for (size_t i = 0, n = std::max(a.size(), b.size()), a0 = n - a.size(), b0 = n - b.size(); i < n; ++i)
        {
            size_t ai = i < a0 ? 1 : a[i - a0];
            size_t bi = i < b0 ? 1 : b[i - b0];
            if (!(ai == bi || ai == 1 || bi == 1))
                return false;
        }
        return true;
    }

    SYNET_INLINE Shape OutputShape(const Shape& a, const Shape& b)
    {
        Shape d(std::max(a.size(), b.size()), 1);
        for (size_t i = 0, n = d.size(), a0 = n - a.size(), b0 = n - b.size(); i < n; ++i)
        {
            size_t ai = i < a0 ? 1 : a[i - a0];
            size_t bi = i < b0 ? 1 : b[i - b0];
            d[i] = std::max(ai, bi);
        }
        return d;
    }

    SYNET_INLINE Shape SourceSteps(const Shape& src, const Shape& dst)
    {
        Shape steps(dst.size(), 0);
        size_t step = 1;
        for (ptrdiff_t i = dst.size() - 1, s0 = dst.size() - src.size(); i >= 0; --i)
        {
            size_t si = i < s0 ? 1 : src[i - s0];
            steps[i] = si == 1 ? 0 : step;
            step *= si;
        }
        return steps;
    }

    SYNET_INLINE Shape FullSrcShape(const Shape& src, const Shape& dst)
    {
        Shape full(dst.size(), 1);
        for (size_t is = 0, id = dst.size() - src.size(); is < src.size(); is++, id++)
            full[id] = src[is];
        return full;
    }

    SYNET_INLINE int Relation(size_t a, size_t b, size_t d)
    {
        if (a < d)
            return -1;
        if (b < d)
            return 1;
        return 0;
    }

    SYNET_INLINE void CompactShapes(Shape& a, Shape& b, Shape& d)
    {
        Shape _a = FullSrcShape(a, d), _b = FullSrcShape(b, d), _d = d;
        a.resize(1), b.resize(1), d.resize(1);
        for (size_t i = 1; i < _d.size(); ++i)
        {
            if (Relation(_a[i - 1], _b[i - 1], _d[i - 1]) == Relation(_a[i], _b[i], _d[i]) || d.back() == 1 || _d[i] == 1)
            {
                a.back() *= _a[i];
                b.back() *= _b[i];
                d.back() *= _d[i];
            }
            else
            {
                a.push_back(_a[i]);
                b.push_back(_b[i]);
                d.push_back(_d[i]);
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    typedef void (*UniversalBinaryPtr)(const uint8_t* a, const Shape& aSteps, const uint8_t* b, const Shape& bSteps, uint8_t* dst, const Shape& dstShape);

    UniversalBinaryPtr GetUniversalBinary(BinaryOperationType op, TensorType type, size_t dim);
}