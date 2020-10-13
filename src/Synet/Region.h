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

#include "Common.h"

namespace Synet
{
    namespace Detail
    {
        template<class T> SYNET_INLINE T Overlap(T x1, T w1, T x2, T w2)
        {
            T l1 = x1 - w1 / 2;
            T l2 = x2 - w2 / 2;
            T left = l1 > l2 ? l1 : l2;
            T r1 = x1 + w1 / 2;
            T r2 = x2 + w2 / 2;
            T right = r1 < r2 ? r1 : r2;
            return right - left;
        }
    }

    template <class T> struct Region
    {
        T x, y, w, h, prob;
        size_t id;
    };

    template<class T> SYNET_INLINE T Intersection(const Region<T>& a, const Region<T>& b)
    {
        T w = Detail::Overlap(a.x, a.w, b.x, b.w);
        T h = Detail::Overlap(a.y, a.h, b.y, b.h);
        return (w < 0 || h < 0) ? 0 : w * h;
    }

    template<class T> SYNET_INLINE T Union(const Region<T>& a, const Region<T>& b)
    {
        T i = Intersection(a, b);
        return a.w * a.h + b.w * b.h - i;
    }

    template<class T> SYNET_INLINE T Overlap(const Region<T>& a, const Region<T>& b)
    {
        return Intersection(a, b) / Union(a, b);
    }
}