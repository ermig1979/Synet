/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
    SYNET_INLINE Shape Shp()
    {
        return Shape();
    }

    SYNET_INLINE Shape Shp(size_t axis0)
    {
        return Shape({ axis0 });
    }

    SYNET_INLINE Shape Shp(size_t axis0, size_t axis1)
    {
        return Shape({ axis0, axis1 });
    }

    SYNET_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2)
    {
        return Shape({ axis0, axis1, axis2 });
    }

    SYNET_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3)
    {
        return Shape({ axis0, axis1, axis2, axis3 });
    }

    SYNET_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3, size_t axis4)
    {
        return Shape({ axis0, axis1, axis2, axis3, axis4 });
    }

    template<class T> SYNET_INLINE Shape Shp(const std::vector<T>& vec)
    {
        Shape shape(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
            shape[i] = (size_t)vec[i];
        return shape;
    }

    template<class T> SYNET_INLINE Shape Shp(const T* data, size_t size)
    {
        Shape shape(size);
        for (size_t i = 0; i < size; ++i)
            shape[i] = (size_t)data[i];
        return shape;
    }

    SYNET_INLINE String ToStr(const Shape& shape)
    {
        std::stringstream ss;
        ss << "{";
        for (size_t i = 0; i < shape.size(); ++i)
            ss << " " << (ptrdiff_t)shape[i];
        ss << " }";
        return ss.str();
    }

    SYNET_INLINE size_t TensorSize(const Shape& shape)
    {
        if (shape.empty())
            return 0;
        else
        {
            size_t size = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                size *= shape[i];
            return size;
        }
    }

    SYNET_INLINE size_t SignificantDimsCount(const Shape& shape)
    {
        size_t significant = 0;
        for (size_t i = 0; i < shape.size(); ++i)
            if (shape[i] > 1)
                significant++;
        return significant;
    }

    SYNET_INLINE Shape AtLeast2D(const Shape& shape)
    {
        return shape.size() > 1 ? shape : Shp(shape[0], shape[0]);
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE Longs Lng()
    {
        return Longs();
    }

    SYNET_INLINE Longs Lng(int64_t val0)
    {
        return Longs({ val0 });
    }

    SYNET_INLINE Longs Lng(int64_t val0, int64_t val1)
    {
        return Longs({ val0, val1 });
    }

    SYNET_INLINE Longs Lng(int64_t val0, int64_t val1, int64_t val2)
    {
        return Longs({ val0, val1, val2 });
    }

    SYNET_INLINE Longs Lng(int64_t val0, int64_t val1, int64_t val2, int64_t val3)
    {
        return Longs({ val0, val1, val2, val3 });
    }

    SYNET_INLINE Longs Lng(int64_t val0, int64_t val1, int64_t val2, int64_t val3, int64_t val4)
    {
        return Longs({ val0, val1, val2, val3, val4 });
    }
}