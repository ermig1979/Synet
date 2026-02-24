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

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <cfloat>

#include <assert.h>

#if !defined(SYNET_BF16_PRINT_DISABLE)
#include "Synet/Quantization/Bf16.h"
#endif

namespace Synet
{
    typedef std::vector<size_t> Shape;
    typedef std::string String;
    typedef std::vector<String> Strings;

    namespace Detail
    {
        template<class T> String TypeID();
        template<> inline String TypeID<float>() { return "32f"; }
        template<> inline String TypeID<int32_t>() { return "32i"; }
        template<> inline String TypeID<uint8_t>() { return "8u"; }
        template<> inline String TypeID<int8_t>() { return "8i"; }
        template<> inline String TypeID<int64_t>() { return "64i"; }
        template<> inline String TypeID<uint64_t>() { return "64u"; }
        template<> inline String TypeID<bool>() { return "Bool"; }
        template<> inline String TypeID<uint16_t>() { return "16b"; }

        template<class T> void DebugPrint(std::ostream& os, T value, size_t precision)
        {
            os << (int)value;
        }

        template<> inline void DebugPrint(std::ostream& os, float value, size_t precision)
        {
            os << std::fixed << std::setprecision(precision) << value;
        }

        template<> inline void DebugPrint(std::ostream& os, int64_t value, size_t precision)
        {
            os << value;
        }

#if !defined(SYNET_BF16_PRINT_DISABLE)
        template<> inline void DebugPrint(std::ostream& os, uint16_t value, size_t precision)
        {
            os << std::fixed << std::setprecision(precision) << BFloat16ToFloat32(value);
        }
#endif

        template<class T> std::ostream& DebugPrint(std::ostream& os, T value, size_t precision, size_t padding)
        {
            std::stringstream ss;
            DebugPrint(ss, value, precision);
            for (size_t i = ss.str().size(); i < padding; ++i)
                os << " ";
            os << ss.str();
            return os;
        }

        inline String DebugPrint(const Shape & shape)
        {
            std::stringstream ss;
            ss << "{ ";
            for (size_t i = 0; i < shape.size(); ++i)
                ss << (ptrdiff_t)shape[i] << " ";
            ss << "}";
            return ss.str();
        }

        inline size_t Size(const Shape& shape)
        {
            size_t size = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                size *= shape[i];
            return size;
        }

        inline size_t Offset(const Shape& shape, const Shape& index)
        {
            assert(shape.size() == index.size());
            size_t offset = 0;
            for (size_t axis = 0; axis < shape.size(); ++axis)
            {
                assert(shape[axis] > 0);
                assert(index[axis] < shape[axis]);
                offset *= shape[axis];
                offset += index[axis];
            }
            return offset;
        }

        template<class T> size_t DebugPrintPadding(const T* data, const Shape& shape, size_t precision)
        {
            size_t size = Size(shape);
            if (size)
            {
                T min = data[0];
                T max = data[0];
                for (size_t i = 1; i < size; ++i)
                {
                    min = std::min(min, data[i]);
                    max = std::max(max, data[i]);
                }
                std::stringstream ssMin, ssMax;
                DebugPrint(ssMin, min, precision);
                DebugPrint(ssMax, max, precision);
                return std::max(ssMin.str().size(), ssMax.str().size());
            }
            return 0;
        }

        template <class T> static void DebugPrint(std::ostream& os, const T* data, const Shape & shape,
            const Shape& firsts, const Shape& lasts, const Strings & separators, Shape index, size_t order, size_t precision, size_t padding)
        {
            if (order == shape.size())
            {
                DebugPrint(os, data[Offset(shape, index)], precision, padding);
                return;
            }
            if (firsts[order] + lasts[order] < shape[order])
            {
                size_t lo = firsts[order], hi = shape[order] - lasts[order];
                for (index[order] = 0; index[order] < lo; ++index[order])
                {
                    DebugPrint(os, data, shape, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
                os << "..." << separators[order];
                for (index[order] = hi; index[order] < shape[order]; ++index[order])
                {
                    DebugPrint(os, data, shape, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
            }
            else
            {
                for (index[order] = 0; index[order] < shape[order]; ++index[order])
                {
                    DebugPrint(os, data, shape, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
            }
        }

        template <class T> static void PrintDiagnostic(std::ostream& os, const T* data, size_t size, size_t precision)
        {
            if (data == NULL || size == 0 || !(std::is_same<T, float>::value || std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value))
                return;
            if (std::is_same<T, float>::value || std::is_same<T, uint16_t>::value)
            {
                float max = -FLT_MAX, min = FLT_MAX;
                bool hasNaN = false;
                if (std::is_same<T, float>::value)
                {
                    float* ptr = (float*)data;
                    for (size_t i = 0; i < size; ++i)
                    {
                        float val = ptr[i];
                        max = std::max(max, val);
                        min = std::min(min, val);
                        hasNaN = hasNaN || std::isnan(val);
                    }
                }
#if !defined(SYNET_BF16_PRINT_DISABLE)
                if (std::is_same<T, uint16_t>::value)
                {
                    uint16_t* ptr = (uint16_t*)data;
                    for (size_t i = 0; i < size; ++i)
                    {
                        float val = BFloat16ToFloat32(ptr[i]);
                        max = std::max(max, val);
                        min = std::min(min, val);
                        hasNaN = hasNaN || std::isnan(val);
                    }
                }
#endif
                os << std::fixed << std::setprecision(precision);
                os << " { " << min << " .. " << max << " }";
                if (max > float(INT_MAX) || min < float(INT_MIN) || std::isnan(min) || std::isnan(max) || hasNaN)
                    os << " warning!";
            }
            if (std::is_same<T, uint8_t>::value)
            {
                int64_t max = (int64_t)std::numeric_limits<T>::min(), min = (int64_t)std::numeric_limits<T>::max();
                if (std::is_same<T, uint8_t>::value)
                {
                    uint8_t* ptr = (uint8_t*)data;
                    for (size_t i = 0; i < size; ++i)
                    {
                        max = std::max<int64_t>(max, ptr[i]);
                        min = std::min<int64_t>(min, ptr[i]);
                    }
                }
                os << " { " << min << " .. " << max << " }";
            }
        }
    }

    template<class T> inline void DebugPrint(std::ostream & os, const T * data, const Shape& shape, const std::string & name, bool cnst, size_t first = 6, size_t last = 2, size_t precision = 8, bool diagnostic = true)
    {
        os << name << " " << Detail::DebugPrint(shape) << " " << Detail::TypeID<T>() << (cnst ? " const" : "");
        if (diagnostic)
            Detail::PrintDiagnostic(os, data, Detail::Size(shape), precision);
        os << std::endl;
        if (data == NULL || first == 0 || last == 0)
            return;
        size_t n = shape.size();
        Shape firsts(n), lasts(n), index(n, 0);
        Strings separators(n);
        for (ptrdiff_t i = n - 1; i >= 0; --i)
        {
            if (i == ptrdiff_t(n - 1))
            {
                firsts[i] = first;
                lasts[i] = last;
                separators[i] = "\t";
            }
            else
            {
                firsts[i] = std::max<size_t>(firsts[i + 1] - 1, 1);
                lasts[i] = std::max<size_t>(lasts[i + 1] - 1, 1);
                separators[i] = separators[i + 1] + "\n";
            }
        }
        size_t padding = Detail::DebugPrintPadding(data, shape, precision);
        Detail::DebugPrint(os, data, shape, firsts, lasts, separators, index, 0, precision, padding);
        if (n == 1 || n == 0)
            os << "\n";
    }

    template<class T> inline void DebugPrint(std::ostream& os, const T* data, size_t size, const std::string& name, bool cnst = false, size_t first = 6, size_t last = 2, size_t precision = 8, bool diagnostic = true)
    {
        DebugPrint(os, data, Shape({ size }), name, cnst, first, last, precision, diagnostic);
    }

    template<class T> inline void DebugPrint(std::ostream& os, const std::vector<T>& vector, const String& name, bool cnst, size_t first, size_t last, size_t precision, bool diagnostic)
    {
        DebugPrint(os, vector.data(), Shape({ vector.size() }), name, cnst, first, last, precision, diagnostic);
    }

    template<class T> inline void DebugPrint(const T* data, const Shape& shape, const std::string& name, bool cnst = false, size_t first = 999, size_t last = 999, size_t precision = 8, bool diagnostic = true)
    {
        std::stringstream ss;
        ss << name;
        for (size_t i = 0; i < shape.size(); ++i)
            ss << "_" << shape[i];
        ss << ".txt";
        std::ofstream ofs(ss.str().c_str());
        if (ofs.is_open())
        {
            DebugPrint(ofs, data, shape, name, cnst, first, last, precision, diagnostic);
            ofs.close();
        }
    }
}
