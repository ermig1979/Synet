/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
    template<class T> SYNET_INLINE  String ValueToString(const T& value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template<> SYNET_INLINE String ValueToString<size_t>(const size_t& value)
    {
        return ValueToString((ptrdiff_t)value);
    }

    template<> SYNET_INLINE String ValueToString<float>(const float& value)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(std::numeric_limits<float>::digits10);
        ss << value;
        return ss.str();
    }

    template<class T> SYNET_INLINE String ValueToString(const std::vector<T>& values)
    {
        std::stringstream ss;
        for (size_t i = 0; i < values.size(); ++i)
            ss << (i ? " " : "") << ValueToString<T>(values[i]);
        return ss.str();
    }

    template<class T> SYNET_INLINE  void StringToValue(const String& string, T& value)
    {
        std::stringstream ss(string);
        ss >> value;
    }

    template<> SYNET_INLINE void StringToValue<size_t>(const String& string, size_t& value)
    {
        StringToValue(string, (ptrdiff_t&)value);
    }

    template<> SYNET_INLINE void StringToValue<bool>(const String& string, bool& value)
    {
        if (string == "0" || string == "false" || string == "False")
            value = false;
        else if (string == "1" || string == "true" || string == "True")
            value = true;
        else
            assert(0);
    }

    template<class T> SYNET_INLINE void StringToValue(const String& string, std::vector<T>& values)
    {
        std::stringstream ss(string);
        values.clear();
        while (!ss.eof())
        {
            String item;
            ss >> item;
            if (item.size())
            {
                T value;
                StringToValue(item, value);
                values.push_back(value);
            }
        }
    }

    SYNET_INLINE String ToLowerCase(const String& src)
    {
        String dst(src);
        for (size_t i = 0; i < dst.size(); ++i)
        {
            if (dst[i] <= 'Z' && dst[i] >= 'A')
                dst[i] = dst[i] - ('Z' - 'z');
        }
        return dst;
    }

    template<typename Enum, int Size> SYNET_INLINE Enum StringToEnum(const String& string)
    {
        int type = Size - 1;
        for (; type >= 0; --type)
        {
            if (ToLowerCase(ValueToString<Enum>((Enum)type)) == ToLowerCase(string))
                return (Enum)type;
        }
        return (Enum)type;
    }

    SYNET_INLINE Strings Separate(const String& str, const String& delimeter)
    {
        size_t current = 0;
        Strings result;
        while (current != String::npos)
        {
            size_t next = str.find(delimeter, current);
            result.push_back(str.substr(current, next - current));
            current = next;
            if (current != String::npos)
                current += delimeter.size();
        }
        return result;
    }

    SYNET_INLINE String ExpandLeft(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        ss << value;
        return ss.str();
    }

    SYNET_INLINE String ExpandRight(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        ss << value;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    SYNET_INLINE String ExpandBoth(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        for (size_t i = 0, left = (count - value.size()) / 2; i < left; i++)
            ss << " ";
        ss << value;
        for (size_t i = ss.str().size(); i < count; i++)
            ss << " ";
        return ss.str();
    }
}