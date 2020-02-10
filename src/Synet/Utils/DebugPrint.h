#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <assert.h>

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


        template<class T> void DebugPrint(std::ostream& os, T value, size_t precision)
        {
            os << (int)value;
        }

        template<> inline void DebugPrint(std::ostream& os, float value, size_t precision)
        {
            os << std::fixed << std::setprecision(precision) << value;
        }

        template<class T> std::ostream& DebugPrint(std::ostream& os, T value, size_t precision, size_t padding)
        {
            std::stringstream ss;
            DebugPrint(ss, value, precision);
            for (size_t i = ss.str().size(); i < padding; ++i)
                os << " ";
            os << ss.str();
            return os;
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
    }

    template<class T> inline void DebugPrint(std::ostream & os, const T * data, const Shape& shape, const std::string & name, size_t first = 6, size_t last = 2, size_t precision = 8)
    {
        os << name << " { ";
        for (size_t i = 0; i < shape.size(); ++i)
            os << shape[i] << " ";
        os << "} ";
        os << Detail::TypeID<T>() << std::endl;
        if (data == NULL)
            return;
        size_t n = shape.size();
        Shape firsts(n), lasts(n), index(n, 0);
        Strings separators(n);
        for (ptrdiff_t i = n - 1; i >= 0; --i)
        {
            if (i == n - 1)
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

    template<class T> inline void DebugPrint(std::ostream& os, const T* data, size_t size, const std::string& name, size_t first = 6, size_t last = 2, size_t precision = 8)
    {
        DebugPrint(os, data, Shape({ size }), name, first, last, precision);
    }

    template<class T> void DebugPrint(std::ostream& os, const std::vector<T>& vector, const String& name, size_t first, size_t last, size_t precision)
    {
        DebugPrint(os, vector.data(), Shape({ vector.size() }), name, first, last, precision);
    }
}
