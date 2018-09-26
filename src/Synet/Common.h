/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

//#define SYNET_SIMD_LIBRARY_ENABLE
//#define SYNET_OPEN_BLAS_ENABLE
//#define SYNET_GEMM_SIMD_LIBRARY
//#define SYNET_GEMM_DYNAMIC

//#define SYNET_GEMM_COMPARE
//#define SYNET_PROTOBUF_ENABLE
//#define SYNET_WINOGRAD_COMPARE
//#define SYNET_CONVOLUTION_STATISTIC

//#define SYNET_CAFFE_ENABLE
//#define SYNET_YOLO_ENABLE
//#define SYNET_TENSORFLOW_ENABLE
//#define SYNET_OPENCV_ENABLE

//#define SYNET_DEBUG_PRINT_ENABLE

#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <float.h>
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4996)
#include <io.h>
#pragma warning (pop)
#else
#include <unistd.h>
#endif

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cmath>

#if defined(SYNET_SIMD_LIBRARY_ENABLE) || defined(SYNET_SIMD_LIBRARY_GEMM_ENABLE)
#include "Simd/SimdLib.h"
#include "Simd/SimdLib.hpp"
#endif

#ifdef SYNET_OPEN_BLAS_ENABLE
extern "C" 
{
#include <cblas.h>
}
#endif

#if defined(_MSC_VER)
#define SYNET_INLINE __forceinline
#elif defined(__GNUC__)
#define SYNET_INLINE inline __attribute__ ((always_inline))
#else
#error This platform is unsupported!
#endif

#ifndef SYNET_PERF_FUNC
#define SYNET_PERF_FUNC()
#endif

#ifndef SYNET_PERF_BLOCK
#define SYNET_PERF_BLOCK(name)
#endif

#ifndef SYNET_PERF_BLOCK_END
#define SYNET_PERF_BLOCK_END(name)
#endif

#define SYNET_CLASS_INSTANCE(name) \
  template class name<float>; 

#define SYNET_STRINGIFY(X) SYNET_STRINGIFY_DO(X)    
#define SYNET_STRINGIFY_DO(X) #X

#define SYNET_CAT(X,Y) SYNET_CAT_DO(X,Y)
#define SYNET_CAT_DO(X,Y) X##Y

#define SYNET_INCLUDE(path, file) SYNET_STRINGIFY(path/file)

namespace Synet
{
    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef std::vector<size_t> Shape;
    typedef std::vector<Shape> Shapes;
    typedef std::vector<size_t> Index;
    typedef std::vector<int> Ints;
    typedef std::vector<float> Floats;

    SYNET_INLINE bool FileExist(const String & path)
    {
#ifdef _MSC_VER
        return (::_access(path.c_str(), 0) != -1);
#else
        return (::access(path.c_str(), F_OK) != -1);
#endif
    }

    SYNET_INLINE Strings Separate(const String & str, const String & delimeter)
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

    inline size_t GetThreadNumber()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        return ::SimdGetThreadNumber();
#elif defined(SYNET_OPEN_BLAS_ENABLE)
        return ::openblas_get_num_threads();
#else
        return 1;
#endif
    }

    inline void SetThreadNumber(size_t threadNumber)
    {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        ::SimdSetThreadNumber(threadNumber);
#endif
#ifdef SYNET_OPEN_BLAS_ENABLE
        ::openblas_set_num_threads((int)threadNumber);
        ::goto_set_num_threads((int)threadNumber);
#endif
    }

    inline bool GetFlushToZero()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        return ::SimdGetFlushToZero() == ::SimdTrue;
#else
        return false;
#endif
    }

    inline void SetFlushToZero(bool value)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        ::SimdSetFlushToZero(value ? ::SimdTrue : ::SimdFalse);
#endif
    }

    template <class T> struct Region
    {
        T x, y, w, h, prob;
        size_t id;
    };
}
