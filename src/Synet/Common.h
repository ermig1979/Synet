/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

//#define SYNET_PROTOBUF_ENABLE
//#define SYNET_PERFORMANCE_STATISTIC

//#define SYNET_CAFFE_ENABLE
//#define SYNET_DARKNET_ENABLE
//#define SYNET_TENSORFLOW_ENABLE
//#define SYNET_OPENCV_ENABLE
//#define SYNET_ONNXRUNTIME_ENABLE
//#define SYNET_ONNX_IE_ENABLE
//#define SYNET_LEGACY_2020_ENABLE

#define SYNET_MALLOC_TRIM_THRESHOLD 1024*1024
//#define SYNET_MALLOC_DEBUG

#define SYNET_INT8_SAFE_ZERO 1

#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <float.h>
#include <limits.h>
#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#include <sys/resource.h>
#include <stdio.h>
#endif
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
#include <queue>
#include <cmath>
#include <iomanip>
#include <type_traits>

#include "Cpl/Param.h"

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
#include "Simd/SimdLib.h"
#include "Simd/SimdLib.hpp"
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

#ifndef SYNET_PERF_DECL
#define SYNET_PERF_DECL(name)
#endif

#ifndef SYNET_PERF_SET
#define SYNET_PERF_SET(name, value)
#endif

#ifndef SYNET_PERF_INIT
#define SYNET_PERF_INIT(name, desc, flop)
#endif

#ifndef SYNET_PERF_TEST
#define SYNET_PERF_TEST(name)
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
    const int BUFFER_COUNT = 2;

    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef std::vector<size_t> Shape;
    typedef std::vector<Shape> Shapes;
    typedef std::vector<size_t> Index;
    typedef std::vector<uint8_t> Bytes;
    typedef std::vector<int> Ints;
    typedef std::vector<int64_t> Longs;
    typedef std::vector<float> Floats;
#ifdef SYNET_SIMD_LIBRARY_ENABLE
    typedef Simd::View<Simd::Allocator> View;
    typedef std::vector<View> Views;
#endif

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

    inline size_t GetThreadNumber()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        return SimdGetThreadNumber();
#else
        return 1;
#endif
    }

    inline void SetThreadNumber(size_t threadNumber)
    {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
        SimdSetThreadNumber(threadNumber);
#endif
    }

    inline bool GetFastMode()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        return ::SimdGetFastMode() == ::SimdTrue;
#else
        return false;
#endif
    }

    inline void SetFastMode(bool value)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        ::SimdSetFastMode(value ? ::SimdTrue : ::SimdFalse);
#endif
    }

    inline void PrintMemoryUsage()
    {
#if defined(__linux__)
        size_t rss, peak_rss;
        FILE* fp = NULL;
        if ((fp = fopen("/proc/self/statm", "r")) == NULL)
            return;
        if (fscanf(fp, "%*s%ld", &rss) != 1)
        {
            fclose(fp);
            return;
        }
        fclose(fp);
        rss *= (size_t)sysconf(_SC_PAGESIZE);

        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
        peak_rss = (size_t)(rusage.ru_maxrss * 1024L);
        std::cout << " WorkingSetSize = " << rss / 1024 / 1024 << " MB, PeakWorkingSetSize = " << peak_rss / 1024 / 1024 << " MB" << std::endl;
#endif
    }

    enum DebugPrintType
    {
        DebugPrintOutput = 0,
        DebugPrintLayerDst,
        DebugPrintLayerWeight,
        DebugPrintInt8Buffers,
        DebugPrintLayerInternal,
    };
}
