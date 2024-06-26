/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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
// 
//#define SYNET_SIMD_SYNET_DISABLE

//#define SYNET_PERFORMANCE_STATISTIC

//#define SYNET_ONNXRUNTIME_ENABLE

#define SYNET_MALLOC_TRIM_THRESHOLD 1024*1024
//#define SYNET_MALLOC_DEBUG

#define SYNET_INT8_SAFE_ZERO 1

//#define SYNET_BF16_ROUND_TEST

#define SYNET_TENSOR_API_OLD

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

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

#if !defined(SYNET_VERSION)
#include "Version.h"
#endif

#if defined(_MSC_VER)
#define SYNET_INLINE __forceinline
#elif defined(__GNUC__)
#define SYNET_INLINE inline __attribute__ ((always_inline))
#else
#error This platform is unsupported!
#endif

#if defined(SYNET_PERFORMANCE_STATISTIC) && !defined(SYNET_PERF_FUNC)
#include "Cpl/Performance.h"
#define SYNET_PERF_FUNC() CPL_PERF_FUNC()
#define SYNET_PERF_BLOCK(name) CPL_PERF_BEG(name)
#define SYNET_PERF_BLOCK_END(name) CPL_PERF_END(name)
#define SYNET_PERF_DECL(name) Cpl::PerformanceMeasurer * name;
#define SYNET_PERF_SET(name, value) name = value;
#define SYNET_PERF_INIT(name, desc, flop) name = Cpl::PerformanceStorage::Global().Get(desc, flop);
#define SYNET_PERF_TEST(name) Cpl::PerformanceHolder CPL_CAT(__pmh,__LINE__)(name);
#elif !defined(SYNET_PERF_FUNC)
#define SYNET_PERF_FUNC()
#define SYNET_PERF_BLOCK(name)
#define SYNET_PERF_BLOCK_END(name)
#define SYNET_PERF_DECL(name)
#define SYNET_PERF_SET(name, value)
#define SYNET_PERF_INIT(name, desc, flop)
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
    typedef std::vector<uint64_t> ULongs;
    typedef std::vector<float> Floats;
#ifdef SYNET_SIMD_LIBRARY_ENABLE
    typedef Simd::View<Simd::Allocator> View;
    typedef std::vector<View> Views;
#endif

    SYNET_INLINE String Version()
    {
        return SYNET_VERSION;
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

    //---------------------------------------------------------------------------------------------

    struct Deletable
    {
        virtual ~Deletable() {}
    };
}

#define SYNET_ERROR(message) \
    { \
       CPL_LOG_SS(Error, message); \
       return 0; \
    }
