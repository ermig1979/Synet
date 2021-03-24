/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#ifndef SYNET_SIMD_LIBRARY_ENABLE
#define SYNET_SIMD_LIBRARY_ENABLE
#endif
#define SYNET_GEMM_SIMD_LIBRARY

#ifndef WIN32
#define SYNET_TEST_FIRST_RUN
#endif
#define SYNET_TEST_SECOND_RUN
//#define SYNET_PERFORMANCE_STATISTIC

//#define SYNET_TEST_MEMORY_LOAD
//#define SYNET_TEST_NET_RESHAPE
//#define SYNET_TEST_SET_INPUT
//#define SYNET_TEST_STB_EXTERNAL

#if !defined(SYNET_TEST_FIRST_RUN) && !defined(SYNET_TEST_SECOND_RUN)
#error SYNET_TEST_FIRST_RUN and SYNET_TEST_SECOND_RUN are undefined!
#endif

#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <algorithm>
#include <type_traits>

#ifdef SYNET_SIMD_LIBRARY_ENABLE
#include "Simd/SimdLib.h"
#include "Simd/SimdLib.hpp"
#include "Simd/SimdView.hpp"
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdPixel.hpp"
#include "Simd/SimdFont.hpp"
#endif

namespace Test
{
    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef std::list<String> StringList;

    typedef Simd::View<Simd::Allocator> View;
    typedef std::vector<View> Views;

    typedef Simd::Point<ptrdiff_t> Point;
    typedef Point Size;

    typedef std::vector<float> Vector;
    typedef std::vector<Vector> Vectors;
}


