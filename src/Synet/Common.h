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

//#define SYNET_PROTOBUF_ENABLE

//#define SYNET_CAFFE_ENABLE

#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <float.h>

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

#ifdef SYNET_SIMD_LIBRARY_ENABLE
#include "Simd/SimdLib.h"
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

namespace Synet
{
    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef std::vector<size_t> Shape;
    typedef std::vector<Shape> Shapes;
    typedef std::vector<size_t> Index;
}