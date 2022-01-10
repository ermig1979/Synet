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

#include "TestUtils.h"

#if defined(SYNET_CPL_ENABLE)

#ifndef CPL_PERF_ENABLE
#define CPL_PERF_ENABLE
#endif

#include "Cpl/Performance.h"

namespace Test
{
    void PrintPerformance(std::ostream& os, double threshold = 0, const String& main = "Network::Predict", const String& term = "Layer::Forward")
    {
        typedef Cpl::PerformanceStorage::FunctionMap FunctionMap;
        FunctionMap merged = Cpl::PerformanceStorage::Global().Merged();

        double time = 0;
        size_t size = 0;
        for (FunctionMap::const_iterator j = merged.begin(); j != merged.end(); j++)
        {
            if (j->first.find(term) != String::npos)
                size = std::max(size, j->first.size());
            if (j->first.find(main) != String::npos)
            {
                time = std::max(time, j->second->Total() * threshold);
                size = std::max(size, j->first.size());
            }
        }

        os << "----- Performance Report -----";
#if defined(SYNET_LAYER_STATISTIC) || defined(SYNET_SIZE_STATISTIC)
        if (time > 0)
            os << " (time >= " << Cpl::ToStr(time, 0) << " ms)";
#endif
        os << std::endl;
#ifdef __SimdLib_hpp__
        Simd::PrintInfo(os);
#endif
#ifdef BLIS_H        
        os << "Blis arch: " << bli_arch_string(bli_arch_query_id()) << std::endl;
#endif
        for (FunctionMap::const_iterator j = merged.begin(); j != merged.end(); j++)
        {
            const String& name = j->first;
            const Cpl::PerformanceMeasurer& perf = *j->second;
            if (name.find(main) != String::npos)
                os << ExpandRight(name, size);
            else if (name.find(term) != String::npos)
            {
                if (perf.Total() >= time)
                    os << ExpandRight(name, size + (name.find(" {  ") == String::npos ? 2 : 1));
                else
                    continue;
            }
            else
                os << name;
            os << ": " << perf.ToStr() << std::endl;
        }
        os << "----- ~~~~~~~~~~~~~~~~~~~ -----" << std::endl;
    }
}

#endif

