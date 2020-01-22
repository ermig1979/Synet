/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#elif defined(__GNUC__)
#include <sys/time.h>
#else
#error Platform is not supported!
#endif

namespace Test
{
    typedef std::string String;

    inline double GetTime()
    {
#if defined(_MSC_VER)
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return double(counter.QuadPart) / double(frequency.QuadPart);
#elif defined(__GNUC__)
        timeval t1;
        gettimeofday(&t1, NULL);
        return t1.tv_sec + t1.tv_usec / 1000000.0;
#else
#error Platform is not supported!
#endif
    }

    //-------------------------------------------------------------------------

    class PerformanceMeasurer
    {
        String	_description;
        double _start;
        int64_t _count, _flop;
        double _total;
        double _min;
        double _max;
        bool _entered;

    public:
        PerformanceMeasurer(const String & description = "Unnamed", int64_t flop = 0)
            : _description(description)
            , _count(0)
            , _total(0)
            , _min(std::numeric_limits<double>::max())
            , _max(std::numeric_limits<double>::min())
            , _entered(false)
            , _flop(flop)
        {
        }

        PerformanceMeasurer(const PerformanceMeasurer & pm)
            : _description(pm._description)
            , _count(pm._count)
            , _total(pm._total)
            , _min(pm._min)
            , _max(pm._max)
            , _entered(pm._entered)
            , _flop(pm._flop)
        {
        }

        void Enter()
        {
            if (!_entered)
            {
                _entered = true;
                _start = GetTime();
            }
        }

        void Leave()
        {
            if (_entered)
            {
                _entered = false;
                double difference = double(GetTime() - _start);
                _total += difference;
                _min = std::min(_min, difference);
                _max = std::max(_max, difference);
                ++_count;
            }
        }

        double Average() const
        {
            return _count ? (_total / _count) : 0;
        }

        double GFlops() const
        {
            return _count && _flop && _total > 0 ? (double(_flop) * _count / _total / 1000000000.0) : 0;
        }

        String Statistic() const
        {
            std::stringstream ss;
            ss << _description << ": ";
            ss << std::setprecision(0) << std::fixed << _total * 1000 << " ms";
            ss << " / " << _count << " = ";
            ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
            ss << std::setprecision(3) << " { min = " << _min * 1000.0 << " ; max = " << _max * 1000.0 << " } ";
            if (_flop)
                ss << " " << std::setprecision(1) << GFlops() << " GFlops";
            return ss.str();
        }

        String Description() const 
        { 
            return _description; 
        }

        void Combine(const PerformanceMeasurer & other)
        {
            _count += other._count;
            _total += other._total;
            _min = std::min(_min, other._min);
            _max = std::max(_max, other._max);
        }
    };

    //-------------------------------------------------------------------------

    class ScopedPerformanceMeasurer
    {
        PerformanceMeasurer * _pm;
    public:

        ScopedPerformanceMeasurer(PerformanceMeasurer * pm) : _pm(pm)
        {
            if (_pm)
                _pm->Enter();
        }

        ~ScopedPerformanceMeasurer()
        {
            if (_pm)
                _pm->Leave();
        }
    };

    //-------------------------------------------------------------------------

    class PerformanceMeasurerStorage
    {
        typedef PerformanceMeasurer Pm;
        typedef std::shared_ptr<Pm> PmPtr;
        typedef std::map<String, PmPtr> FunctionMap;
        typedef std::map<std::thread::id, FunctionMap> ThreadMap;

        ThreadMap _map;
        mutable std::mutex _mutex;

        FunctionMap & ThisThread()
        {
            static thread_local FunctionMap* thread = NULL;
            if (thread == NULL)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                thread = &_map[std::this_thread::get_id()];
            }
            return *thread;
        }

    public:
        static PerformanceMeasurerStorage s_storage;

        PerformanceMeasurerStorage()
        {
        }

        ~PerformanceMeasurerStorage()
        {
        }

        PerformanceMeasurer * Get(const String & name, int64_t flop = 0)
        {
            FunctionMap& thread = ThisThread();
            PerformanceMeasurer* pm = NULL;
            FunctionMap::iterator it = thread.find(name);
            if (it == thread.end())
            {
                pm = new PerformanceMeasurer(name, flop);
                thread[name].reset(pm);
            }
            else
                pm = it->second.get();
            return pm;
        }

        PerformanceMeasurer * Get(const String & function, const String & block, int64_t flop = 0)
        {
            return Get(function + " { " + block + " } ", flop);
        }

        void Clear()
        {
            _map.clear();
        }

        void Print(std::ostream & os)
        {
            if (this == 0)
                return;
            std::lock_guard<std::mutex> lock(_mutex);
            FunctionMap total;
            for (ThreadMap::const_iterator i = _map.begin(); i != _map.end(); i++)
            {
                for (FunctionMap::const_iterator j = i->second.begin(); j != i->second.end(); j++)
                {
                    if (j->second->Average() != 0)
                    {
                        if (total.find(j->first) == total.end())
                            total[j->first].reset(new PerformanceMeasurer(*j->second));
                        else
                            total[j->first]->Combine(*j->second);
                    }
                }
            }

            os << "----- Performance -----" << std::endl;
#ifdef __SimdLib_hpp__
            Simd::PrintInfo(os);
#endif
#ifdef BLIS_H        
            os << "Blis arch: " << bli_arch_string(bli_arch_query_id()) << std::endl;
#endif
            for (FunctionMap::const_iterator j = total.begin(); j != total.end(); j++)
                os << j->second->Statistic() << std::endl;
            os << "----- ~~~~~~~~~~~ -----" << std::endl;
        }
    };
}

#ifdef _MSC_VER
#define TEST_FUNCTION __FUNCTION__
#else
#define TEST_FUNCTION __PRETTY_FUNCTION__
#endif

#define TEST_PERF_FUNC() Test::ScopedPerformanceMeasurer SYNET_CAT(__spm, __LINE__)(Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION));
#define TEST_PERF_BLOCK(name) Test::ScopedPerformanceMeasurer SYNET_CAT(__spm, __LINE__)(Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, name));
#define TEST_PERF_BLOCK_END(name) Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, name)->Leave();


