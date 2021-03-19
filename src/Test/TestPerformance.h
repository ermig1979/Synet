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

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__GNUC__)
#include <sys/time.h>
#else
#error Platform is not supported!
#endif

namespace Test
{
#if defined(_MSC_VER)
    inline double Time()
    {
        LARGE_INTEGER counter, frequency;
        QueryPerformanceCounter(&counter);
        QueryPerformanceFrequency(&frequency);
        return double(counter.QuadPart) / double(frequency.QuadPart);
    }

    inline int64_t TimeCounter()
    {
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return counter.QuadPart;
    }

    inline int64_t TimeFrequency()
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return frequency.QuadPart;
    }
#elif defined(__GNUC__)
    inline double Time()
    {
        timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + t.tv_usec * 0.000001;
    }
#if 0
    inline int64_t TimeCounter()
    {
        timeval t;
        gettimeofday(&t, NULL);
        return int64_t(t.tv_sec) * int64_t(1000000) + int64_t(t.tv_usec);
    }

    inline int64_t TimeFrequency()
    {
        return int64_t(1000000);
    }
#else
    inline int64_t TimeCounter()
    {
        timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        return int64_t(t.tv_sec) * int64_t(1000000000) + int64_t(t.tv_nsec);
    }

    inline int64_t TimeFrequency()
    {
        return int64_t(1000000000);
    }
#endif
#else
#error Platform is not supported!
#endif

    inline double Miliseconds(int64_t count)
    {
        return double(count) / double(TimeFrequency()) * 1000.0;
    }

    //-------------------------------------------------------------------------

    class PerformanceMeasurer
    {
        String	_name;
        int64_t _start, _current, _total, _min, _max, _count, _flop;
        bool _entered, _paused;

    public:
        PerformanceMeasurer(const String & name = "Unnamed", int64_t flop = 0)
            : _name(name)
            , _flop(flop)
            , _count(0)
            , _current(0)
            , _total(0)
            , _min(std::numeric_limits<int64_t>::max())
            , _max(std::numeric_limits<int64_t>::min())
            , _entered(false)
            , _paused(false)
        {
        }

        PerformanceMeasurer(const PerformanceMeasurer& pm)
            : _name(pm._name)
            , _flop(pm._flop)
            , _count(pm._count)
            , _start(pm._start)
            , _current(pm._current)
            , _total(pm._total)
            , _min(pm._min)
            , _max(pm._max)
            , _entered(pm._entered)
            , _paused(pm._paused)
        {
        }

        inline void Enter()
        {
            if (!_entered)
            {
                _entered = true;
                _paused = false;
                _start = TimeCounter();
            }
        }

        inline void Leave(bool pause = false)
        {
            if (_entered || _paused)
            {
                if (_entered)
                {
                    _entered = false;
                    _current += TimeCounter() - _start;
                }
                if (!pause)
                {
                    _total += _current;
                    _min = std::min(_min, _current);
                    _max = std::max(_max, _current);
                    ++_count;
                    _current = 0;
                }
                _paused = pause;
            }
        }

        inline double Total() const
        {
            return Miliseconds(_total);
        }

        inline double Average() const
        {
            return _count ? (Total() / _count) : 0;
        }

        inline double GFlops() const
        {
            return _count && _flop && _total > 0 ? (double(_flop) * _count / Total() / 1000000.0) : 0;
        }

        inline String Statistic() const
        {
            std::stringstream ss;
            ss << ToString(Total(), 0) << " ms";
            ss << " / " << _count << " = ";
            ss << ToString(Average(), 3) << " ms";
            ss << " {min=" << ToString(Miliseconds(_min), 3) << "; max=" << ToString(Miliseconds(_max), 3) << "}";
            if (_flop)
                ss << " " << ToString(GFlops(), 1) << " GFlops";
            return ss.str();
        }

        inline void Combine(const PerformanceMeasurer& other)
        {
            _count += other._count;
            _total += other._total;
            _min = std::min(_min, other._min);
            _max = std::max(_max, other._max);
        }

        inline String Name() const
        { 
            return _name; 
        }
    };

    //-------------------------------------------------------------------------

    class PerformanceMeasurerHolder
    {
        PerformanceMeasurer* _pm;

    public:
        inline PerformanceMeasurerHolder(PerformanceMeasurer* pm, bool enter = true)
            : _pm(pm)
        {
            if (_pm && enter)
                _pm->Enter();
        }

        inline void Enter()
        {
            if (_pm)
                _pm->Enter();
        }

        inline void Leave(bool pause)
        {
            if (_pm)
                _pm->Leave(pause);
        }

        inline ~PerformanceMeasurerHolder()
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

        inline FunctionMap & ThisThread()
        {
            static thread_local FunctionMap* thread = NULL;
            if (thread == NULL)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                thread = &_map[std::this_thread::get_id()];
            }
            return *thread;
        }

        inline String RemoveArgs(const String & name)
        {
            size_t beg = name.find("("), end = name.find(")");
            if (beg != String::npos && end != String::npos)
                return name.substr(0, beg + 1) + name.substr(end, name.size() - 1);
            return name;
        }

        inline String RemoveDesc(const String& name)
        {
            size_t pos = name.find(" {");
            if (pos != String::npos)
                return name.substr(0, pos);
            return name;
        }

    public:
        static PerformanceMeasurerStorage s_storage;

        PerformanceMeasurerStorage()
        {
        }

        ~PerformanceMeasurerStorage()
        {
        }

        inline PerformanceMeasurer * Get(const String & name, int64_t flop = 0)
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

        inline PerformanceMeasurer * Get(const String & function, const String & block, int64_t flop = 0)
        {
            return Get(function + " { " + block + " } ", flop);
        }

        inline void Clear()
        {
            _map.clear();
        }

        void Print(std::ostream & os, double threshold = 0, const String & main = "Network::Predict", const String& term = "Layer::Forward")
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

            double time = 0;
            size_t size = 0;
            for (FunctionMap::const_iterator j = total.begin(); j != total.end(); j++)
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
                os << " (time >= " << ToString(time, 0) << " ms)";
#endif
            os << std::endl;
#ifdef __SimdLib_hpp__
            Simd::PrintInfo(os);
#endif
#ifdef BLIS_H        
            os << "Blis arch: " << bli_arch_string(bli_arch_query_id()) << std::endl;
#endif
            for (FunctionMap::const_iterator j = total.begin(); j != total.end(); j++)
            {
                const String& name = j->first;
                const PerformanceMeasurer& perf = *j->second;
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
                os << ": " << perf.Statistic() << std::endl;
            }
            os << "----- ~~~~~~~~~~~~~~~~~~~ -----" << std::endl;
        }

        PerformanceMeasurer GetCombined(const String& name)
        {
            PerformanceMeasurer combined;
            for (ThreadMap::const_iterator t = _map.begin(); t != _map.end(); t++)
            {
                FunctionMap::const_iterator f = t->second.find(name);
                if (f != t->second.end() && f->second->Average() != 0)
                {
                    if (combined.Average() == 0)
                        combined = *f->second;
                    else
                        combined.Combine(*f->second);
                }
            }
            return combined;
        }
    };
}

#ifdef _MSC_VER
#define TEST_FUNCTION __FUNCTION__
#else
#define TEST_FUNCTION __PRETTY_FUNCTION__
#endif

#define TEST_PERF_FUNC() ::Test::PerformanceMeasurerHolder SYNET_CAT(__pmh, __LINE__)(::Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION));
#define TEST_PERF_FUNC_FLOP(flop) ::Test::PerformanceMeasurerHolder SYNET_CAT(__pmh, __LINE__)(::Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, flop));
#define TEST_PERF_BLOCK(name) ::Test::PerformanceMeasurerHolder SYNET_CAT(__pmh, __LINE__)(::Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, name));
#define TEST_PERF_BLOCK_FLOP(name, flop) ::Test::PerformanceMeasurerHolder SYNET_CAT(__pmh, __LINE__)(::Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, name, flop));
#define TEST_PERF_BLOCK_END(name) ::Test::PerformanceMeasurerStorage::s_storage.Get(TEST_FUNCTION, name)->Leave();

