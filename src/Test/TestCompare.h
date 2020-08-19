/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

#include "TestCommon.h"
#include "TestUtils.h"
#include "TestOptions.h"
#include "TestPerformance.h"
#include "TestSynet.h"
#include "TestImage.h"

namespace Test
{
    template<class FirstNetwork, class SecondNetwork> class Comparer
    {
    public:
        Comparer(const Options& options)
            : _options(options)
            , _progressMessageSizeMax(0)
            , _notifiedFirst(false)
            , _notifiedSecond(false)
        {
            assert(_options.testThreads >= 0);
            if (_options.enable & ENABLE_FIRST)
                _firsts.resize(_options.TestThreads());
            if (_options.enable & ENABLE_SECOND)
                _seconds.resize(_options.TestThreads());
        }

        bool Run()
        {
            PrintStartMessage();
            if (!LoadTestParam())
                return false;
            if (!CreateDirectories())
                return false;
            if (!InitNetworks())
                return false;
            if (!CreateTestList())
                return false;
            if (_options.testThreads == 0)
            {
                if (!SingleThreadComparison())
                    return false;
            }
            else
            {
                if (!MultiThreadsComparison())
                    return false;
            }
            return true;
        }

    private:
        const Options& _options;
        TestParamHolder _param;
        std::vector<FirstNetwork> _firsts;
        std::vector<SecondNetwork> _seconds;

        struct Output
        {
            Tensors first;
            Tensors second;
        };
        typedef std::vector<Output> Outputs;
        struct TestData
        {
            Strings path;
            Tensors input;
            Outputs output;
        };
        typedef std::shared_ptr<TestData> TestDataPtr;
        typedef std::vector<TestDataPtr> TestDataPtrs;
        TestDataPtrs _tests;

        struct Thread
        {
            size_t current;
            bool first, second;
            std::thread thread;
            Thread() : current(0), first(false), second(false) {}
        };
        std::vector<Thread> _threads;
        std::condition_variable _startFirst, _startSecond;
        bool _notifiedFirst, _notifiedSecond;
        size_t _progressMessageSizeMax;

        typedef Synet::Difference<float> Difference;
        typedef std::vector<Difference> Differences;

        void PrintStartMessage() const
        {
            std::cout << "Start ";
            if (_options.enable & ENABLE_FIRST)
                std::cout << Options::FullName(_firsts[0].Name(), _firsts[0].Type()) << " ";
            if (_options.enable == (ENABLE_FIRST | ENABLE_SECOND))
                std::cout << "and ";
            if (_options.enable & ENABLE_SECOND)
                std::cout << Options::FullName(_seconds[0].Name(), _seconds[0].Type()) << " ";
            if (_options.testThreads > 0)
                std::cout << _options.testThreads << "-threads ";
            else
                std::cout << "single-thread ";
            std::cout << (_options.enable == (ENABLE_FIRST | ENABLE_SECOND) ? "comparison " : "performance ");
            std::cout << "tests :" << std::endl;
        }

        bool PrintFinishMessage() const
        {
            std::cout << ExpandRight("Tests are finished successfully!", _progressMessageSizeMax) << std::endl << std::endl;
            return true;
        }

        bool LoadTestParam()
        {
            if (!_param.Load(_options.testParam))
            {
                std::cout << "Can't load file '" << _options.testParam << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool InitNetwork(const String& model, const String& weight, Network& network) const
        {
            if (!FileExists(model))
            {
                std::cout << "File '" << model << "' is not exist!" << std::endl;
                return false;
            }
            if (!FileExists(weight))
            {
                std::cout << "File '" << weight << "' is not exist!" << std::endl;
                return false;
            }
            Network::Options options(_options.outputDirectory, _options.workThreads, 
                _options.consoleSilence, _options.batchSize, _options.debugPrint, _options.regionThreshold);
            if (!network.Init(model, weight, options, _param()))
            {
                std::cout << "Can't load " << network.Name() << " from '" << model << "' and '" << weight << "' !" << std::endl;
                return false;
            }
            Shape shape = network.SrcShape(0);
            if (!(shape[1] == 1 || shape[1] == 3))
            {
                std::cout << "Wrong " << network.Name() << " classifier channels count '" << shape[1] << " !" << std::endl;
                return false;
            }
            return true;
        }

        bool InitNetworks()
        {
#ifdef SYNET_TEST_FIRST_RUN        
            if (_options.enable & ENABLE_FIRST)
            {
                if (!InitNetwork(_options.firstModel, _options.firstWeight, _firsts[0]))
                    return false;
                _options.firstName = _firsts[0].Name();
                _options.firstType = _firsts[0].Type();
            }
#endif
#ifdef SYNET_TEST_SECOND_RUN
            if (_options.enable & ENABLE_SECOND) 
            {
                if(!InitNetwork(_options.secondModel, _options.secondWeight, _seconds[0]))
                    return false;
                _options.secondName = _seconds[0].Name();
                _options.secondType = _seconds[0].Type();
            }
#endif            
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
            if (_options.enable == (ENABLE_FIRST | ENABLE_SECOND))
            {
                if (_firsts[0].SrcCount() != _seconds[0].SrcCount())
                {
                    std::cout << "Networks have difference source number: " <<
                        _firsts[0].SrcCount() << " != " << _seconds[0].SrcCount() << std::endl;
                    return false;
                }
                for (size_t s = 0; s < _firsts[0].SrcCount(); ++s)
                {
                    const Shape& os = _firsts[0].SrcShape(s);
                    const Shape& ss = _seconds[0].SrcShape(s);
                    if (os != ss)
                    {
                        std::cout << "Networks have difference Src[" << s << "] size: ";
                        std::cout << _firsts[0].Name() << " {" << os[0];
                        for (size_t j = 1; j < os.size(); ++j)
                            std::cout << ", " << os[j];
                        std::cout << "} != " << _seconds[0].Name() << " {" << ss[0];
                        for (size_t j = 1; j < ss.size(); ++j)
                            std::cout << ", " << ss[j];
                        std::cout << "} ! " << std::endl;
                        return false;
                    }
                }
            }
#endif
            return true;
        }

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
            {
                std::cout << "Can't create output directory '" << _options.outputDirectory << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool CreateTestList(const Network& network)
        {
            String imageDirectory = _options.imageDirectory;
            if (imageDirectory.empty())
                imageDirectory = Test::MakePath(DirectoryByPath(_options.testParam), _param().images());
            if (!DirectoryExists(imageDirectory))
            {
                std::cout << "Test image directory '" << imageDirectory << "' is not exists!" << std::endl;
                return false;
            }
            StringList images = GetFileList(imageDirectory, _options.imageFilter, true, false);
            images.sort();

            Strings names;
            names.reserve(images.size());
            size_t curr = 0;
            for(StringList::const_iterator it = images.begin(); it != images.end(); ++it, ++curr)
                if(curr >= _options.imageBegin && curr < _options.imageEnd)
                    names.push_back(*it);

            size_t sN = network.SrcCount(), bN = _options.batchSize;
            size_t tN = names.size() / bN / sN;
            if (tN == 0)
            {
                std::cout << "There is no one image in '" << imageDirectory << "' for '" << _options.imageFilter << "' filter!" << std::endl;
                return false;
            }

            Floats lower = _param().lower(), upper = _param().upper();
            _tests.clear();
            _tests.reserve(tN);
            for (size_t t = 0; t < tN; ++t)
            {
                TestDataPtr test(new TestData());
                test->path.resize(bN * sN);
                test->input.resize(sN);
                test->output.resize(_options.TestThreads());
                for (size_t s = 0; s < sN; ++s)
                {
                    test->input[s].Reshape(network.SrcShape(s));
                    float* input = test->input[s].CpuData();
                    for (size_t b = 0; b < bN; ++b)
                    {
                        size_t p = s * bN + b;
                        test->path[p] = MakePath(imageDirectory, names[(t * bN + b) * sN + s]);
                        View original;
                        if (!LoadImage(test->path[p], original))
                        {
                            std::cout << "Can't read '" << test->path[p] << "' image!" << std::endl;
                            return false;
                        }
                        Shape shape = network.SrcShape(s);
                        if (shape.size() == 4)
                        {
                            if (lower.size() == 1)
                                lower.resize(shape[1], lower[0]);
                            if (upper.size() == 1)
                                upper.resize(shape[1], upper[0]);

                            View converted(original.Size(), shape[1] == 1 ? View::Gray8 : View::Bgr24);
                            Simd::Convert(original, converted);
                            View resized(Size(shape[3], shape[2]), converted.format);
                            Simd::ResizeBilinear(converted, resized);

                            Views channels(shape[1]);
                            if (shape[1] > 1)
                            {
                                for (size_t i = 0; i < shape[1]; ++i)
                                    channels[i].Recreate(resized.Size(), View::Gray8);
                                Simd::DeinterleaveBgr(resized, channels[0], channels[1], channels[2]);
                            }
                            else
                                channels[0] = resized;

                            for (size_t c = 0; c < channels.size(); ++c)
                            {
                                for (size_t y = 0; y < channels[c].height; ++y)
                                {
                                    const uint8_t* row = channels[c].Row<uint8_t>(y);
                                    ::SimdUint8ToFloat32(row, channels[c].width, &lower[c], &upper[c], input);
                                    input += channels[c].width;
                                }
                            }
                        }
                        else if (shape.size() == 2)
                        {
                            if (shape[0] != original.height || shape[1] != original.width)
                            {
                                std::cout << "Incompatible size of '" << test->path[p] << "' image!" << std::endl;
                                return false;
                            }
                            for (size_t y = 0; y < original.height; ++y)
                            {
                                const uint8_t* row = original.Row<uint8_t>(y);
                                const float lo = 0.0f, hi = 255.0f;
                                ::SimdUint8ToFloat32(row, original.width, &lo, &hi, input);
                                input += original.width;
                            }
                        }
                        else
                        {
                            std::cout << "Can't map to source '" << test->path[p] << "' image!" << std::endl;
                            return false;
                        }
                    }
                }
                _tests.push_back(test);
            }
            return true;
        }

        bool CreateTestList()
        {
#ifdef SYNET_TEST_FIRST_RUN 
            if (_options.enable & ENABLE_FIRST)
                return CreateTestList(_firsts[0]);
#endif
#ifdef SYNET_TEST_SECOND_RUN 
            if (_options.enable & ENABLE_SECOND)
                return CreateTestList(_seconds[0]);
#endif
            return false;
        }

        bool DebugPrint(Network& network, size_t i) const
        {
            if (_options.debugPrint)
            {
                String name = Options::FullName(network.Name(), network.Type()) + "_f" + std::to_string(_options.tensorFormat) +
                    "_b" + std::to_string(_options.batchSize) + "_i" + std::to_string(i) + ".log";
                String path = MakePath(_options.outputDirectory, name);
                std::ofstream log(path);
                if (log.is_open())
                {
                    network.DebugPrint(_tests[i]->input, log, _options.debugPrint, _options.debugPrintFirst, _options.debugPrintLast, _options.debugPrintPrecision);
                    log.close();
                }
                else
                {
                    std::cout << "Can't open '" << path << "' file!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool AnnotateRegions(const Network& network, const String& inputPath) const
        {
            if (_options.annotateRegions)
            {
                View image;
                if (!LoadImage(inputPath, image))
                {
                    std::cout << "Can't read '" << inputPath << "' image!" << std::endl;
                    return false;
                }
                Regions regions = network.GetRegions(image.Size(), _options.regionThreshold, _options.regionOverlap);
                uint32_t white = 0xFFFFFFFF;
                for (size_t i = 0; i < regions.size(); ++i)
                {
                    const Region& region = regions[i];
                    ptrdiff_t l = ptrdiff_t(region.x - region.w / 2);
                    ptrdiff_t t = ptrdiff_t(region.y - region.h / 2);
                    ptrdiff_t r = ptrdiff_t(region.x + region.w / 2);
                    ptrdiff_t b = ptrdiff_t(region.y + region.h / 2);
                    Simd::DrawRectangle(image, l, t, r, b, white);
                }
                String outputPath = MakePath(_options.outputDirectory, Options::FullName(network.Name(), network.Type()) + "_" + GetNameByPath(inputPath));
                if (!SaveImage(image, outputPath))
                {
                    std::cout << "Can't write '" << outputPath << "' image!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        String TestFailedMessage(const TestData& test, size_t index, size_t thread)
        {
            std::stringstream ss;
            ss << "At thread " << thread << " test " << index << " '" << test.path[0];
            for (size_t k = 1; k < test.path.size(); ++k)
                ss << ", " << test.path[k];
            ss << "' is failed!";
            return ss.str();
        }

        bool Compare(float a, float b, float t) const
        {
            float d = ::fabs(a - b);
            return d <= t || d / std::max(::fabs(a), ::fabs(b)) <= t;
        }

        bool Compare(const Tensor& f, const Tensor& s, const Shape & i, size_t d, const String & m) const
        {
            using Synet::Detail::DebugPrint;
            float _f = f.CpuData(i)[0], _s = s.CpuData(i)[0];
            if (!Compare(_f, _s, _options.compareThreshold))
            {
                std::cout << m << std::endl << std::fixed;
                std::cout << "Dst[" << d << "]" << DebugPrint(f.Shape()) << " at ";
                std::cout << DebugPrint(i) << " : " << _f << " != " << _s << std::endl;
                return false;
            }
            return true;
        }

        void PrintError(const Difference & d, size_t i, const String& msg) const
        {
            using Synet::Detail::DebugPrint;
            const Difference::Statistics& s = d.GetStatistics();
            const Difference::Specific& e = s.exceed;
            const Difference::Specific& m = s.max;
            std::cout << msg << std::endl << std::fixed;
            std::cout << "Dst[" << i << "]" << DebugPrint(d.GetShape()) << " at ";
            std::cout << DebugPrint(e.index) << " : diff = " << e.diff << " (" << e.first << " != " << e.second << ")";
            std::cout << ", num = " << double(e.count) / s.count << "(" << e.count << ")";
            std::cout << ", avg = " << s.mean << ", std = " << s.sdev << ", abs = " << s.adev;
            std::cout << ", max " << DebugPrint(m.index) << " diff = " << s.max.diff << " (" << m.first << " != " << m.second << ")" << std::endl;
        }

        bool CompareResults(const TestData& test, size_t index, size_t thread)
        {
            using Synet::Detail::DebugPrint;
            const Output& output = test.output[thread];
            String failed = TestFailedMessage(test, index, thread);
            if (output.first.size() != output.second.size())
            {
                std::cout << failed << std::endl;
                std::cout << "Dst count : " << output.first.size() << " != " << output.second.size() << std::endl;
                return false;
            }
            for (size_t d = 0; d < output.first.size(); ++d)
            {
                const Tensor& f = output.first[d];
                const Tensor& s = output.second[d];
                if (f.Shape() != s.Shape())
                {
                    std::cout << failed << std::endl;
                    std::cout << "Dst[" << d << "] shape : " << DebugPrint(f.Shape()) << " != " << DebugPrint(s.Shape()) << std::endl;
                    return false;
                }
                Difference difference(f, s);
                if (difference.Valid())
                {
                    if (!difference.Estimate(_options.compareThreshold, _options.compareQuantile))
                    {
                        PrintError(difference, d, failed);
                        return false;
                    }
                    continue;
                }
                switch (f.Count())
                {
                case 1:
                    for (size_t n = 0; n < f.Axis(0); ++n)
                        if (!Compare(f, s, Shp(n), d, failed))
                            return false;
                    break;
                case 2:
                    for (size_t n = 0; n < f.Axis(0); ++n)
                        for (size_t c = 0; c < f.Axis(1); ++c)
                            if (!Compare(f, s, Shp(n, c), d, failed))
                                return false;
                    break;
                case 3:
                    for (size_t n = 0; n < f.Axis(0); ++n)
                        for (size_t c = 0; c < f.Axis(1); ++c)
                            for (size_t y = 0; y < f.Axis(2); ++y)
                                if (!Compare(f, s, Shp(n, c, y), d, failed))
                                    return false;
                    break;
                case 4:
                    for (size_t n = 0; n < f.Axis(0); ++n)
                        for (size_t c = 0; c < f.Axis(1); ++c)
                            for (size_t y = 0; y < f.Axis(2); ++y)
                                for (size_t x = 0; x < f.Axis(3); ++x)
                                    if (!Compare(f, s, Shp(n, c, y, x), d, failed))
                                        return false;
                    break;
                default:
                    std::cout << "Error! Dst has unsupported shape " << Synet::Detail::DebugPrint(f.Shape()) << std::endl;
                    return false;
                }
            }
            return true;
        }

        String ProgressString(size_t current, size_t total)
        {
            std::stringstream progress;
            progress << "Test progress : " << ToString(100.0 * current / total, 1) << "% ";
            if (_threads.size() > 1)
            {
                const size_t m = 10;
                progress << "[ ";
                for (size_t t = 0, n = std::min(m, _threads.size()); t < n; ++t)
                    progress << ToString(100.0 * _threads[t].current / total, 1) << "% ";
                if (_threads.size() > m)
                    progress << "... ";
                progress << "] ";
            }
            _progressMessageSizeMax = std::max(_progressMessageSizeMax, progress.str().size());
            return progress.str();
        }

        bool SingleThreadComparison()
        {
            size_t repeats = std::max<size_t>(1, _options.repeatNumber), total = _tests.size() * repeats, current = 0;
            for (size_t i = 0; i < _tests.size(); ++i)
            {
                TestData& test = *_tests[i];
                for (size_t r = 0; r < repeats; ++r, ++current)
                {
                    std::cout << ProgressString(current, total) << std::flush;
#ifdef SYNET_TEST_FIRST_RUN
                    if (_options.enable & ENABLE_FIRST)
                    {
                        Copy(_firsts[0].Predict(test.input), test.output[0].first);
                        if (r == 0)
                        {
                            if (!DebugPrint(_firsts[0], i))
                                return false;
                            if (!AnnotateRegions(_firsts[0], test.path[0]))
                                return false;
                        }
                    }
#endif
#ifdef SYNET_TEST_SECOND_RUN
                    if (_options.enable & ENABLE_SECOND)
                    {
                        Copy(_seconds[0].Predict(test.input), test.output[0].second);
                        if (r == 0)
                        {
                            if (!DebugPrint(_seconds[0], i))
                                return false;
                            if (!AnnotateRegions(_seconds[0], test.path[0]))
                                return false;
                        }
                    }
#endif
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
                    if (r == 0 && _options.enable == (ENABLE_FIRST | ENABLE_SECOND) && !CompareResults(test, i, 0))
                        return false;
#endif             
                    std::cout << " \r" << std::flush;
                }
            }
#ifdef SYNET_TEST_FIRST_RUN
            if (_options.enable & ENABLE_FIRST)
                _options.firstMemoryUsage = _firsts[0].MemoryUsage();
#endif
#ifdef SYNET_TEST_SECOND_RUN
            if (_options.enable & ENABLE_SECOND)
                _options.secondMemoryUsage = _seconds[0].MemoryUsage();
#endif
            return PrintFinishMessage();
        }

        bool MultiThreadsComparison()
        {
            size_t current = 0, total = _options.repeatNumber ?
                _tests.size() * _options.repeatNumber : size_t(_options.executionTime * 1000);
            _threads.resize(_options.TestThreads());
            for (size_t t = 0; t < _threads.size(); ++t)
                _threads[t].thread = std::thread(TestThread, this, t, total);

            while (current < total)
            {
                bool first = true, second = true;
                current = total;
                for (size_t t = 0; t < _threads.size(); ++t)
                {
                    current = std::min(current, _threads[t].current);
                    first = first && _threads[t].first;
                    second = second && _threads[t].second;
                }
                if (first)
                {
                    _notifiedFirst = true;
                    _startFirst.notify_all();
                }
                if (second)
                {
                    _notifiedSecond = true;
                    _startSecond.notify_all();
                }
                std::cout << ProgressString(current, total) << std::flush;
                Sleep(1);
                std::cout << " \r" << std::flush;
            }

            _options.secondMemoryUsage = 0;
            for (size_t t = 0; t < _threads.size(); ++t)
            {
                if (_threads[t].thread.joinable())
                    _threads[t].thread.join();
                for (size_t i = 0; i < _tests.size(); ++i)
                {
                    TestData& test = *_tests[i];
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
                    if (_options.enable == (ENABLE_FIRST | ENABLE_SECOND) && !CompareResults(test, i, t))
                        return false;
#endif 
                }
#ifdef SYNET_TEST_FIRST_RUN
                if (_options.enable & ENABLE_FIRST)
                    _options.firstMemoryUsage += _firsts[t].MemoryUsage();
#endif 
#ifdef SYNET_TEST_SECOND_RUN
                if (_options.enable & ENABLE_SECOND)
                    _options.secondMemoryUsage += _seconds[t].MemoryUsage();
#endif            
            }
            return PrintFinishMessage();
        }

        static void TestThread(Comparer* comparer, size_t thread, size_t total)
        {
            const Options& options = comparer->_options;
            size_t current = 0, networks = 1;
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
            if (options.enable == (ENABLE_FIRST | ENABLE_SECOND))
                networks = 2;
#endif 
#ifdef SYNET_TEST_FIRST_RUN 
            if (options.enable & ENABLE_FIRST)
            {
                if (thread && !comparer->InitNetwork(options.firstModel, options.firstWeight, comparer->_firsts[thread]))
                    ::exit(0);
                comparer->_threads[thread].first = true;
                std::mutex mutex;
                std::unique_lock<std::mutex> lock(mutex);
                while (!comparer->_notifiedFirst)
                    comparer->_startFirst.wait(lock);
                comparer->_notifiedFirst = false;
                if (options.repeatNumber)
                {
                    for (size_t i = 0; i < comparer->_tests.size(); ++i)
                    {
                        TestData& test = *comparer->_tests[i];
                        for (size_t r = 0; r < options.repeatNumber; ++r, ++current)
                        {
                            Copy(comparer->_firsts[thread].Predict(test.input), test.output[thread].first);
                            comparer->_threads[thread].current = current / networks;
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = Time(), duration = 0;
                    while (duration < options.executionTime)
                    {
                        for (size_t i = 0; i < comparer->_tests.size() && (duration < options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *comparer->_tests[i];
                            Copy(comparer->_firsts[thread].Predict(test.input), test.output[thread].first);
                            duration = Time() - start;
                            comparer->_threads[thread].current = std::min(total, size_t(duration * 1000)) / networks;
                        }
                        canstop = true;
                    }
                }
                comparer->_firsts[thread].Free();
            }
#endif
#ifdef SYNET_TEST_SECOND_RUN
            if (options.enable & ENABLE_SECOND)
            {
                if (thread && !comparer->InitNetwork(options.secondModel, options.secondWeight, comparer->_seconds[thread]))
                    ::exit(0);
                comparer->_threads[thread].second = true;
                std::mutex mutex;
                std::unique_lock<std::mutex> lock(mutex);
                while (!comparer->_notifiedSecond)
                    comparer->_startSecond.wait(lock);
                comparer->_notifiedSecond = false;
                if (options.repeatNumber)
                {
                    for (size_t i = 0; i < comparer->_tests.size(); ++i)
                    {
                        TestData& test = *comparer->_tests[i];
                        for (size_t r = 0; r < options.repeatNumber; ++r, ++current)
                        {
                            Copy(comparer->_seconds[thread].Predict(test.input), test.output[thread].second);
                            comparer->_threads[thread].current = current / networks;
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = Time(), duration = 0;
                    while (duration < options.executionTime)
                    {
                        for (size_t i = 0; i < comparer->_tests.size() && (duration < options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *comparer->_tests[i];
                            Copy(comparer->_seconds[thread].Predict(test.input), test.output[thread].second);
                            duration = Time() - start;
                            comparer->_threads[thread].current = (total * (networks - 1) + std::min(total, size_t(duration * 1000))) / networks;
                        }
                        canstop = true;
                    }
                }
                comparer->_seconds[thread].Free();
            }
#endif           
            comparer->_threads[thread].current = total;
        }

        inline void Sleep(unsigned int miliseconds)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
        }

        static inline void Copy(const Tensors & src, Tensors & dst)
        {
            dst.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                dst[i].Clone(src[i]);
        }
    };
}
