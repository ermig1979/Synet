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

#include "TestCommon.h"
#include "TestUtils.h"
#include "TestOptions.h"
#include "TestPerformance.h"
#include "TestSynet.h"
#include "TestImage.h"

namespace Test
{
    template<class OtherNetwork> class Comparer
    {
    public:
        Comparer(const Options& options)
            : _options(options)
        {
            assert(_options.testThreads >= 0);
            if (_options.enable & ENABLE_OTHER)
                _others.resize(_options.TestThreads());
            if (_options.enable & ENABLE_SYNET)
                _synets.resize(_options.TestThreads());
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
        std::vector<OtherNetwork> _others;
        std::vector<SynetNetwork> _synets;

        struct Output
        {
            Vectors other;
            Vectors synet;
        };
        typedef std::vector<Output> Outputs;
        struct TestData
        {
            Strings path;
            Vectors input;
            Outputs output;
        };
        typedef std::shared_ptr<TestData> TestDataPtr;
        typedef std::vector<TestDataPtr> TestDataPtrs;
        TestDataPtrs _tests;
        Shape _currents;
        std::vector<std::thread> _threads;

        void PrintStartMessage() const
        {
            std::cout << "Start ";
            if (_options.enable & ENABLE_OTHER)
                std::cout << _others[0].Name() << " ";
            if (_options.enable == (ENABLE_OTHER | ENABLE_SYNET))
                std::cout << "and ";
            if (_options.enable & ENABLE_SYNET)
                std::cout << _synets[0].Name() << " ";
            if (_options.testThreads > 0)
                std::cout << _options.testThreads << "-threads ";
            else
                std::cout << "single-thread ";
            std::cout << (_options.enable == (ENABLE_OTHER | ENABLE_SYNET) ? "comparison " : "performance ");
            std::cout << "tests :" << std::endl;
        }

        bool PrintFinishMessage() const
        {
            std::cout << "Tests are finished successfully!" << std::endl << std::endl;
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
            if (!network.Init(model, weight, _options, _param()))
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
#ifdef SYNET_OTHER_RUN        
            if (_options.enable & ENABLE_OTHER)
            {
                if(!InitNetwork(_options.otherModel, _options.otherWeight, _others[0]))
                    return false;
                _options.otherName = _others[0].Name();
            }
#endif
#ifdef SYNET_SYNET_RUN
            if ((_options.enable & ENABLE_SYNET) && !InitNetwork(_options.synetModel, _options.synetWeight, _synets[0]))
                return false;
#endif            
#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
            if (_options.enable == (ENABLE_OTHER | ENABLE_SYNET))
            {
                if (_others[0].SrcCount() != _synets[0].SrcCount())
                {
                    std::cout << "Networks have difference source number: " <<
                        _others[0].SrcCount() << " != " << _synets[0].SrcCount() << std::endl;
                    return false;
                }
                for (size_t s = 0; s < _others[0].SrcCount(); ++s)
                {
                    const Shape& os = _others[0].SrcShape(s);
                    const Shape& ss = _synets[0].SrcShape(s);
                    if (os != ss)
                    {
                        std::cout << "Networks have difference Src[" << s << "] size: ";
                        std::cout << _others[0].Name() << " {" << os[0];
                        for (size_t j = 1; j < os.size(); ++j)
                            std::cout << ", " << os[j];
                        std::cout << "} != " << _synets[0].Name() << " {" << ss[0];
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
            if (!DirectoryExists(_options.imageDirectory))
            {
                std::cout << "Test image directory '" << _options.imageDirectory << "' is not exists!" << std::endl;
                return false;
            }
            StringList images = GetFileList(_options.imageDirectory, _options.imageFilter, true, false);
            images.sort();
            Strings names(images.begin(), images.end());
            size_t sN = network.SrcCount(), bN = _options.batchSize;
            size_t tN = names.size() / bN / sN;
            if (tN == 0)
            {
                std::cout << "There is no one image in '" << _options.imageDirectory << "' for '" << _options.imageFilter << "' filter!" << std::endl;
                return false;
            }
            StringList::const_iterator name = images.begin();

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
                    test->input[s].resize(network.SrcSize(s));
                    float* input = test->input[s].data();
                    for (size_t b = 0; b < bN; ++b)
                    {
                        size_t p = s * bN + b;
                        test->path[p] = MakePath(_options.imageDirectory, names[(t * bN + b) * sN + s]);
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
#ifdef SYNET_OTHER_RUN 
            if (_options.enable & ENABLE_OTHER)
                return CreateTestList(_others[0]);
#endif
#ifdef SYNET_SYNET_RUN 
            if (_options.enable & ENABLE_SYNET)
                return CreateTestList(_synets[0]);
#endif
            return false;
        }

        bool DebugPrint(Network& network, size_t i) const
        {
            if (_options.debugPrint)
            {
                String name = network.Name() + "_f" + std::to_string(_options.tensorFormat) +
                    "_b" + std::to_string(_options.batchSize) + "_i" + std::to_string(i) + ".log";
                String path = MakePath(_options.outputDirectory, name);
                std::ofstream log(path);
                if (log.is_open())
                {
                    network.DebugPrint(log, _options.debugPrint, _options.debugPrintFirst, _options.debugPrintLast, _options.debugPrintPrecision);
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
                String outputPath = MakePath(_options.outputDirectory, network.Name() + "_" + GetNameByPath(inputPath));
                if (!SaveImage(image, outputPath))
                {
                    std::cout << "Can't write '" << outputPath << "' image!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool Compare(float a, float b, float t) const
        {
            float d = ::fabs(a - b);
            return d <= t || d / std::max(::fabs(a), ::fabs(b)) <= t;
        }

        bool CompareResults(const TestData& test, size_t index, size_t thread)
        {
            const Output& output = test.output[thread];
            if (output.other.size() != output.synet.size())
            {
                std::cout << "Test " << index << " '" << test.path[0];
                for (size_t k = 1; k < test.path.size(); ++k)
                    std::cout << ", " << test.path[k];
                std::cout << "' is failed!" << std::endl;
                std::cout << "Dst count : " << output.other.size() << " != " << output.synet.size() << std::endl;
                return false;
            }
            for (size_t d = 0; d < output.other.size(); ++d)
            {
                if (output.other[d].size() != output.synet[d].size())
                {
                    std::cout << "Test " << index << " '" << test.path[0];
                    for (size_t k = 1; k < test.path.size(); ++k)
                        std::cout << ", " << test.path[k];
                    std::cout << "' is failed!" << std::endl;
                    std::cout << "Dst[" << d << "] size : " << output.other[d].size() << " != " << output.synet[d].size() << std::endl;
                    return false;
                }

                for (size_t j = 0; j < output.synet[d].size(); ++j)
                {
                    if (!Compare(output.other[d][j], output.synet[d][j], _options.threshold))
                    {
                        std::cout << "Test " << index << " '" << test.path[0];
                        for (size_t k = 1; k < test.path.size(); ++k)
                            std::cout << ", " << test.path[k];
                        std::cout << "' is failed!" << std::endl;
                        std::cout << "Dst[" << d << "][" << j << "] : " << output.other[d][j] << " != " << output.synet[d][j] << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        bool SingleThreadComparison()
        {
            size_t repeats = std::max<size_t>(1, _options.repeatNumber), total = _tests.size() * repeats, current = 0;
            for (size_t i = 0; i < _tests.size(); ++i)
            {
                TestData& test = *_tests[i];
                for (size_t r = 0; r < repeats; ++r, ++current)
                {
                    std::cout << "Test progress : " << ToString(100.0 * current / total, 1) << "% " << std::flush;
#ifdef SYNET_OTHER_RUN
                    if (_options.enable & ENABLE_OTHER)
                    {
                        test.output[0].other = _others[0].Predict(test.input);
                        if (r == 0)
                        {
                            if (!DebugPrint(_others[0], i))
                                return false;
                            if (!AnnotateRegions(_others[0], test.path[0]))
                                return false;
                        }
                    }
#endif
#ifdef SYNET_SYNET_RUN
                    if (_options.enable & ENABLE_SYNET)
                    {
                        test.output[0].synet = _synets[0].Predict(test.input);
                        if (r == 0)
                        {
                            if (!DebugPrint(_synets[0], i))
                                return false;
                            if (!AnnotateRegions(_synets[0], test.path[0]))
                                return false;
                        }
                    }
#endif
#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
                    if (r == 0 && _options.enable == (ENABLE_OTHER | ENABLE_SYNET) && !CompareResults(test, i, 0))
                        return false;
#endif             
                    std::cout << " \r" << std::flush;
                }
            }
#ifdef SYNET_SYNET_RUN
            if (_options.enable & ENABLE_SYNET)
                _options.synetMemoryUsage = _synets[0].MemoryUsage();
#endif
            return PrintFinishMessage();
        }

        bool MultiThreadsComparison()
        {
            size_t current = 0, total = _options.repeatNumber ?
                _tests.size() * _options.repeatNumber : size_t(_options.executionTime * 1000);
            _currents.resize(_options.TestThreads(), 0);
            _threads.resize(_options.TestThreads());
            for (size_t t = 0; t < _threads.size(); ++t)
                _threads[t] = std::thread(TestThread, this, t);

            while (current < total)
            {
                current = total;
                for (size_t t = 0; t < _currents.size(); ++t)
                    current = std::min(current, _currents[t]);
                std::cout << "Test progress : " << ToString(100.0 * current / total, 1) << "% " << std::flush;
                Sleep(1);
                std::cout << " \r" << std::flush;
            }

            _options.synetMemoryUsage = 0;
            for (size_t t = 0; t < _threads.size(); ++t)
            {
                if (_threads[t].joinable())
                    _threads[t].join();
                for (size_t i = 0; i < _tests.size(); ++i)
                {
                    TestData& test = *_tests[i];
#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
                    if (_options.enable == (ENABLE_OTHER | ENABLE_SYNET) && !CompareResults(test, i, t))
                        return false;
#endif 
                }
#ifdef SYNET_SYNET_RUN
                if (_options.enable & ENABLE_SYNET)
                    _options.synetMemoryUsage += _synets[t].MemoryUsage();
#endif            
            }
            return PrintFinishMessage();
        }

        static void TestThread(Comparer* comparer, size_t thread)
        {
            const Options& options = comparer->_options;
            size_t current = 0, networks = 1;
#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
            if (options.enable == (ENABLE_OTHER | ENABLE_SYNET))
                networks = 2;
#endif 
#ifdef SYNET_OTHER_RUN 
            if (options.enable & ENABLE_OTHER)
            {
                if (thread && !comparer->InitNetwork(options.otherModel, options.otherWeight, comparer->_others[thread]))
                    ::exit(0);
                if (options.repeatNumber)
                {
                    for (size_t i = 0; i < comparer->_tests.size(); ++i)
                    {
                        TestData& test = *comparer->_tests[i];
                        for (size_t r = 0; r < options.repeatNumber; ++r, ++current)
                        {
                            test.output[thread].other = comparer->_others[thread].Predict(test.input);
                            comparer->_currents[thread] = current / networks;
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = GetTime(), duration = 0;
                    while (duration < options.executionTime)
                    {
                        for (size_t i = 0; i < comparer->_tests.size() && (duration < options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *comparer->_tests[i];
                            test.output[thread].other = comparer->_others[thread].Predict(test.input);
                            duration = GetTime() - start;
                            comparer->_currents[thread] = size_t(duration * 1000) / networks;
                        }
                        canstop = true;
                    }
                }
                comparer->_others[thread].Free();
            }
#endif
#ifdef SYNET_SYNET_RUN
            if (options.enable & ENABLE_SYNET)
            {
                if (thread && !comparer->InitNetwork(options.synetModel, options.synetWeight, comparer->_synets[thread]))
                    ::exit(0);
                if (options.repeatNumber)
                {
                    for (size_t i = 0; i < comparer->_tests.size(); ++i)
                    {
                        TestData& test = *comparer->_tests[i];
                        for (size_t r = 0; r < options.repeatNumber; ++r, ++current)
                        {
                            test.output[thread].synet = comparer->_synets[thread].Predict(test.input);
                            comparer->_currents[thread] = current / networks;
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = GetTime(), duration = 0;
                    while (duration < options.executionTime)
                    {
                        for (size_t i = 0; i < comparer->_tests.size() && (duration < options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *comparer->_tests[i];
                            test.output[thread].synet = comparer->_synets[thread].Predict(test.input);
                            duration = GetTime() - start;
                            comparer->_currents[thread] = size_t((options.executionTime*(networks - 1) + duration) * 1000) / networks;
                        }
                        canstop = true;
                    }
                }
            }
#endif           
            comparer->_currents[thread] = options.repeatNumber ?
                comparer->_tests.size() * options.repeatNumber : size_t(options.executionTime * 1000);
        }

        inline void Sleep(unsigned int miliseconds)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
        }
    };
}


