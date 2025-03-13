/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Utils/Difference.h"

#include "TestCommon.h"
#include "TestUtils.h"
#include "TestOptions.h"
#include "TestPerformance.h"
#include "TestSynet.h"
#include "TestOutputComparer.h"

#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#endif

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
            , _nextProgressUpdate(Cpl::Miliseconds(Cpl::TimeCounter()))
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
            String debug;
            Thread() : current(0), first(false), second(false) {}
        };
        std::vector<Thread> _threads;
        std::condition_variable _startFirst, _startSecond;
        bool _notifiedFirst, _notifiedSecond;
        size_t _progressMessageSizeMax;
        double _nextProgressUpdate;

        typedef Simd::Pixel::Bgra32 Color;
        typedef std::vector<Color> Colors;
        mutable Colors _colors;

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

        bool PrintFinishMessage(int64_t start) const
        {
            std::stringstream msg;
            msg << "Tests are finished successfully in " << Test::ExecTimeStr(start) << ".";
            std::cout << ExpandRight(msg.str(), _progressMessageSizeMax) << std::endl << std::endl;
            return true;
        }

        bool LoadTestParam()
        {
            if (!_param.Load(_options.testParam))
                SYNET_ERROR("Can't load file '" << _options.testParam << "' !");
            return true;
        }

        bool InitNetwork(String model, String weight, Network& network) const
        {
            if (network.Name() != "OnnxRuntime")
            {
                if (!Cpl::FileExists(model))
                {
                    String alt = Cpl::ChangeExtension(model, ".dsc");
                    if (alt != model && network.Name() != "Synet")
                    {
                        if (!Cpl::FileExists(alt))
                            SYNET_ERROR("Files '" << model << "' and '" << alt << "' are not exist!");
                        model = alt;
                    }
                    else
                        SYNET_ERROR("File '" << model << "' is not exist!");
                }
            }
            if (!Cpl::FileExists(weight))
            {
                String alt = Cpl::ChangeExtension(weight, ".dat");
                if (alt != weight && network.Name() != "Synet")
                {
                    if (!Cpl::FileExists(alt))
                        SYNET_ERROR("Files '" << weight << "' and '" << alt << "' are not exist!");
                    weight = alt;
                }
                else
                    SYNET_ERROR("File '" << weight << "' is not exist!");
            }
            Network::Options options(_options.outputDirectory, _options.workThreads, _options.consoleSilence, _options.batchSize,
                (_options.performanceLog & network.PerfLogMask()), _options.debugPrint, _options.regionThreshold, _options.bf16);
            if (!network.Init(model, weight, options, _param()))
                SYNET_ERROR("Can't load " << network.Name() << " from '" << model << "' and '" << weight << "' !");
            Shape shape = network.SrcShape(0);
            if (!(shape[1] == 1 || shape[1] == 3 || _param().inputType() == "binary"))
                SYNET_ERROR("Wrong " << network.Name() << " network model channels count '" << shape[1] << " !");
            return true;
        }

        bool InitNetworkFirst()
        {
#if defined(SYNET_TEST_FIRST_RUN)
            if (_options.enable & ENABLE_FIRST)
            {
                if (!InitNetwork(_options.firstModel, _options.firstWeight, _firsts[0]))
                    return false;
                _options.firstName = _firsts[0].Name();
                _options.firstType = _firsts[0].Type();
            }
#endif
            return true;
        }

        bool InitNetworkSecond()
        {
#if defined(SYNET_TEST_SECOND_RUN)
            if (_options.enable & ENABLE_SECOND)
            {
                if (!InitNetwork(_options.secondModel, _options.secondWeight, _seconds[0]))
                    return false;
                _options.secondName = _seconds[0].Name();
                _options.secondType = _seconds[0].Type();
            }
#endif
            return true;
        }

        bool InitNetworks()
        {
            if (_options.reverseExecution)
            {
                if (!(InitNetworkSecond() && InitNetworkFirst()))
                    return false;
            }
            else
            {
                if (!(InitNetworkFirst() && InitNetworkSecond()))
                    return false;
            }
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
            if (_options.enable == (ENABLE_FIRST | ENABLE_SECOND))
            {
                if (_firsts[0].SrcCount() != _seconds[0].SrcCount())
                    SYNET_ERROR("Networks have difference source number: " << _firsts[0].SrcCount() << " != " << _seconds[0].SrcCount());
                for (size_t s = 0; s < _firsts[0].SrcCount(); ++s)
                {
                    const Shape& os = _firsts[0].SrcShape(s);
                    const Shape& ss = _seconds[0].SrcShape(s);
                    if (os != ss)
                    {
                        std::stringstream err;
                        err << "Networks have difference Src[" << s << "] size: ";
                        err << _firsts[0].Name() << " {" << os[0];
                        for (size_t j = 1; j < os.size(); ++j)
                            err << ", " << os[j];
                        err << "} != " << _seconds[0].Name() << " {" << ss[0];
                        for (size_t j = 1; j < ss.size(); ++j)
                            err << ", " << ss[j];
                        err << "} !";
                        SYNET_ERROR(err.str());
                    }
                }
            }
#endif
            return true;
        }

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
                SYNET_ERROR("Can't create output directory '" << _options.outputDirectory << "' !");
            return true;
        }

        bool RequiredExtension(const String& name)
        {
            String ext = ExtensionByPath(name);
            static const char* EXTS[] = { "JPG", "jpeg", "jpg", "png", "ppm", "pgm", "bin" };
            for (size_t i = 0, n = sizeof(EXTS) / sizeof(EXTS[0]); i < n; ++i)
                if (ext == EXTS[i])
                    return true;
            return false;
        }

        void ResizeImage(const View& src, View& dst) const
        {
            if (_param().smartResize())
            {
                Simd::Fill(dst, 0);
                ptrdiff_t d = src.width * dst.height - src.height * dst.width;
                size_t w = d > 0 ? dst.width : src.width * dst.height / src.height;
                size_t h = d < 0 ? dst.height : src.height * dst.width / src.width;
                Simd::Resize(src, dst.Region(Size(w, h), View::MiddleLeft).Ref(), SimdResizeMethodArea);
            }
            else
            {
                Simd::Resize(src, dst, SimdResizeMethodArea);
            }
        }

        bool RequiredInput(size_t index) const
        {
            if (_param().input().size())
                return _param().input()[index].from().empty();
            return true;
        }

        size_t RequiredInputNumber(const Network& network) const
        {
            if (_param().input().size())
            {
                size_t count = 0;
                for (size_t i = 0; i < _param().input().size(); ++i)
                    if (_param().input()[i].from().empty())
                        count++;
                return count;
            }
            return network.SrcCount();
        }

        bool CreateTestListImages(const Network& network, const String& directory)
        {
            StringList images = GetFileList(directory, _options.imageFilter, true, false);
            images.sort();

            Strings names;
            names.reserve(images.size());
            size_t curr = 0, rN = RequiredInputNumber(network), imgBeg = _options.imageBegin * rN, imgEnd = _options.imageEnd * rN;
            for (StringList::const_iterator it = images.begin(); it != images.end(); ++it)
            {
                if (RequiredExtension(*it))
                {
                    if (curr >= imgBeg && curr < imgEnd)
                        names.push_back(*it);
                    curr++;
                }
            }
            size_t sN = network.SrcCount(), bN = _options.batchSize;
            size_t tN = names.size() / bN / rN;
            if (tN == 0)
                SYNET_ERROR("There is no one image in '" << directory << "' for '" << _options.imageFilter << "' filter!");

            Floats lower = _param().lower(), upper = _param().upper();
            _tests.clear();
            _tests.reserve(tN);
            for (size_t t = 0; t < tN; ++t)
            {
                TestDataPtr test(new TestData());
                test->path.resize(bN * rN);
                test->input.resize(sN);
                test->output.resize(_options.TestThreads());
                Points imageSizes;
                for (size_t s = 0, r = 0; s < sN; ++s)
                {
                    Tensor& tensor = test->input[s];
                    tensor.Reshape(network.SrcType(s), network.SrcShape(s), Synet::TensorFormatUnknown);
                    if (RequiredInput(s))
                    {
                        float* input = tensor.Data<float>();
                        for (size_t b = 0; b < bN; ++b)
                        {
                            size_t p = r * bN + b;
                            test->path[p] = MakePath(directory, names[(t * bN + b) * rN + r]);
                            View original;
                            if (!LoadImage(test->path[p], original))
                                SYNET_ERROR("Can't read '" << test->path[p] << "' image!");
                            imageSizes.push_back(original.Size());
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
                                ResizeImage(converted, resized);

                                Simd::SynetSetInput(resized, lower.data(), upper.data(), input, shape[1], SimdTensorFormatNchw, _param().order() == "rgb");
                                input += shape[1] * shape[2] * shape[3];
                            }
                            else if (shape.size() == 2)
                            {
                                if (shape[0] != original.height || shape[1] != original.width)
                                    SYNET_ERROR("Incompatible size of '" << test->path[p] << "' image!");
                                for (size_t y = 0; y < original.height; ++y)
                                {
                                    const uint8_t* row = original.Row<uint8_t>(y);
                                    const float lo = 0.0f, hi = 255.0f;
                                    ::SimdUint8ToFloat32(row, original.width, &lo, &hi, input);
                                    input += original.width;
                                }
                            }
                            else
                                SYNET_ERROR("Can't map to source '" << test->path[p] << "' image!");
                        }
                        r++;
                    }
                    else if (_param().input().size() && _param().input()[s].from() == "image_size")
                    {
                        if (tensor.GetType() == Synet::TensorType32i)
                        {
                            for (size_t b = 0; b < bN; ++b)
                            {
                                tensor.Data<int32_t>(Shp(b, 0))[0] = (int32_t)imageSizes[b].y;
                                tensor.Data<int32_t>(Shp(b, 0))[1] = (int32_t)imageSizes[b].x;
                            }
                        }
                    }
                    else
                    {
                        SYNET_ERROR("Can't process input parameter 'from'!");
                    }
                }
                _tests.push_back(test);
            }
            return true;
        }

        bool CreateTestListBinary(const Network& network, const String& directory)
        {
            StringList files = GetFileList(directory, _options.imageFilter, true, false);
            files.sort();

            Strings names;
            names.reserve(files.size());
            for (StringList::const_iterator it = files.begin(); it != files.end(); ++it)
                if (RequiredExtension(*it))
                    names.push_back(*it);

            size_t sN = network.SrcCount(), bN = _options.batchSize;
            if (names.size() != sN)
                SYNET_ERROR("The number of binary files " << names.size() << " is differ from number of network sources " << sN << " in '" << directory << "' !");

            _tests.clear();
            for (size_t n = 0; n < sN; ++n)
            {
                size_t sS = network.SrcSize(n);
                Vector data;
                String path = MakePath(directory, names[n]);
                if (!Synet::LoadBinaryData(path, data))
                    SYNET_ERROR("Can't load binary file '" << path << "' !");
                size_t tN = data.size() / sS, tB = _options.binaryBegin / _options.batchSize, tE = std::min(_options.binaryEnd / _options.batchSize, tN);
                if (tB >= tE)
                    SYNET_ERROR("Wrong parameters: -bb=" << _options.binaryBegin << ", -be=" << _options.binaryEnd << ", binary size = " << tN << " !");
                tN = tE - tB;
                if (tN == 0)
                    SYNET_ERROR("The binary file '" << path << "' is too small!");
                if (n == 0)
                    _tests.resize(tN);
                else if (_tests.size() != tN)
                    SYNET_ERROR("The binary files are not compartible!");
                for (size_t i = 0; i < tN; i += 1)
                {
                    size_t offs = (tB + i) * sS;
                    TestDataPtr& test = _tests[i];
                    if (n == 0)
                    {
                        test.reset(new TestData());
                        test->path.resize(bN * sN);
                        test->input.resize(sN);
                        test->output.resize(_options.TestThreads());
                    }
                    Tensor& tensor = test->input[n];
                    tensor.Reshape(Synet::TensorType32f, network.SrcShape(n), Synet::TensorFormatUnknown);
                    memcpy(tensor.Data<float>(), data.data() + offs, sS * sizeof(float));
                }
            }
            return true;
        }

        bool CreateTestList(const Network& network)
        {
            String imageDirectory = _options.imageDirectory;
            if (imageDirectory.empty())
                imageDirectory = Test::MakePath(DirectoryByPath(_options.testParam), _param().images());
            if (!DirectoryExists(imageDirectory))
                SYNET_ERROR("Test image directory '" << imageDirectory << "' is not exists!");
            if (_param().inputType() == "images")
                return CreateTestListImages(network, imageDirectory);
            else if (_param().inputType() == "binary")
                return CreateTestListBinary(network, imageDirectory);
            else
                SYNET_ERROR("Unknown input type '" << _param().inputType() << "' !");
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
                    SYNET_ERROR("Can't open '" << path << "' file!");
            }
            return true;
        }

        Color GetColor(size_t index) const
        {
            if (index >= _colors.size())
            {
                if (_colors.empty())
                {
                    _colors.push_back(Color(0xFF, 0xFF, 0xFF));
                    _colors.push_back(Color(0x00, 0x00, 0xFF));
                    _colors.push_back(Color(0x00, 0xFF, 0x00));
                    _colors.push_back(Color(0xFF, 0x00, 0x00));
                    _colors.push_back(Color(0x00, 0xFF, 0xFF));
                    _colors.push_back(Color(0xFF, 0xFF, 0x00));
                    _colors.push_back(Color(0xFF, 0x00, 0xFF));
                }
                while (index >= _colors.size())
                    _colors.push_back(Color(::rand(), ::rand(), ::rand()));
            }
            return _colors[index];
        }

        bool AnnotateRegions(const Network& network, const String& inputPath) const
        {
            if (_options.annotateRegions && _param().inputType() == "images")
            {
                View image;
                if (!LoadImage(inputPath, image))
                    SYNET_ERROR("Can't read '" << inputPath << "' image!");
                Regions regions = network.GetRegions(image.Size(), _options.regionThreshold, _options.regionOverlap);
                for (size_t i = 0; i < regions.size(); ++i)
                {
                    const Region& region = regions[i];
                    ptrdiff_t l = ptrdiff_t(region.x - region.w / 2);
                    ptrdiff_t t = ptrdiff_t(region.y - region.h / 2);
                    ptrdiff_t r = ptrdiff_t(region.x + region.w / 2);
                    ptrdiff_t b = ptrdiff_t(region.y + region.h / 2);
                    Simd::DrawRectangle(image, l, t, r, b, GetColor(region.id));
                }
                String outputPath = MakePath(_options.outputDirectory, Options::FullName(network.Name(), network.Type()) + "_" + GetNameByPath(inputPath));
                if (!SaveImage(image, outputPath))
                    SYNET_ERROR("Can't write '" << outputPath << "' image!");
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

        bool CompareResults(const TestData& test, size_t index, size_t thread)
        {
            const Output& output = test.output[thread];
            String failed = TestFailedMessage(test, index, thread);
            OutputComparer outputComparer(_options, _param(), test.input[0].Shape(), output.first);
            return outputComparer.Compare(output.first, output.second, failed);
        }

        String ProgressString(size_t current, size_t total)
        {
            const size_t m = 10, n = std::min(m, _threads.size());
            std::stringstream progress;
            progress << "Test progress : " << ToString(100.0 * current / total, 1) << "% ";
            if (_threads.size() > 1)
            {
                progress << "[ ";
                for (size_t t = 0; t < n; ++t)
                    progress << ToString(100.0 * _threads[t].current / total, 1) << "% ";
                if (_threads.size() > m)
                    progress << "... ";
                progress << "] ";
            }
#if defined(__linux__) && 0
            if (_threads.size())
            {
                progress << " [ ";
                for (size_t t = 0; t < n; ++t)
                    progress << _threads[t].debug << " ";
                progress << "] ";
            }
            progress << CoreFreqInfo();
            double time = Cpl::Miliseconds(Cpl::TimeCounter());
            if (time >= _nextProgressUpdate)
            {
                progress << std::endl;
                _nextProgressUpdate = time + 300;
            }
#endif
            _progressMessageSizeMax = std::max(_progressMessageSizeMax, progress.str().size());
            return progress.str();
        }

        bool SingleThreadRunFirst(size_t index, size_t repeat)
        {
#ifdef SYNET_TEST_FIRST_RUN
            if (_options.enable & ENABLE_FIRST)
            {
                TestData& test = *_tests[index];
                Copy(_firsts[0].Predict(test.input), test.output[0].first);
                if (repeat == 0)
                {
                    if (!DebugPrint(_firsts[0], index))
                        return false;
                    if (!AnnotateRegions(_firsts[0], test.path[0]))
                        return false;
                }
            }
#endif
            return true;
        }

        bool SingleThreadRunSecond(size_t index, size_t repeat)
        {
#ifdef SYNET_TEST_SECOND_RUN
            if (_options.enable & ENABLE_SECOND)
            {
                TestData& test = *_tests[index];
#if 0
                if (repeat == 0)
                    if (!DebugPrint(_seconds[0], index))
                        return false;
#endif
                Copy(_seconds[0].Predict(test.input), test.output[0].second);
                if (repeat == 0)
                {
                    if (!DebugPrint(_seconds[0], index))
                        return false;
                    if (!AnnotateRegions(_seconds[0], test.path[0]))
                        return false;
                }
            }
#endif
            return true;
        }

        bool SingleThreadComparison()
        {
            PinThread(0);
            int64_t start = Cpl::TimeCounter();
            size_t repeats = std::max<size_t>(1, _options.repeatNumber), total = _tests.size() * repeats, current = 0;
            for (size_t i = 0; i < _tests.size(); ++i)
            {
                TestData& test = *_tests[i];
                for (size_t r = 0; r < repeats; ++r, ++current)
                {
                    std::cout << ProgressString(current, total) << std::flush;
                    if (_options.reverseExecution)
                    {
                        if(!(SingleThreadRunSecond(i, r) && SingleThreadRunFirst(i, r)))
                            return false;
                    }
                    else
                    {
                        if (!(SingleThreadRunFirst(i, r) && SingleThreadRunSecond(i, r)))
                            return false;
                    }
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
            return PrintFinishMessage(start);
        }

        bool MultiThreadsComparison()
        {
            PinThread(SimdCpuInfo(SimdCpuInfoThreads) - 1);
            int64_t start = Cpl::TimeCounter();
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
            return PrintFinishMessage(start);
        }

        void MultuThreadRunFirst(size_t thread, size_t total, size_t& current, size_t networks, size_t second)
        {
#ifdef SYNET_TEST_FIRST_RUN 
            if (_options.enable & ENABLE_FIRST)
            {
                if (thread && !InitNetwork(_options.firstModel, _options.firstWeight, _firsts[thread]))
                {
                    CPL_LOG_SS(Error, "Can't initialize network " << Options::FullName(_firsts[thread].Name(), _firsts[thread].Type()) << " for " << thread << " thread!");
                    ::exit(0);
                }
                _threads[thread].first = true;
                std::mutex mutex;
                std::unique_lock<std::mutex> lock(mutex);
                while (!_notifiedFirst)
                    _startFirst.wait(lock);
                _notifiedFirst = false;
                if (_options.repeatNumber)
                {
                    for (size_t i = 0; i < _tests.size(); ++i)
                    {
                        TestData& test = *_tests[i];
                        for (size_t r = 0; r < _options.repeatNumber; ++r, ++current)
                        {
                            Copy(_firsts[thread].Predict(test.input), test.output[thread].first);
                            _threads[thread].current = current / networks;
                            _threads[thread].debug = CoreFreqInfo();
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = Cpl::Time(), duration = 0;
                    while (duration < _options.executionTime)
                    {
                        for (size_t i = 0; i < _tests.size() && (duration < _options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *_tests[i];
                            Copy(_firsts[thread].Predict(test.input), test.output[thread].first);
                            duration = Cpl::Time() - start;
                            _threads[thread].current = (total * (networks - 1) * second +
                                std::min(total, size_t(duration * 1000))) / networks;
                            _threads[thread].debug = CoreFreqInfo();
                        }
                        canstop = true;
                    }
                }
                _firsts[thread].Free();
            }
#endif
        }

        void MultuThreadRunSecond(size_t thread, size_t total, size_t& current, size_t networks, size_t second)
        {
#ifdef SYNET_TEST_SECOND_RUN 
            if (_options.enable & ENABLE_SECOND)
            {
                if (thread && !InitNetwork(_options.secondModel, _options.secondWeight, _seconds[thread]))
                {
                    CPL_LOG_SS(Error, "Can't initialize network " << Options::FullName(_seconds[thread].Name(), _seconds[thread].Type()) << " for " << thread << " thread!");
                    ::exit(0);
                }
                _threads[thread].second = true;
                std::mutex mutex;
                std::unique_lock<std::mutex> lock(mutex);
                while (!_notifiedSecond)
                    _startSecond.wait(lock);
                _notifiedSecond = false;
                if (_options.repeatNumber)
                {
                    for (size_t i = 0; i < _tests.size(); ++i)
                    {
                        TestData& test = *_tests[i];
                        for (size_t r = 0; r < _options.repeatNumber; ++r, ++current)
                        {
                            Copy(_seconds[thread].Predict(test.input), test.output[thread].second);
                            _threads[thread].current = current / networks;
                            _threads[thread].debug = CoreFreqInfo();
                        }
                    }
                }
                else
                {
                    bool canstop = false;
                    double start = Cpl::Time(), duration = 0;
                    while (duration < _options.executionTime)
                    {
                        for (size_t i = 0; i < _tests.size() && (duration < _options.executionTime || !canstop); ++i)
                        {
                            TestData& test = *_tests[i];
                            Copy(_seconds[thread].Predict(test.input), test.output[thread].second);
                            duration = Cpl::Time() - start;
                            _threads[thread].current = (total * (networks - 1) * second +
                                std::min(total, size_t(duration * 1000))) / networks;
                            _threads[thread].debug = CoreFreqInfo();
                        }
                        canstop = true;
                    }
                }
                _seconds[thread].Free();
            }
#endif
        }

        static void TestThread(Comparer* comparer, size_t thread, size_t total)
        {
            PinThread(thread);
            const Options& options = comparer->_options;
            size_t current = 0, networks = 1;
#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_TEST_SECOND_RUN)
            if (options.enable == (ENABLE_FIRST | ENABLE_SECOND))
                networks = 2;
#endif             
            if (options.reverseExecution)
            {
                comparer->MultuThreadRunSecond(thread, total, current, networks, 0);
                comparer->MultuThreadRunFirst(thread, total, current, networks, 1);
            }
            else
            {
                comparer->MultuThreadRunFirst(thread, total, current, networks, 0);
                comparer->MultuThreadRunSecond(thread, total, current, networks, 1);
            }
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

        static bool PinThread(size_t core)
        {
#if defined(__linux__)
            pthread_t this_thread = pthread_self();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core, &cpuset);
            if (pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset)) 
            {
                CPL_LOG_SS(Warning, "Can't set affinity " << core << " to " << this_thread << " thread : " << std::strerror(errno) << " !");
                return false;
            }
#if 0
            int policy = 0;
            sched_param params;
            if (pthread_getschedparam(this_thread, &policy, &params))
            {
                CPL_LOG_SS(Warning, "Can't get thread parameters of " << this_thread << " thread : " << std::strerror(errno) << " !");
                return false;
            }

            CPL_LOG_SS(Info, "Core : " << core << " Policy: " << policy  << " Priority: " << params.sched_priority << " max: " << sched_get_priority_max(policy) << " min: " << sched_get_priority_min(policy));

            if (pthread_setschedparam(this_thread, policy, &params))
            {
                CPL_LOG_SS(Warning, "Can't set thread parameters of " << this_thread << " thread : " << std::strerror(errno) << " !");
                return false;
            }

            CPL_LOG_SS(Info, "Core : " << core << " Policy: " << policy << " Priority: " << params.sched_priority);
#endif
#endif
            return true;
        }

        static String CoreFreqInfo()
        {
            std::stringstream info;
#if defined(__linux__)
            info << " " << sched_getcpu() << ": " << ToString(double(SimdCpuInfo(SimdCpuInfoCurrentFrequency)) / 1000000000.0, 1) << " GHz.";
#endif
            return info.str();
        }
    };
}
