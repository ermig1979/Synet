/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Cpl/Args.h"

#include "TestCommon.h"
#include "TestUtils.h"
#include "TestNetwork.h"
#include "TestPerformance.h"

namespace Test
{
    struct Stability
    {
        typedef Synet::Network<float> Network;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;

        //-----------------------------------------------------------------------------------------

        struct Options : public Cpl::ArgsParser
        {
            bool result;
            String synetModel;
            String synetWeight;
            String testParam;
            String imageDirectory;
            size_t imageNumber;
            String outputDirectory;
            String logName;
            size_t testThreads;
            size_t repeatNumber;
            int batchSize;
            int debugPrint;
            int debugPrintFirst;
            int debugPrintLast;
            int debugPrintPrecision;
            float compareThreshold;
            bool compactWeight;

            Options(int argc, char* argv[])
                : ArgsParser(argc, argv, true)
                , result(true)
            {
                synetModel = GetArg("-sm", "synet.xml");
                synetWeight = GetArg("-sw", "synet.bin");
                testParam = GetArg("-tp", "param.xml");
                imageDirectory = GetArg("-id", "image");
                imageNumber = Cpl::ToVal<size_t>(GetArg("-in", "10"));
                outputDirectory = GetArg("-od", "output");
                logName = GetArg("-ln", "", false);
                testThreads = Cpl::ToVal<size_t>(GetArg("-tt", "0"));
                repeatNumber = std::max(0, Cpl::ToVal<int>(GetArg("-rn", "1")));
                batchSize = Cpl::ToVal<int>(GetArg("-bs", "1"));
                debugPrint = Cpl::ToVal<int>(GetArg("-dp", "0"));
                debugPrintFirst = Cpl::ToVal<int>(GetArg("-dpf", "5"));
                debugPrintLast = Cpl::ToVal<int>(GetArg("-dpl", "2"));
                debugPrintPrecision = Cpl::ToVal<int>(GetArg("-dpp", "4"));
                compareThreshold = Cpl::ToVal<float>(GetArg("-ct", "0.001"));
                compactWeight = Cpl::ToVal<bool>(GetArg("-cw", "1"));
            }

            ~Options()
            {
                std::stringstream ss;
                PrintPerformance(ss, 0.0);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
                ss << SimdPerformanceStatistic();
#endif
                //if (!consoleSilence)
                    std::cout << ss.str();
                if (!logName.empty())
                {
                    if (!CreateOutputDirectory(logName))
                        return;
                    std::ofstream log(logName.c_str());
                    if (log.is_open())
                    {
                        log << ss.str();
                        log.close();
                    }
                }
            }

            bool NeedOutputDirectory() const
            {
                return debugPrint;
            }

            size_t TestThreads() const
            {
                return std::max<size_t>(1, testThreads);
            }
        };

        //-----------------------------------------------------------------------------------------

        Stability(const Options &options)
            : _options(options)
        {
            _progressMessageSizeMax = 0;
            Synet::SetThreadNumber(1);
            _threads.resize(_options.TestThreads());
        }

        bool Run()
        {
            PrintStartMessage();
            if (!LoadTestParam())
                return false;
            if (!CreateDirectories())
                return false;
            if (!InitNetwork(_options.synetModel, _options.synetWeight, _threads[0]))
                return false;
            if (!CreateTestListImages(_threads[0], _options.imageDirectory))
                return false;
            if (_options.testThreads)
            {
                return false;
            }
            else
            {
                if (!SingleThread())
                    return false;
            }
            PrintFinishMessage();
            return true;
        }

    private:
        typedef std::vector<char> Bytes;
        typedef std::vector<Tensor*> TensorPtrs;

        struct Output
        {
            Tensors current, control;
        };
        typedef std::vector<Output> Outputs;

        struct Data
        {
            Strings path;
            Tensor input;
            Outputs output;
        };
        typedef std::shared_ptr<Data> DataPtr;
        typedef std::vector<DataPtr> DataPtrs;
        DataPtrs _datas;

        struct Thread
        {
            Network network;
            Floats lower, upper;
            String model, weight;
            std::thread thread;
            Tensors input, output;
            size_t current;

            Thread()
                : current(0)
            {
            }
        };

        typedef std::vector<Thread> Threads;

        const Options & _options;
        TestParamHolder _param;
        Floats _lower, _upper;
        Threads _threads;
        size_t _progressMessageSizeMax;

    protected:

        void PrintStartMessage() const
        {
            std::cout << "Start stability test in ";
            if (_options.testThreads > 0)
                std::cout << _options.testThreads << "-threads ";
            else
                std::cout << "single-thread ";
            std::cout << ":" << std::endl;
        }

        bool PrintFinishMessage() const
        {
            std::cout << Cpl::ExpandRight("Tests are finished successfully!", _progressMessageSizeMax) << std::endl << std::endl;
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

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
            {
                std::cout << "Can't create output directory '" << _options.outputDirectory << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool RequiredExtension(const String& name) const
        {
            String ext = Test::ExtensionByPath(name);
            static const char* EXTS[] = { "JPG", "jpg", "png", "ppm", "pgm", "bin" };
            for (size_t i = 0, n = sizeof(EXTS) / sizeof(EXTS[0]); i < n; ++i)
                if (ext == EXTS[i])
                    return true;
            return false;
        }

        bool LoadBinary(const String& path, String& data) const
        {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs)
                return false;
            ifs.unsetf(std::ios::skipws);
            ifs.seekg(0, std::ios::end);
            size_t size = ifs.tellg();
            ifs.seekg(0);
            data.resize(size + 1, 0);
            ifs.read(data.data(), (std::streamsize)size);
            ifs.close();
            return true;
        }

        bool CacheNetwork(const String& model, const String& weight, Thread& thread) const
        {
            if (thread.model.empty() || thread.weight.empty())
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
                if (!LoadBinary(model, thread.model))
                {
                    std::cout << "Can't cache model '" << model << "' file!" << std::endl;
                    return false;
                }
                if (!LoadBinary(weight, thread.weight))
                {
                    std::cout << "Can't cache weight '" << weight << "' file!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool InitNetwork(const String & model, const String & weight, Thread & thread) const
        {
            if (!CacheNetwork(model, weight, thread))
                return false;
            if (!thread.network.Load(thread.model.c_str(), thread.model.length(), thread.weight.c_str(), thread.weight.length()))
            {
                std::cout << "Can't load model '" << model << "' and weight '" << weight << weight << "' to network!" << std::endl;
                return false;
            }
            if (thread.network.Src().size() != 1)
            {
                std::cout << "Stability test support only 1 source!" << std::endl;
                return false;
            }                
            if (thread.network.Format() != Synet::TensorFormatNhwc || thread.network.Src()[0]->Count() != 4)
            {
                std::cout << "Stability test support only NHWC format!" << std::endl;
                return false;
            }
            if (thread.network.NchwShape()[0] != _options.batchSize)
            {
                if (!thread.network.SetBatch(_options.batchSize))
                    return false;
            }
            if(_options.compactWeight)
                thread.network.CompactWeight(); 
            Shape shape = thread.network.NchwShape();
            thread.lower = _param().lower();
            thread.upper = _param().upper();
            if (thread.lower.size() == 1)
                thread.lower.resize(shape[1], thread.lower[0]);
            if (thread.upper.size() == 1)
                thread.upper.resize(shape[1], thread.upper[0]);
            return true;
        }

        bool CreateTestListImages(const Thread& thread, const String& directory)
        {
            if (!DirectoryExists(directory))
            {
                std::cout << "Test image directory '" << directory << "' is not exists!" << std::endl;
                return false;
            }
            StringList images = GetFileList(directory, "*.*", true, false);
            images.sort();

            Strings names;
            names.reserve(images.size());
            size_t curr = 0;
            for (StringList::const_iterator it = images.begin(); it != images.end() && curr < _options.imageNumber; ++it)
            {
                if (RequiredExtension(*it))
                {
                    names.push_back(*it);
                    curr++;
                }
            }

            size_t num = names.size() / _options.batchSize;
            if (num == 0)
            {
                std::cout << "There is no one image in directory '" << directory << "'!" << std::endl;
                return false;
            }
            _datas.clear();
            _datas.reserve(num);
            for (size_t i = 0; i < num; ++i)
            {
                DataPtr data(new Data());
                data->path.resize(_options.batchSize);
                data->input.Reshape(thread.network.Src()[0]->Shape());
                data->output.resize(_options.TestThreads());
                float* input = data->input.CpuData();
                for (size_t b = 0; b < _options.batchSize; ++b)
                {
                    data->path[b] = MakePath(directory, names[i * _options.batchSize + b]);
                    View original;
                    if (!LoadImage(data->path[b], original))
                    {
                        std::cout << "Can't read '" << data->path[b] << "' image!" << std::endl;
                        return false;
                    }
                    Shape shape = thread.network.NchwShape();

                    View converted(original.Size(), shape[1] == 1 ? View::Gray8 : View::Bgr24);
                    Simd::Convert(original, converted);

                    View resized(Size(shape[3], shape[2]), converted.format);
                    Simd::Resize(converted, resized, SimdResizeMethodArea);

                    if (_param().order() == "rgb" && shape[1] == 3)
                        (View::Format&)resized.format = View::Rgb24;
                    Simd::SynetSetInput(resized, thread.lower.data(), thread.upper.data(), input, shape[1], SimdTensorFormatNhwc);
                    input += shape[1] * shape[2] * shape[3];
                }
                _datas.push_back(data);
            }
            return true;
        }

        bool SingleThread()
        {
            size_t repeats = std::max<size_t>(1, _options.repeatNumber), total = _datas.size() * repeats, current = 0;
            for (size_t i = 0; i < _datas.size(); ++i)
            {
                Data& data = *_datas[i];
                for (size_t r = 0; r < repeats; ++r, ++current)
                {
                    std::cout << ProgressString(current, total) << std::flush;
                    if (!(RunSingleTest(0, i, r)))
                        return false;
                    std::cout << " \r" << std::flush;
                }
            }
            return true;
        }

        bool RunSingleTest(size_t thread, size_t index, size_t repeat)
        {
            CPL_PERF_FUNC();

            Data& data = *_datas[index];
            if (!InitNetwork(_options.synetModel, _options.synetWeight, _threads[thread]))
                return false;
            Copy(data.input, *_threads[thread].network.Src()[0]);
            _threads[thread].network.Forward();
            Copy(_threads[thread].network.Dst(), data.output[thread].current);
            if (!Compare(data, index, thread))
                return false;
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

        inline void Sleep(unsigned int miliseconds)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
        }

        static inline void Copy(const TensorPtrs& src, Tensors& dst)
        {
            dst.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                Copy(*src[i], dst[i]);
        }

        static inline void Copy(const Tensors& src, Tensors& dst)
        {
            dst.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                Copy(src[i], dst[i]);
        }

        static inline void Copy(const Tensor& src, Tensor& dst)
        {
            if (src.Shape() != dst.Shape())
                dst.Reshape(src.Shape(), src.Format());
            memcpy(dst.CpuData(), src.CpuData(), src.RawSize());
        }

        bool Compare(float a, float b, float t) const
        {
            float d = ::fabs(a - b);
            return d <= t || d / std::max(::fabs(a), ::fabs(b)) <= t;
        }

        bool Compare(const Tensor& f, const Tensor& s, const Shape& i, size_t d, const String& m) const
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

        String TestFailedMessage(const Data& data, size_t index, size_t thread)
        {
            std::stringstream ss;
            ss << "At thread " << thread << " test " << index << " '" << data.path[0];
            for (size_t k = 1; k < data.path.size(); ++k)
                ss << ", " << data.path[k];
            ss << "' is failed!";
            return ss.str();
        }

        bool Compare(Data& data, size_t index, size_t thread)
        {
            using Synet::Detail::DebugPrint;
            Output& output = data.output[thread];
            if (output.control.empty())
            {
                Copy(output.current, output.control);
                return true;
            }
            String failed = TestFailedMessage(data, index, thread);
            if (output.control.size() != output.current.size())
            {
                std::cout << failed << std::endl;
                std::cout << "Dst count : " << output.control.size() << " != " << output.current.size() << std::endl;
                return false;
            }
            for (size_t d = 0; d < output.control.size(); ++d)
            {
                const Tensor& control = output.control[d];
                const Tensor& current = output.current[d];
                if (control.Shape() != current.Shape())
                {
                    std::cout << failed << std::endl;
                    std::cout << "Dst[" << d << "] shape : " << DebugPrint(control.Shape()) << " != " << DebugPrint(current.Shape()) << std::endl;
                    return false;
                }
                switch (control.Count())
                {
                case 1:
                    for (size_t n = 0; n < control.Axis(0); ++n)
                        if (!Compare(control, current, Shp(n), d, failed))
                            return false;
                    break;
                case 2:
                    for (size_t n = 0; n < control.Axis(0); ++n)
                        for (size_t c = 0; c < control.Axis(1); ++c)
                            if (!Compare(control, current, Shp(n, c), d, failed))
                                return false;
                    break;
                case 3:
                    for (size_t n = 0; n < control.Axis(0); ++n)
                        for (size_t c = 0; c < control.Axis(1); ++c)
                            for (size_t y = 0; y < control.Axis(2); ++y)
                                if (!Compare(control, current, Shp(n, c, y), d, failed))
                                    return false;
                    break;
                case 4:
                    for (size_t n = 0; n < control.Axis(0); ++n)
                        for (size_t c = 0; c < control.Axis(1); ++c)
                            for (size_t y = 0; y < control.Axis(2); ++y)
                                for (size_t x = 0; x < control.Axis(3); ++x)
                                    if (!Compare(control, current, Shp(n, c, y, x), d, failed))
                                        return false;
                    break;
                default:
                    std::cout << "Error! Dst has unsupported shape " << Synet::Detail::DebugPrint(control.Shape()) << std::endl;
                    return false;
                }
            }
            return true;
        }
    };
}

int main(int argc, char* argv[])
{
    Test::Stability::Options options(argc, argv);

    Test::Stability stability(options);

    options.result = stability.Run();

    return options.result ? 0 : 1;
}


