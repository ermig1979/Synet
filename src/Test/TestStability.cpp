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
            String outputDirectory;
            size_t testThreads;
            size_t repeatNumber;
            int batchSize;
            int debugPrint;
            int debugPrintFirst;
            int debugPrintLast;
            int debugPrintPrecision;

            Options(int argc, char* argv[])
                : ArgsParser(argc, argv, true)
                , result(true)
            {
                synetModel = GetArg("-sm", "synet.xml");
                synetWeight = GetArg("-sw", "synet.bin");
                testParam = GetArg("-tp", "param.xml");
                imageDirectory = GetArg("-id", "image");
                outputDirectory = GetArg("-od", "output");
                testThreads = Cpl::ToVal<size_t>(GetArg("-tt", "0"));
                repeatNumber = std::max(0, Cpl::ToVal<int>(GetArg("-rn", "1")));
                batchSize = Cpl::ToVal<int>(GetArg("-bs", "1"));
                debugPrint = FromString<int>(GetArg("-dp", "0"));
                debugPrintFirst = FromString<int>(GetArg("-dpf", "5"));
                debugPrintLast = FromString<int>(GetArg("-dpl", "2"));
                debugPrintPrecision = FromString<int>(GetArg("-dpp", "4"));
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

        Stability(const Options options)
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
        struct Data
        {
            Strings path;
            Tensor input;
            Tensors output, control;
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

        Options _options;
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

        bool InitNetwork(const String & model, const String & weight, Thread & thread) const
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
            if (!thread.network.Load(model, weight))
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
            for (StringList::const_iterator it = images.begin(); it != images.end(); ++it)
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
            for (size_t t = 0; t < num; ++t)
            {
                DataPtr data(new Data());
                data->path.resize(_options.batchSize);
                data->output.resize(_options.TestThreads());
                data->input.Reshape(thread.network.Src()[0]->Shape());
                float* input = data->input.CpuData();
                for (size_t b = 0; b < _options.batchSize; ++b)
                {
                    data->path[b] = MakePath(directory, names[t * _options.batchSize + b]);
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
                    if (!(SingleThread(i, r)))
                        return false;
                    Sleep(100);
                    //if (!CompareResults(test, i, 0))
                    //    return false;
                    std::cout << " \r" << std::flush;
                }
            }
            return true;
        }

        bool SingleThread(size_t index, size_t repeat)
        {
            Data& data = *_datas[index];
            _threads[0].network.Forward();

            //Copy(_threads[0].Predict(data.input), data.output[0].second);
            //if (repeat == 0)
            //{
            //    if (!DebugPrint(_seconds[0], index))
            //        return false;
            //}
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
    };
}

int main(int argc, char* argv[])
{
    Test::Stability::Options options(argc, argv);

    Test::Stability stability(options);

    options.result = stability.Run();

    return options.result ? 0 : 1;
}


