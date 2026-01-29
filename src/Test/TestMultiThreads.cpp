/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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
#include "TestCommon.h"

#include "Synet/Network.h"

#include "TestReport.h"
#include "TestOptions.h"
#include "TestParams.h"
#include "TestRegionDecoder.h"


namespace Test
{
    typedef Synet::Network Net;
    typedef Synet::Region<float> Region;
    typedef std::vector<Region> Regions;
    typedef Synet::Floats Floats;
    typedef Synet::Tensor<float> Tensor;
    typedef std::vector<Tensor> Tensors;
    typedef Synet::Index Index;

    //------------------------------------------------------------------------------------------------

    class MultiThreadsTest
    {
    public:
        MultiThreadsTest(const Options& options)
            : _options(options)
        {
        }

        bool Run()
        {
            PrintStartMessage();
            if (!LoadTestParam())
                return false;
            if (!CreateDirectories())
                return false;
            if (!InitSynet())
                return false;
            if (!CreateTestList())
                return false;
            return true;
        }

    private:
        const Options& _options;
        TestParamHolder _param;

        Net _net;
        bool _trans, _sort;
        Floats _lower, _upper;
        size_t _synetMemoryUsage;
        RegionDecoder _regionDecoder;

        typedef Tensors Output;
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

        void PrintStartMessage() const
        {
            std::cout << "Start ";
            //if (_options.enable & ENABLE_FIRST)
            //    std::cout << Options::FullName(_firsts[0].Name(), _firsts[0].Type()) << " ";
            //if (_options.enable == (ENABLE_FIRST | ENABLE_SECOND))
            //    std::cout << "and ";
            //if (_options.enable & ENABLE_SECOND)
            //    std::cout << Options::FullName(_seconds[0].Name(), _seconds[0].Type()) << " ";
            std::cout << _options.testThreads << "-threads ";
            //std::cout << (_options.enable == (ENABLE_FIRST | ENABLE_SECOND) ? "comparison " : "performance ");
            std::cout << "tests :" << std::endl;
        }

        bool LoadTestParam()
        {
            if (!_param.Load(_options.testParam))
                SYNET_ERROR("Can't load file '" << _options.testParam << "' !");
            return true;
        }

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
                SYNET_ERROR("Can't create output directory '" << _options.outputDirectory << "' !");
            return true;
        }

        bool InitSynet()
        {
            Synet::SetThreadNumber(_options.workThreads);
            if (!LoadModel())
                SYNET_ERROR("Can't load model from '" << _options.secondModel << "' and '" << _options.secondWeight << "' !");;
            _trans = _net.Format() == Synet::TensorFormatNhwc;
            _sort = _param().output().empty();
            if (_param().input().size() || _param().output().size())
            {
                if (!ReshapeModel())
                    return false;
            }
            else if (_net.Src().size() == 1)
            {
                const Shape& shape = _net.NchwShape();
                if (shape.size() == 4 && shape[0] != _options.batchSize)
                {
                    if (!_net.Reshape(shape[3], shape[2], _options.batchSize, _options.TestThreads()))
                        return false;
                }
            }
            _net.CompactWeight(!(_options.debugPrint & 2));
            _lower = _param().lower();
            _upper = _param().upper();
            _synetMemoryUsage = _net.MemoryUsage();
            _regionDecoder.Init(_net, _param());

            Shape shape = _net.NchwShape();
            if (!(_param().inputType() == "binary" || shape[1] == 1 || shape[1] == 3))
                SYNET_ERROR("Wrong model channels count '" << shape[1] << " !");

            return true;
        }

        bool LoadModel()
        {
            if (!Cpl::FileExists(_options.secondModel))
                SYNET_ERROR("File '" << _options.secondModel << "' is not exist!");
            if (!Cpl::FileExists(_options.secondWeight))
                SYNET_ERROR("File '" << _options.secondWeight << "' is not exist!");
            Synet::Options synOpt;
            synOpt.performanceLog = (Synet::Options::PerfomanceLog)_options.performanceLog;
            if (_options.bf16)
                synOpt.bf16Support = Synet::Options::Bf16SupportSoft;
            //synOpt.bf16Support = Synet::Options::Bf16SupportNone;
            return _net.Load(_options.secondModel, _options.secondWeight, synOpt, _options.TestThreads());
        }

        bool ReshapeModel()
        {
            Strings srcNames;
            Shapes srcShapes;
            for (size_t i = 0; i < _param().input().size(); ++i)
            {
                const InputParam& shape = _param().input()[i];
                srcNames.push_back(shape.name());
                Shape srcShape;
                if (shape.dims().size())
                    srcShape = shape.dims();
                else
                {
                    for (size_t j = 0; j < shape.shape().size(); ++j)
                    {
                        const SizeParam& size = shape.shape()[j];
                        if (size.size() > 0)
                            srcShape.push_back(size.size());
                        else
                            SYNET_ERROR("Test parameter input.shape.size must be > 0!");
                    }
                }
                if (srcShape.size() > 1)
                {
                    if (_options.batchSize != 1 && srcShape[0] == 1)
                        srcShape[0] = _options.batchSize;
                    if (_trans && srcShape.size() == 4)
                        srcShape = Shape({ srcShape[0], srcShape[2], srcShape[3], srcShape[1] });
                }
                if (srcShape.empty())
                    SYNET_ERROR("Test parameter input.shape is empty!");
                srcShapes.push_back(srcShape);
            }

            Strings dstNames;
            for (size_t i = 0; i < _param().output().size(); ++i)
                dstNames.push_back(_param().output()[i].name());

            bool equal = false;
            if (srcShapes.size() == _net.Src().size() && _net.Back().size() == dstNames.size())
            {
                equal = true;
                for (size_t i = 0; i < srcShapes.size() && equal; ++i)
                    if (srcShapes[i] != _net.Src()[i]->Shape())
                        equal = false;
                for (size_t i = 0; i < dstNames.size() && equal; ++i)
                    if (dstNames[i] != _net.Back()[i]->Param().name())
                        equal = false;
            }
            return equal || _net.Reshape(srcNames, srcShapes, dstNames, _options.TestThreads());
        }

        bool RequiredExtension(const String& name) const
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

        size_t RequiredInputNumber() const
        {
            if (_param().input().size())
            {
                size_t count = 0;
                for (size_t i = 0; i < _param().input().size(); ++i)
                    if (_param().input()[i].from().empty())
                        count++;
                return count;
            }
            return _net.Src().size();
        }

        bool CreateTestListImages(const String& directory)
        {
            StringList images = GetFileList(directory, _options.imageFilter, true, false);
            images.sort();

            Strings names;
            names.reserve(images.size());
            size_t curr = 0, rN = RequiredInputNumber(), imgBeg = _options.imageBegin * rN, imgEnd = _options.imageEnd * rN;
            for (StringList::const_iterator it = images.begin(); it != images.end(); ++it)
            {
                if (RequiredExtension(*it))
                {
                    if (curr >= imgBeg && curr < imgEnd)
                        names.push_back(*it);
                    curr++;
                }
            }
            size_t sN = _net.Src().size(), bN = _options.batchSize;
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
                    tensor.Reshape(_net.Src()[s]->GetType(), _net.Src()[s]->Shape(), Synet::TensorFormatUnknown);
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
                            Shape shape = _net.Src()[s]->Shape();
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

        bool CreateTestListBinary(const String& directory)
        {
            StringList files = GetFileList(directory, _options.imageFilter, true, false);
            files.sort();

            Strings names;
            names.reserve(files.size());
            for (StringList::const_iterator it = files.begin(); it != files.end(); ++it)
                if (RequiredExtension(*it))
                    names.push_back(*it);

            size_t sN = _net.Src().size(), bN = _options.batchSize;
            if (names.size() != sN)
                SYNET_ERROR("The number of binary files " << names.size() << " is differ from number of network sources " << sN << " in '" << directory << "' !");

            _tests.clear();
            for (size_t n = 0; n < sN; ++n)
            {
                size_t sS = _net.Src()[n]->Size();
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
                    tensor.Reshape(Synet::TensorType32f, _net.Src()[n]->Shape(), Synet::TensorFormatUnknown);
                    memcpy(tensor.Data<float>(), data.data() + offs, sS * sizeof(float));
                }
            }
            return true;
        }

        bool CreateTestList()
        {
            String imageDirectory = _options.imageDirectory;
            if (imageDirectory.empty())
                imageDirectory = Test::MakePath(DirectoryByPath(_options.testParam), _param().images());
            if (!DirectoryExists(imageDirectory))
                SYNET_ERROR("Test image directory '" << imageDirectory << "' is not exists!");
            if (_param().inputType() == "images")
                return CreateTestListImages(imageDirectory);
            else if (_param().inputType() == "binary")
                return CreateTestListBinary(imageDirectory);
            else
                SYNET_ERROR("Unknown input type '" << _param().inputType() << "' !");
        }
    };
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    Test::MultiThreadsTest multiThreadsTest(options);

    options.result = multiThreadsTest.Run();

    return options.result ? 0 : 1;
}