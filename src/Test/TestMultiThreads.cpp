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

#define SECOND_MODEL_DEFAULT "synet1.xml"
#define SECOND_WEIGHT_DEFAULT "synet1.bin"

#include "TestReport.h"
#include "TestOptions.h"
#include "TestParams.h"
#include "TestRegionDecoder.h"
#include "TestOutputComparer.h"

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
            , _progressMessageSizeMax(0)
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
            return ThreadsComparison();
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

        struct Thread
        {
            size_t current;
            bool first;
            std::thread thread;
            String debug;
            Thread() : current(0), first(false) {}
        };
        std::vector<Thread> _threads;
        std::condition_variable _start;
        bool _notified;
        size_t _progressMessageSizeMax;
        double _nextProgressUpdate;

        void PrintStartMessage() const
        {
            std::cout << "Start MultiThreadsTest for ";
            std::cout << _options.testThreads << "-threads: " << std::endl;
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

                                Simd::SynetSetInput(resized, lower.data(), upper.data(), input, _net.NchwShape()[1], 
                                    _options.tensorFormat ? SimdTensorFormatNhwc : SimdTensorFormatNchw, _param().order() == "rgb");
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

        template<class T> static void NetworkSetInput(const Tensor& src, Tensor& dst)
        {
            assert(src.Size() == dst.Size() && src.GetType() == dst.GetType());
            if (dst.Format() == Synet::TensorFormatNhwc && dst.Count() == 4  && 0)
            {
                for (size_t n = 0; n < src.Axis(0); ++n)
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t y = 0; y < src.Axis(2); ++y)
                            for (size_t x = 0; x < src.Axis(3); ++x)
                                dst.Data<T>(Shape({ n, y, x, c }))[0] = src.Data<T>(Shape({ n, c, y, x }))[0];
            }
            else
                memcpy(dst.RawData(), src.RawData(), src.RawSize());
        }

        void NetworkSetInput(const Tensors& src, size_t thread)
        {
            assert(src.size() == _net.Src().size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                switch (src[i].GetType())
                {
                case Synet::TensorType32f: NetworkSetInput<float>(src[i], *_net.Src(thread)[i]); break;
                case Synet::TensorType32i: NetworkSetInput<int32_t>(src[i], *_net.Src(thread)[i]); break;
                case Synet::TensorType64i: NetworkSetInput<int64_t>(src[i], *_net.Src(thread)[i]); break;
                default:
                    assert(0);
                }
            }
        }


        void NetworkSetOutput(Tensors& output, size_t thread)
        {
            if (_sort)
            {
                typedef std::map<String, Net::Tensor*> Dst;
                Dst dst;
                for (size_t i = 0; i < _net.Dst().size(); ++i)
                    dst[_net.Dst(thread)[i]->Name()] = _net.Dst(thread)[i];
                output.resize(dst.size());
                size_t i = 0;
                for (Dst::const_iterator it = dst.begin(); it != dst.end(); ++it, ++i)
                    NetworkSetOutput(*it->second, *_net.Back()[i], output[i]);
            }
            else
            {
                output.resize(_net.Dst().size());
                for (size_t i = 0; i < _net.Dst().size(); ++i)
                    NetworkSetOutput(*_net.Dst(thread)[i], *_net.Back()[i], output[i]);
            }
        }

        void NetworkSetOutput(const Net::Tensor& src, const Net::Layer& back, Tensor& dst)
        {
            switch (src.GetType())
            {
            case Synet::TensorType32f:
                NetworkSetOutputT<float>(src, back, dst);
                break;
            case Synet::TensorType32i:
                NetworkSetOutputT<int32_t>(src, back, dst);
                break;
            case Synet::TensorType64i:
                NetworkSetOutputT<int64_t>(src, back, dst);
                break;
            case Synet::TensorType8u:
                NetworkSetOutputT<uint8_t>(src, back, dst);
                break;
            default:
                assert(0);
            }
        }

        template<class T> void NetworkSetOutputT(const Tensor& src, const Net::Layer& back, Tensor& dst)
        {
            if (src.Count() == 4 && src.Axis(3) == 7 && back.Param().type() == Synet::LayerTypeDetectionOutput)
            {
                assert(src.Axis(0) == 1);
                Vector tmp;
                const T* pSrc = src.Data<T>();
                for (size_t j = 0; j < src.Axis(2); ++j, pSrc += 7)
                {
                    if (pSrc[0] == -1)
                        break;
                    if (pSrc[2] <= _options.regionThreshold)
                        continue;
                    size_t offset = tmp.size();
                    tmp.resize(offset + 7);
                    tmp[offset + 0] = (float)pSrc[0];
                    tmp[offset + 1] = (float)pSrc[1];
                    tmp[offset + 2] = (float)pSrc[2];
                    tmp[offset + 3] = (float)pSrc[3];
                    tmp[offset + 4] = (float)pSrc[4];
                    tmp[offset + 5] = (float)pSrc[5];
                    tmp[offset + 6] = (float)pSrc[6];
                }
                SortDetectionOutput(tmp.data(), tmp.size());
                dst.Reshape(Synet::TensorType32f, Shp(1, 1, tmp.size() / 7, 7));
                memcpy(dst.RawData(), tmp.data(), dst.RawSize());
            }
            else if (_param().detection().decoder() == "rtdetr")
            {
                assert(src.Axis(0) == 1 && src.Axis(2) == 6);
                Vector tmp;
                const T* pSrc = src.Data<T>();
                for (size_t i = 0, n = src.Axis(1); i < n; ++i, pSrc += 6)
                {
                    if (pSrc[4] <= _options.regionThreshold)
                        continue;
                    size_t offset = tmp.size();
                    tmp.resize(offset + 6);
                    tmp[offset + 0] = (float)pSrc[0];
                    tmp[offset + 1] = (float)pSrc[1];
                    tmp[offset + 2] = (float)pSrc[2];
                    tmp[offset + 3] = (float)pSrc[3];
                    tmp[offset + 4] = (float)pSrc[4];
                    tmp[offset + 5] = (float)pSrc[5];
                }
                SortRtdetr(tmp.data(), tmp.size());
                dst.Reshape(Synet::TensorType32f, Shp(1, tmp.size() / 6, 6));
                memcpy(dst.RawData(), tmp.data(), dst.RawSize());
            }
            else
            {
                bool trans = src.Format() == Synet::TensorFormatNhwc;
                bool batch = _net.Src()[0]->Axis(0) != 1;
                if (trans && src.Count() == 4)
                {
                    dst.Reshape(Synet::TensorType32f, Shp(src.Axis(0), src.Axis(3), src.Axis(1), src.Axis(2)), Synet::TensorFormatNchw);
                    for (size_t n = 0; n < src.Axis(0); ++n)
                        for (size_t c = 0; c < src.Axis(3); ++c)
                            for (size_t y = 0; y < src.Axis(1); ++y)
                                for (size_t x = 0; x < src.Axis(2); ++x)
                                    dst.Data<float>(Shp(n, c, y, x))[0] = (float)src.Data<T>(Shp(n, y, x, c))[0];
                }
                else if (trans && src.Count() == 3)
                {
                    if (batch)
                    {
                        dst.Reshape(Synet::TensorType32f, Shp(src.Axis(0), src.Axis(2), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t n = 0; n < src.Axis(0); ++n)
                            for (size_t c = 0; c < src.Axis(2); ++c)
                                for (size_t s = 0; s < src.Axis(1); ++s)
                                    dst.Data<float>(Shp(n, c, s))[0] = (float)src.Data<T>(Shp(n, s, c))[0];
                    }
                    else
                    {
                        dst.Reshape(Synet::TensorType32f, Shp(src.Axis(2), src.Axis(0), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t c = 0; c < src.Axis(2); ++c)
                            for (size_t y = 0; y < src.Axis(0); ++y)
                                for (size_t x = 0; x < src.Axis(1); ++x)
                                    dst.Data<float>(Shp(c, y, x))[0] = (float)src.Data<T>(Shp(y, x, c))[0];
                    }
                }
                else if (trans && src.Count() == 2 && (src.Axis(0) == 1 || src.Format() == Synet::TensorFormatNhwc))
                {
                    dst.Reshape(Synet::TensorType32f, Shp(src.Axis(1), src.Axis(0)), Synet::TensorFormatNchw);
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t s = 0; s < src.Axis(0); ++s)
                            dst.Data<float>(Shp(c, s))[0] = (float)src.Data<T>(Shp(s, c))[0];
                }
                else
                {
                    dst.Reshape(Synet::TensorType32f, src.Shape(), Synet::TensorFormatNchw);
                    for (size_t i = 0; i < src.Size(); ++i)
                        dst.Data<float>()[i] = (float)src.Data<T>()[i];
                }
            }
            dst.SetName(src.Name());
        }

        void NetworkPredict(const Tensors& src, size_t thread, Tensors& dst)
        {
            NetworkSetInput(src, thread);
            {
                CPL_PERF_BEGF("Network::Forward", _net.Flop());
                _net.Forward(thread);
            }
            NetworkSetOutput(dst, thread);
        }

        void ThreadRun(size_t thread, size_t total, size_t& current)
        {
            if (_options.repeatNumber)
            {
                for (size_t i = 0; i < _tests.size(); ++i)
                {
                    TestData& test = *_tests[i];
                    for (size_t r = 0; r < _options.repeatNumber; ++r, ++current)
                    {
                        NetworkPredict(test.input, thread, test.output[thread]);
                        _threads[thread].current = current;
                        //_threads[thread].debug = CoreFreqInfo();
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
                        NetworkPredict(test.input, thread, test.output[thread]);
                        duration = Cpl::Time() - start;
                        _threads[thread].current = std::min(total, size_t(duration * 1000));
                        //_threads[thread].debug = CoreFreqInfo();
                    }
                    canstop = true;
                }
            }
        }

        static void TestThread(MultiThreadsTest* multiThreadsTest, size_t thread, size_t total)
        {
            const Options& options = multiThreadsTest->_options;
            if (options.pinThread)
                PinThread(thread);

            size_t current = 0;

            multiThreadsTest->ThreadRun(thread, total, current);

            multiThreadsTest->_threads[thread].current = total;
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
            _progressMessageSizeMax = std::max(_progressMessageSizeMax, progress.str().size());
            return progress.str();
        }

        String TestFailedMessage(const TestData& test, size_t index, size_t thread) const
        {
            std::stringstream ss;
            ss << "At thread " << thread << " test " << index << " '" << test.path[0];
            for (size_t k = 1; k < test.path.size(); ++k)
                ss << ", " << test.path[k];
            ss << "' is failed!";
            return ss.str();
        }

        bool CompareResults(const TestData& test, size_t index, size_t thread) const
        {
            const Output& control = test.output[0];
            const Output& current = test.output[thread];
            String failed = TestFailedMessage(test, index, thread);
            OutputComparer outputComparer(_options, _param(), test.input[0].Shape(), control);
            return outputComparer.Compare(control, current, failed);
        }

        bool ThreadsComparison()
        {
            if (_options.pinThread)
                PinThread(SimdCpuInfo(SimdCpuInfoThreads) - 1);
            int64_t start = Cpl::TimeCounter();
            size_t current = 0, total = _options.repeatNumber ?
                _tests.size() * _options.repeatNumber : size_t(_options.executionTime * 1000);
            _threads.resize(_options.TestThreads());
            for (size_t t = 0; t < _threads.size(); ++t)
                _threads[t].thread = std::thread(TestThread, this, t, total);

            while (current < total)
            {
                current = total;
                for (size_t t = 0; t < _threads.size(); ++t)
                    current = std::min(current, _threads[t].current);
                std::cout << ProgressString(current, total) << std::flush;
                Sleep(1);
                std::cout << " \r" << std::flush;
            }

            _options.secondMemoryUsage = _net.MemoryUsage();
            for (size_t t = 0; t < _threads.size(); ++t)
            {
                if (_threads[t].thread.joinable())
                    _threads[t].thread.join();
            }
            for (size_t t = 1; t < _threads.size(); ++t)
            {
                for (size_t i = 0; i < _tests.size(); ++i)
                {
                    if (!CompareResults(*_tests[i], i, t))
                        return false;
                }
            }
            return PrintFinishMessage(start);
        }

        inline void Sleep(unsigned int miliseconds)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
        }

        //static inline void Copy(const Tensors& src, Tensors& dst)
        //{
        //    dst.resize(src.size());
        //    for (size_t i = 0; i < src.size(); ++i)
        //        dst[i].Clone(src[i]);
        //}

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

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    Test::MultiThreadsTest multiThreadsTest(options);

    options.result = multiThreadsTest.Run();

    return options.result ? 0 : 1;
}