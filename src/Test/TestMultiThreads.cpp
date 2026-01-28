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

            //Shape shape = _net.SrcShape(0);
            //if (!(shape[1] == 1 || shape[1] == 3 || _param().inputType() == "binary"))
            //    SYNET_ERROR("Wrong " << network.Name() << " network model channels count '" << shape[1] << " !");
            //return true;

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