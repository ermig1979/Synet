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

#include "Test/TestCompare.h"

#include "Synet/Converters/InferenceEngine.h"

#ifdef SYNET_OTHER_RUN

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <ie_blob.h>
#include <ie_plugin_dispatcher.hpp>
#include <ext_list.hpp>
#include <inference_engine.hpp>

namespace Test
{
    struct InferenceEngineNetwork : public Network
    {
        InferenceEngineNetwork()
        {
        }

        virtual ~InferenceEngineNetwork()
        {
        }

        virtual String Name() const
        {
            return "Inference Engine";
        }

        virtual size_t SrcCount() const
        {
            return _ieInput.size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _ieInput[index]->getTensorDesc().getDims();
            if (_batchSize > 1)
            {
                assert(shape.size() == 4);
                shape[0] = _batchSize;
            }
            return shape;
        }

        virtual size_t SrcSize(size_t index) const
        {
            Shape shape = SrcShape(index);
            size_t size = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                size *= shape[i];
            return size;
        }

        virtual bool Init(const String & model, const String & weight, size_t threadNumber, size_t batchSize, const TestParam & param)
        {
            TEST_PERF_FUNC();

            try
            {
                _iePlugin = InferenceEngine::PluginDispatcher({ "" }).getPluginByDevice("CPU");
                _iePlugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());

                InferenceEngine::CNNNetReader reader;
                reader.ReadNetwork(model);
                reader.ReadWeights(weight);
                InferenceEngine::CNNNetwork network = reader.getNetwork();

                std::map<std::string, std::string> config;
                config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(threadNumber);
                _batchSize = 1;
                if (batchSize > 1)
                {
                    try
                    {
                        network.setBatchSize(batchSize);
                        config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
                        InferenceEngine::ExecutableNetwork executableNet = _iePlugin.LoadNetwork(network, config);
                        _ieInferRequest = executableNet.CreateInferRequest();
                        _ieInferRequest.SetBatch(batchSize);
                    }
                    catch (std::exception & e)
                    {
                        std::cout << "Inference Engine init trouble: '" << e.what() << "', try to emulate batch > 1." << std::endl;
                        _batchSize = batchSize;
                        network.setBatchSize(1);
                        config.erase(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
                        InferenceEngine::ExecutableNetwork executableNet = _iePlugin.LoadNetwork(network, config);
                        _ieInferRequest = executableNet.CreateInferRequest();
                    }
                }
                else
                {
                    InferenceEngine::ExecutableNetwork executableNet = _iePlugin.LoadNetwork(network, config);
                    _ieInferRequest = executableNet.CreateInferRequest();
                }

                _ieInput.clear();
                InferenceEngine::InputsDataMap inputsInfo = network.getInputsInfo();
                for (InferenceEngine::InputsDataMap::iterator it = inputsInfo.begin(); it != inputsInfo.end(); ++it)
                {
                    auto inputName = it->first;
                    it->second->setPrecision(InferenceEngine::Precision::FP32);
                    _ieInput.push_back(_ieInferRequest.GetBlob(inputName));
                }

                _outputNames.clear();
                _ieOutput.clear();
                if (param.output().size())
                {
                    for (size_t i = 0; i < param.output().size(); ++i)
                    {
                        _outputNames.push_back(param.output()[i].name());
                        _ieOutput.push_back(_ieInferRequest.GetBlob(_outputNames[i]));
                    }
                }
                else
                {
                    InferenceEngine::OutputsDataMap outputsInfo = network.getOutputsInfo();
                    for (InferenceEngine::OutputsDataMap::iterator it = outputsInfo.begin(); it != outputsInfo.end(); ++it)
                    {
                        _outputNames.push_back(it->first);
                        _ieOutput.push_back(_ieInferRequest.GetBlob(it->first));
                    }
                }
            }
            catch (std::exception & e)
            {
                std::cout << "Inference Engine init error: " << e.what() << std::endl;
                return false;
            }

            {
                Vectors stub(SrcCount());
                for (size_t i = 0; i < SrcCount(); ++i)
                    stub[i].resize(SrcSize(i));
                SetInput(stub, 0);
                _ieInferRequest.Infer();
            }

            return true;
        }

        virtual const Vectors & Predict(const Vectors & src)
        {
            if (_batchSize == 1)
            {
                SetInput(src, 0);
                {
                    TEST_PERF_FUNC();
                    _ieInferRequest.Infer();
                }
                SetOutput(0);
            }
            else
            {
                TEST_PERF_BLOCK("batch emulation");
                for (size_t b = 0; b < _batchSize; ++b)
                {
                    SetInput(src, b);
                    _ieInferRequest.Infer();
                    SetOutput(b);
                }
            }
            return _output;
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        virtual void DebugPrint(std::ostream & os, int flag, int first, int last)
        {
            for (size_t o = 0; o < _ieOutput.size(); ++o)
            {
                InferenceEngine::SizeVector dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                dims[0] = _batchSize;
                Synet::Tensor<float> tensor(dims);
                SetOutput(dims, strides, 0, _output[o].data(), tensor.CpuData());
                tensor.DebugPrint(os, _outputNames.empty() ? String("???") : String(_outputNames[o]), false, first, last);
            }
        }
#endif

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            Regions regions;
            for (size_t i = 0; i < _output[0].size(); i += 7)
            {
                if (_output[0][i + 2] > threshold)
                {
                    Region region;
                    region.id = (size_t)_output[0][i + 1];
                    region.prob = _output[0][i + 2];
                    region.x = size.x*(_output[0][i + 3] + _output[0][i + 5]) / 2.0f;
                    region.y = size.y*(_output[0][i + 4] + _output[0][i + 6]) / 2.0f;
                    region.w = size.x*(_output[0][i + 5] - _output[0][i + 3]);
                    region.h = size.y*(_output[0][i + 6] - _output[0][i + 4]);
                    regions.push_back(region);
                }
            }
            return regions;
        }

    private:
        typedef InferenceEngine::SizeVector Sizes;

        InferenceEngine::InferencePlugin _iePlugin;
        InferenceEngine::InferRequest _ieInferRequest;
        std::vector<InferenceEngine::Blob::Ptr> _ieInput;
        std::vector<InferenceEngine::Blob::Ptr> _ieOutput;
        Vectors _output;
        Strings _outputNames;
        size_t _batchSize;

        void SetInput(const Vectors & x, size_t b)
        {
            assert(_ieInput.size() == x.size() && _ieInput[0]->getTensorDesc().getLayout() == InferenceEngine::Layout::NCHW);
            for (size_t i = 0; i < x.size(); ++i)
            {
                const InferenceEngine::SizeVector & dims = _ieInput[i]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieInput[i]->getTensorDesc().getBlockingDesc().getStrides();
                const float * src = x[i].data() + b * x[i].size() / _batchSize;
                float * dst = (float*)_ieInput[i]->buffer();
                SetInput(dims, strides, 0, src, dst);
            }
        }

        void SetInput(const Sizes & dims, const Sizes & strides, size_t current, const float * src, float * dst)
        {
            if (current == dims.size() - 1)
            {
                memcpy(dst, src, dims[current] * sizeof(float));
            }
            else
            {
                size_t srcStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    srcStride *= dims[i];
                size_t dstStride = strides[current];
                for (size_t i = 0; i < dims[current]; ++i)
                    SetInput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }

        void SetOutput(size_t b)
        {
            _output.resize(_ieOutput.size());
            for (size_t o = 0; o < _ieOutput.size(); ++o)
            {
                const InferenceEngine::SizeVector & dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                if (dims.size() == 4 && dims[3] == 7)
                {
                    const float * pOut = _ieOutput[o]->buffer();
                    if (b == 0)
                        _output[o].clear();
                    for (size_t j = 0; j < dims[2]; ++j, pOut += 7)
                    {
                        if (pOut[0] == -1 || pOut[2] <= 0.3f)
                            break;
                        size_t size = _output[o].size();
                        _output[o].resize(size + 7);
                        _output[o][size + 0] = pOut[0];
                        _output[o][size + 1] = pOut[1];
                        _output[o][size + 2] = pOut[2];
                        _output[o][size + 3] = pOut[3];
                        _output[o][size + 4] = pOut[4];
                        _output[o][size + 5] = pOut[5];
                        _output[o][size + 6] = pOut[6];
                    }
                    SortDetectionOutput(_output[o].data(), _output[o].size());
                }
                else
                {
                    size_t size = 1;
                    for (size_t i = 0; i < dims.size(); ++i)
                        size *= dims[i];
                    _output[o].resize(size*_batchSize);
                    SetOutput(dims, strides, 0, _ieOutput[o]->buffer(), _output[o].data() + b * size);
                }
            }
        }

        void SetOutput(const Sizes & dims, const Sizes & strides, size_t current, const float * src, float * dst)
        {
            if (current == dims.size() - 1)
            {
                memcpy(dst, src, dims[current] * sizeof(float));
            }
            else
            {
                size_t srcStride = strides[current];
                size_t dstStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    dstStride *= dims[i];
                for(size_t i = 0; i < dims[current]; ++i)
                    SetOutput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }
    };
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#else //SYNET_OTHER_RUN
namespace Test
{
    struct InferenceEngineNetwork : public Network
    {
    };
}
#endif//SYNET_OTHER_RUN

Test::PerformanceMeasurerStorage Test::PerformanceMeasurerStorage::s_storage;

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    if (options.mode == "convert")
    {
        SYNET_PERF_FUNC();
        std::cout << "Convert network from Inference Engine to Synet :" << std::endl;
        options.result = Synet::ConvertInferenceEngineToSynet(options.otherModel, options.otherWeight, options.tensorFormat == 1, options.synetModel, options.synetWeight);
        std::cout << "Conversion is finished " << (options.result ? "successfully." : "with errors.") << std::endl;
    }
    else if (options.mode == "compare")
        options.result = Test::CompareOtherAndSynet<Test::InferenceEngineNetwork>(options);
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}