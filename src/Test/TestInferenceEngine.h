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

#include "TestCommon.h"
#include "TestPerformance.h"
#include "TestNetwork.h"

#define SYNET_TEST_IE_VERSION 202201

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

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
                assert(shape.size() >= 2);
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

        virtual bool Init(const String& model, const String& weight, const Options& options, const TestParam& param)
        {
            CPL_PERF_FUNC();
            _regionThreshold = options.regionThreshold;
            _decoderName = param.detection().decoder();

            ::setenv("OMP_NUM_THREADS", std::to_string(options.workThreads).c_str(), 1);
            ::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
            try
            {
                if (!ReadNetwork(model, weight, param.model() == "onnx"))
                    return false;

                if (!InitInput(param))
                    return false;

                if (!InitOutput(param))
                    return false;

                AddInterimOutput(options);

                StringMap config;
                config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = std::to_string(options.workThreads);
                config[InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD] = InferenceEngine::PluginConfigParams::NO;
                _batchSize = 1;
                if (options.batchSize > 1)
                {
                    try
                    {
                        _ieNetwork->setBatchSize(options.batchSize);
                        config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT] = std::to_string(options.batchSize);
                        config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
                        CreateExecutableNetworkAndInferRequest(config);
                        _ieInferRequest->SetBatch(options.batchSize);
                        GetBlobs();
                        StubInfer();
                    }
                    catch (std::exception& e)
                    {
                        if (!options.consoleSilence)
                            CPL_LOG_SS(Warning, "Inference Engine init trouble: '" << e.what() << "', try to emulate batch > 1.");
                        _batchSize = options.batchSize;
                        _ieNetwork->setBatchSize(1);
                        config.erase(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT);
                        config.erase(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED);
                        CreateExecutableNetworkAndInferRequest(config);
                    }
                }
                else
                    CreateExecutableNetworkAndInferRequest(config);
                GetBlobs();
                StubInfer();
            }
            catch (std::exception& e)
            {
                SYNET_ERROR("Inference Engine init error: " << e.what());
            }

            if (options.debugPrint & (1 << Synet::DebugPrintLayerDst))
                _ieExecutableNetwork->GetExecGraphInfo().serialize(MakePath(options.outputDirectory, "ie_exec_outs.xml"));

            return true;
        }

        virtual const Tensors& Predict(const Tensors& src)
        {
            if (_batchSize == 1)
            {
                SetInput(src, 0);
                {
                    CPL_PERF_FUNC();
                    _ieInferRequest->Infer();
                }
                SetOutput(0);
            }
            else
            {
                CPL_PERF_BEG("batch emulation");
                for (size_t b = 0; b < _batchSize; ++b)
                {
                    SetInput(src, b);
                    _ieInferRequest->Infer();
                    SetOutput(b);
                }
            }
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream& os, int flag, int first, int last, int precision)
        {
            for (size_t i = 0; i < _ieInterim.size(); ++i)
                DebugPrint(os, *_ieInterim[i], _interimNames[i], flag, first, last, precision);

            for (size_t o = 0; o < _ieOutput.size(); ++o)
                DebugPrint(os, *_ieOutput[o], _outputNames[o], flag, first, last, precision);
        }

        virtual Regions GetRegions(const Size& size, float threshold, float overlap) const
        {
            Regions regions;
            if (_output[0].Axis(-1) == 7)
            {
                for (size_t i = 0; i < _output[0].Size(); i += 7)
                {
                    const float* output = _output[0].Data<float>();
                    if (output[i + 2] > threshold)
                    {
                        Region region;
                        region.id = (size_t)output[i + 1];
                        region.prob = output[i + 2];
                        region.x = size.x * (output[i + 3] + output[i + 5]) / 2.0f;
                        region.y = size.y * (output[i + 4] + output[i + 6]) / 2.0f;
                        region.w = size.x * (output[i + 5] - output[i + 3]);
                        region.h = size.y * (output[i + 6] - output[i + 4]);
                        regions.push_back(region);
                    }
                }
            }
            return regions;
        }

        virtual void Free()
        {
            Network::Free();
            _ieInput.clear();
            _ieOutput.clear();
            _ieInterim.clear();
            _ieInferRequest.reset();
            _ieExecutableNetwork.reset();
            _ieExecutableNetwork.reset();
            _ieNetwork.reset();
            _ieCore.reset();
            _inputNames.clear();
            _outputNames.clear();
            _interimNames.clear();
            _batchSize = 0;
#ifdef __linux__
            malloc_trim(0);
#endif
        }

    private:
        typedef InferenceEngine::SizeVector Sizes;
        typedef std::map<std::string, std::string> StringMap;

        const std::string _ieDeviceName = "CPU";
        std::shared_ptr<InferenceEngine::Core> _ieCore;
        std::shared_ptr<InferenceEngine::CNNNetwork> _ieNetwork;
        std::shared_ptr<InferenceEngine::ExecutableNetwork> _ieExecutableNetwork;
        std::shared_ptr<InferenceEngine::InferRequest> _ieInferRequest;
        std::vector<InferenceEngine::Blob::Ptr> _ieInput, _ieOutput, _ieInterim;
        Strings _inputNames, _outputNames, _interimNames;
        size_t _batchSize;

        bool ReadNetwork(const String & model, const String& weight, bool onnx)
        {
            String src, dst, bin;
            if (onnx)
            {
                src = weight;
                dst = WithoutExtension(src) + ".onnx";
            }
            else
            {
                src = model;
                dst = WithoutExtension(src) + ".xml";
                bin = weight;
            }
            if (!FileExists(dst))
            {
                if (!FileCopy(src, dst))
                    SYNET_ERROR("Can't copy file form '" << src << "' to '" << dst << "' !");
            }            
            _ieCore = std::make_shared<InferenceEngine::Core>();
            _ieNetwork = std::make_shared<InferenceEngine::CNNNetwork>(_ieCore->ReadNetwork(dst, bin));
            return true;
        }

        bool InitInput(const TestParam& param)
        {
            _inputNames.clear();
            if (param.input().size())
            {
                typedef InferenceEngine::ICNNNetwork::InputShapes InputShapes;
                InputShapes shapes = _ieNetwork->getInputShapes();
                if (shapes.size() != param.input().size())
                {
                    std::cout << "Incorrect input count :" << param.input().size() << std::endl;
                    return false;
                }
                for (size_t i = 0; i < param.input().size(); ++i)
                {
                    const String& name = param.input()[i].name();
                    _inputNames.push_back(name);
                    if (shapes.find(name) == shapes.end())
                        SYNET_ERROR("Input with name '" << name << "' is not exist! ");
                    Shape & shape = shapes[name];
                    shape.clear();
                    for (size_t j = 0; j < param.input()[i].shape().size(); ++j)
                        shape.push_back(param.input()[i].shape()[j].size());
                }
                _ieNetwork->reshape(shapes);
            }
            else
            {
                InferenceEngine::InputsDataMap inputsInfo = _ieNetwork->getInputsInfo();
                for (InferenceEngine::InputsDataMap::iterator it = inputsInfo.begin(); it != inputsInfo.end(); ++it)
                    _inputNames.push_back(it->first);
            }
            return true;
        }

        bool InitOutput(const TestParam& param)
        {
            _outputNames.clear();
            if (param.output().size())
            {
                for (size_t i = 0; i < param.output().size(); ++i)
                    _outputNames.push_back(param.output()[i].name());
            }
            else
            {
                InferenceEngine::OutputsDataMap outputsInfo = _ieNetwork->getOutputsInfo();
                for (InferenceEngine::OutputsDataMap::iterator it = outputsInfo.begin(); it != outputsInfo.end(); ++it)
                    _outputNames.push_back(it->first);
            }
            return true;
        }

        void CreateExecutableNetworkAndInferRequest(const StringMap& config)
        {
            _ieExecutableNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(
                _ieCore->LoadNetwork(*_ieNetwork, _ieDeviceName, config));
            _ieInferRequest = _ieExecutableNetwork->CreateInferRequestPtr();
        }

        void GetBlobs()
        {
            _ieInput.clear();
            for (size_t i = 0; i < _inputNames.size(); ++i)
                _ieInput.push_back(_ieInferRequest->GetBlob(_inputNames[i]));
            _ieOutput.clear();
            for (size_t i = 0; i < _outputNames.size(); ++i)
                _ieOutput.push_back(_ieInferRequest->GetBlob(_outputNames[i]));
            _ieInterim.clear();
            for (size_t i = 0; i < _interimNames.size(); ++i)
                _ieInterim.push_back(_ieInferRequest->GetBlob(_interimNames[i]));
        }

        void StubInfer()
        {
            Tensors stub(SrcCount());
            for (size_t i = 0; i < SrcCount(); ++i)
                stub[i].Reshape(SrcShape(i));
            SetInput(stub, 0);
            _ieInferRequest->Infer();
        }

        void SetInput(const Tensors& x, size_t b)
        {
            assert(_ieInput.size() == x.size() && _ieInput[0]->getTensorDesc().getLayout() == InferenceEngine::Layout::NCHW);
            for (size_t i = 0; i < x.size(); ++i)
            {
                const InferenceEngine::SizeVector& dims = _ieInput[i]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector& strides = _ieInput[i]->getTensorDesc().getBlockingDesc().getStrides();
                const float* src = x[i].Data<float>() + b * x[i].Size() / _batchSize;
                float* dst = (float*)_ieInput[i]->buffer();
                SetInput(dims, strides, 0, src, dst);
            }
        }

        void SetInput(const Sizes& dims, const Sizes& strides, size_t current, const float* src, float* dst)
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
                const InferenceEngine::SizeVector& dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector& strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                const InferenceEngine::Precision& precision = _ieOutput[o]->getTensorDesc().getPrecision();
                if (dims.size() == 4 && dims[3] == 7 && _outputNames[o].find("Yolo") == std::string::npos)
                {
                    assert(dims[0] == 1);
                    Vector tmp;
                    const float* pOut = _ieOutput[o]->buffer();
                    for (size_t j = 0; j < dims[2]; ++j, pOut += 7)
                    {
                        if (pOut[0] == -1)
                            break;
                        if (pOut[2] <= _regionThreshold)
                            continue;
                        size_t size = tmp.size();
                        tmp.resize(size + 7);
                        tmp[size + 0] = pOut[0];
                        tmp[size + 1] = pOut[1];
                        tmp[size + 2] = pOut[2];
                        tmp[size + 3] = pOut[3];
                        tmp[size + 4] = pOut[4];
                        tmp[size + 5] = pOut[5];
                        tmp[size + 6] = pOut[6];
                    }
                    SortDetectionOutput(tmp.data(), tmp.size());
                    _output[o].Reshape(Shp(1, 1, tmp.size() / 7, 7), Synet::TensorFormatNchw);
                    memcpy(_output[o].Data<float>(), tmp.data(), _output[o].Size() * sizeof(float));
                }
                else
                {
                    if (b == 0)
                    {
                        Shape shape = dims;
                        if (_batchSize != 1)
                        {
                            if (shape[0] == 1)
                                shape[0] = _batchSize;
                            else
                                shape.insert(shape.begin(), _batchSize);
                        }
                        _output[o].Reshape(shape, Synet::TensorFormatNchw);
                    }
                    size_t size = 1;
                    for (size_t i = 0; i < dims.size(); ++i)
                        size *= dims[i];
                    switch (precision)
                    {
                    case InferenceEngine::Precision::FP32:
                        SetOutput(dims, strides, 0, (const float*)_ieOutput[o]->buffer(), _output[o].Data<float>() + b * size);
                        break;
                    case InferenceEngine::Precision::I32:
                        SetOutput(dims, strides, 0, (const int32_t*)_ieOutput[o]->buffer(), _output[o].Data<float>() + b * size);
                        break;
                    case InferenceEngine::Precision::I64:
                        SetOutput(dims, strides, 0, (const int64_t*)_ieOutput[o]->buffer(), _output[o].Data<float>() + b * size);
                        break;
                    default:
                        assert(0);
                    }
                }
            }
        }

        template<class S, class D> void SetOutput(const Sizes& dims, const Sizes& strides, size_t current, const S* src, D* dst)
        {
            if (current == dims.size() - 1)
            {
                for (size_t i = 0; i < dims[current]; ++i)
                    dst[i] = (D)src[i];
            }
            else
            {
                size_t srcStride = strides[current];
                size_t dstStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    dstStride *= dims[i];
                for (size_t i = 0; i < dims[current]; ++i)
                    SetOutput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }

        void AddInterimOutput(const Options& options)
        {
            _interimNames.clear();
            if ((options.debugPrint & (1 << Synet::DebugPrintLayerDst)) == 0)
                return;
#if SYNET_TEST_IE_VERSION < 202101
            StringMap config, interim;
            InferenceEngine::ExecutableNetwork exec = _ieCore->LoadNetwork(*_ieNetwork, _ieDeviceName, config);
            InferenceEngine::CNNNetwork net = exec.GetExecGraphInfo();
            if (options.debugPrint & (1 << Synet::DebugPrintLayerDst))
                net.serialize(MakePath(options.outputDirectory, "ie_exec_orig.xml"));
            for (InferenceEngine::details::CNNNetworkIterator it = net.begin(); it != net.end(); ++it)
            {
                const InferenceEngine::CNNLayer& layer = **it;
                StringMap::const_iterator names = layer.params.find("originalLayersNames");
                StringMap::const_iterator order = layer.params.find("execOrder");
                if (names != layer.params.end() && names->second.size() && order != layer.params.end())
                {
                    String name = Synet::Separate(names->second, ",").back();

                    bool unique = true;
                    for (size_t o = 0; unique && o < _outputNames.size(); ++o)
                        if (_outputNames[o] == name)
                            unique = false;
                    if (unique)
                    {
                        String number = order->second;
                        while (number.size() < 6)
                            number = String("0") + number;
                        interim[number] = name;
                    }
                }
            }
            for (StringMap::const_iterator it = interim.begin(); it != interim.end(); ++it)
            {
                String name = it->second;
                _ieNetwork->addOutput(name);
                _interimNames.push_back(name);
            }
#endif
        }

        void DebugPrint(std::ostream& os, InferenceEngine::Blob& blob, const String& name, int flag, int first, int last, int precision)
        {
            os << "Layer: " << name;
#if SYNET_TEST_IE_VERSION < 202101
            os << " : " << GetLayerType(name);
#endif
            os << " : " << std::endl;
            Sizes dims = blob.getTensorDesc().getDims();
            const Sizes& strides = blob.getTensorDesc().getBlockingDesc().getStrides();
            Synet::TensorFormat format = Synet::TensorFormatUnknown;
            if (blob.getTensorDesc().getLayout() == InferenceEngine::Layout::NHWC)
                format = Synet::TensorFormatNhwc;
            //dims[0] = _batchSize;
            switch (blob.getTensorDesc().getPrecision())
            {
            case InferenceEngine::Precision::FP32:
            {
                Synet::Tensor<float> tensor(Synet::TensorType32f, dims, format);
                const float* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.Data<float>());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::I32:
            {
                Synet::Tensor<int32_t> tensor(Synet::TensorType32i, dims, format);
                const int32_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.Data<float>());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::I64:
            {
                Synet::Tensor<int64_t> tensor(Synet::TensorType64i, dims, format);
                const int64_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.Data<float>());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::U8:
            {
                Synet::Tensor<uint8_t> tensor(Synet::TensorType8u, dims, format);
                const uint8_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.Data<float>());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case InferenceEngine::Precision::I8:
            {
                Synet::Tensor<int8_t> tensor(Synet::TensorType8i, dims, format);
                const int8_t* pOut = blob.buffer();
                SetOutput(dims, strides, 0, pOut, tensor.Data<float>());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            default:
                CPL_LOG_SS(Error, "Can't debug print for layer '" << name << "' , unknown precision: " << blob.getTensorDesc().getPrecision());
                break;
            }
        }

#if SYNET_TEST_IE_VERSION < 202101
        String GetLayerType(const String& name)
        {
            for (InferenceEngine::details::CNNNetworkIterator it = _ieNetwork->begin(); it != _ieNetwork->end(); ++it)
            {
                const InferenceEngine::CNNLayer& layer = **it;
                if (layer.name == name)
                    return layer.type;
            }
            return String();
        }
#endif
    };
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


