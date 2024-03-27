/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
#include "TestRegionDecoder.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include <stdexcept>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"

namespace Test
{
    struct OnnxRuntimeNetwork : public Network
    {
        OnnxRuntimeNetwork()
        {
            _inputValues = std::make_shared<Values>();
            _outputValues = std::make_shared<Values>();
        }

        virtual ~OnnxRuntimeNetwork()
        {
        }

        virtual String Name() const
        {
            return "OnnxRuntime";
        }

        virtual size_t SrcCount() const
        {
            return _inputShapes.size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _inputShapes[index];
            if (_batchSize > 1)
            {
                assert(shape.size() >= 2);
                shape[0] = _batchSize;
            }
            return shape;
        }

        virtual Synet::TensorType SrcType(size_t index) const
        {
            return _inputTypes[index];
        }

        virtual size_t SrcSize(size_t index) const
        {
            return Synet::Detail::Size(SrcShape(index));
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
            CPL_PERF_FUNC();
            Free();
            _regionThreshold = options.regionThreshold;
            _decoderName = param.detection().decoder();

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            sessionOptions.SetInterOpNumThreads((int)options.workThreads);
            sessionOptions.SetIntraOpNumThreads((int)options.workThreads);
            if (OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, 0) != nullptr)
                SYNET_ERROR("Can not Initialize ONNXRT CPU Session!");
            std::stringstream logName;
            logName << "log_";
            logName << std::hex << std::this_thread::get_id();
            logName << ".txt";
            sessionOptions.SetLogId(logName.str().c_str());
            sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

            _session.reset(new Ort::Session(s_env.env, weight.c_str(), sessionOptions));

            _inputNameBuffers.reserve(_session->GetInputCount());
            for (size_t i = 0; i < _session->GetInputCount(); i++)
            {
                _inputNameBuffers.push_back(String(_session->GetInputNameAllocated(i, s_env.allocator).get()));
                _inputNames.push_back(_inputNameBuffers[i].c_str());
                _inputShapes.push_back(Convert<size_t, int64_t>(_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
                _inputTypes.push_back(Convert(_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType()));
                _inputValues->emplace_back(nullptr);
            }

            if(param.input().size() && _session->GetInputCount() != param.input().size())
                SYNET_ERROR("Check parameter 'input' size!");

            if (_inputShapes[0][2] == -1 && _inputShapes[0][3] == -1)
            {
                if (param.input().empty())
                    SYNET_ERROR("OnnxRuntime model has dynamic size but input size in parameters is absent!");
                if (param.input()[0].dims().size() == 4)
                {
                    _inputShapes[0][2] = param.input()[0].dims()[2];
                    _inputShapes[0][3] = param.input()[0].dims()[3];
                }
                else
                {
                    _inputShapes[0][2] = param.input()[0].shape()[2].size();
                    _inputShapes[0][3] = param.input()[0].shape()[3].size();
                }
            }

            for (size_t i = 0; i < _inputShapes.size(); ++i)
            {
                if (_inputShapes[i][0] == -1)
                {
                    _inputShapes[i][0] = options.batchSize;
                    _batchSize = 1;
                }
                else
                {
                    _batchSize = options.batchSize;
                    if (_batchSize > 1 && !options.consoleSilence && i == 0)
                        CPL_LOG_SS(Warning, "OnnxRuntime model can't be reshaped, try to emulate batch > 1.");
                }
            }

            if (param.output().size())
            {
                _outputNameBuffers.reserve(param.output().size());
                for (size_t i = 0; i < param.output().size(); ++i)
                {
                    _outputNameBuffers.push_back(param.output()[i].name());
                    _outputNames.push_back(_outputNameBuffers[i].c_str());
                }
            }
            else
            {
                _outputNameBuffers.reserve(_session->GetOutputCount());
                for (size_t i = 0; i < _session->GetOutputCount(); i++)
                {
                    _outputNameBuffers.push_back(String(_session->GetOutputNameAllocated(i, s_env.allocator).get()));
                    _outputNames.push_back(_outputNameBuffers[i].c_str());
                }
                std::sort(_outputNames.begin(), _outputNames.end(), [](const char* a, const char* b) -> bool { return strcmp(a, b) < 1; });
            }

            Tensors stubInput(SrcCount());
            for (size_t s = 0; s < stubInput.size(); ++s)
            { 
                stubInput[s].Reshape(_inputTypes[s], _inputShapes[s], Synet::TensorFormatUnknown);
                if (!TensorToValue(stubInput[s], _inputValues->at(s)))
                    SYNET_ERROR("Can't create stub input tensors for first Ort session run!");
            }

            ClearOutputValues();

            _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), _inputValues->data(), _inputNames.size(),
                _outputNames.data(), _outputValues->data(), _outputNames.size());

            if (!(_outputValues->size() == _outputNames.size() && _outputValues->front().IsTensor()))
                SYNET_ERROR("Check parameter 'output' size!");

            _dynamicOutput = IsDynamicOutput() || param.dynamicOutput();
            if(!_dynamicOutput)
                ReshapeOutput();

            _regionDecoder.Init(_inputShapes[0], _outputNameBuffers, param);
            if (param.detection().decoder() == "iim")
                _iim.Init(param.detection().iim());
            if (param.detection().decoder() == "rtdetr")
                _rtdetr.Init();

            return true;
        }

        virtual void Free()
        {
            _session.reset();
            _inputNameBuffers.clear();
            _inputNames.clear();
            _inputShapes.clear();
            _inputTypes.clear();
            _inputValues->clear();
            _outputNameBuffers.clear();
            _outputNames.clear();
            _outputValues->clear();
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            if (_batchSize == 1)
            {
                SetInput(src, 0);
                if (_dynamicOutput)
                    ClearOutputValues();
                {
                    CPL_PERF_FUNC();
                    _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), _inputValues->data(), _inputNames.size(),
                        _outputNames.data(), _outputValues->data(), _outputNames.size());
                }
                if (_dynamicOutput)
                    ReshapeOutput();
                SetOutput(0);
            }
            else
            {
                if (_dynamicOutput)
                    assert(0);
                CPL_PERF_BEG("batch emulation");
                for (size_t b = 0; b < _batchSize; ++b)
                {
                    SetInput(src, b);
                    _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), _inputValues->data(), _inputNames.size(),
                        _outputNames.data(), _outputValues->data(), _outputNames.size());
                    SetOutput(b);
                }
            }
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream & os, int flag, int first, int last, int precision)
        {
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                os << "Layer: " << _outputNames[i] << " : " << std::endl;
                _output[i].DebugPrint(os, "dst[0]", false, first, last, precision);
            }
        }

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            if (_regionDecoder.Enable())
                return _regionDecoder.GetRegions(_output, size, threshold, overlap);
            else if (_iim.Enable())
            {
                const float* bin = NULL;
                for (size_t i = 0; i < _outputNames.size(); ++i)
                    if (_outputNames[i] == _iim.Name())
                        bin = _output[0].CpuData();
                return _iim.GetRegions(bin, _inputShapes[0][3], _inputShapes[0][2], size.x, size.y);
            }
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(_output[0].CpuData(), _output[0].Axis(1), size.x, size.y, threshold, overlap);
            else
            {
                Regions regions;
                if (_output[0].Axis(-1) == 7)
                {
                    for (size_t i = 0; i < _output[0].Size(); i += 7)
                    {
                        const float* output = _output[0].CpuData();
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
        }

        virtual size_t MemoryUsage() const
        {
            return 0;
        }

    private:
        typedef std::vector<int64_t> Dim;
        typedef std::vector<Dim> Dims;
        typedef Ort::Value Value;
        typedef std::vector<Value> Values;
        typedef std::shared_ptr<Values> ValuesPtr;

        std::shared_ptr<Ort::Session> _session;

        Strings _inputNameBuffers, _outputNameBuffers;
        std::vector<const char*> _inputNames;
        Shapes _inputShapes;
        std::vector<Synet::TensorType> _inputTypes;
        ValuesPtr _inputValues;

        std::vector<const char*> _outputNames;
        ValuesPtr _outputValues;

        size_t _batchSize;
        bool _dynamicOutput;

        RegionDecoder _regionDecoder;
        Synet::IimDecoder _iim;
        Synet::RtdetrDecoder _rtdetr;

        struct Env
        {
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::Env env;
            Ort::MemoryInfo memoryInfo; 

            Env()
                : env( { ORT_LOGGING_LEVEL_WARNING, "RootLogger" } )
                , memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
            {
            }
        };
        static Env s_env;

        Synet::TensorType Convert(const ONNXTensorElementDataType &src)
        {
            switch (src)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return Synet::TensorType32f;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return Synet::TensorType32i;
            default: return Synet::TensorTypeUnknown;
            }
        }

        bool TensorToValue(Tensor& src, Value& dst)
        {
            Dim dim = Convert<int64_t, size_t>(src.Shape());
            switch (src.GetType())
            {
            case Synet::TensorType32f: dst = Ort::Value::CreateTensor<float>(s_env.memoryInfo, src.Data<float>(), src.Size(), dim.data(), dim.size()); break;
            case Synet::TensorType32i: dst = Ort::Value::CreateTensor<int32_t>(s_env.memoryInfo, src.Data<int32_t>(), src.Size(), dim.data(), dim.size()); break;
            default:
                return false;
            }
            return dst.IsTensor();
        }

        template<class D, class S> std::vector<D> Convert(const std::vector<S>& src)
        {
            std::vector<D> dst(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                dst[i] = (D)src[i];
            return dst;
        }

        void SetInput(const Tensors& src, size_t b)
        {
            assert(_inputNames.size() == src.size());
            for (size_t i = 0; i < src.size(); i++)
            {
                Shape shape = _inputShapes[i], index(shape.size(), 0);
                index[0] = b;
                if(_batchSize > 1)
                    shape[0] = _batchSize;
                size_t size = Synet::Detail::Size(shape);
                assert(src[i].Shape() == shape);
                Dim dim = Convert<int64_t, size_t>(_inputShapes[i]);
                switch (_inputTypes[i])
                {
                case Synet::TensorType32f: _inputValues->at(i) = Ort::Value::CreateTensor<float>(s_env.memoryInfo, (float*)src[i].Data<float>(index), size, dim.data(), dim.size()); break;
                case Synet::TensorType32i: _inputValues->at(i) = Ort::Value::CreateTensor<int32_t>(s_env.memoryInfo, (int32_t*)src[i].Data<int32_t>(index), size, dim.data(), dim.size()); break;
                default:
                    break;
                }
            }
        }

        void SetOutput(size_t b)
        {
            _output.resize(_outputNames.size());
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                Shape shape = Convert<size_t, int64_t>(_outputValues->at(i).GetTensorTypeAndShapeInfo().GetShape());
                if (shape.empty())
                    shape = Shp(1);
                Shape index(shape.size(), 0);
                index[0] = b;
                float* dst = _output[i].Data<float>(index);
                size_t size = Synet::Detail::Size(shape);
                ONNXTensorElementDataType type = _outputValues->at(i).GetTensorTypeAndShapeInfo().GetElementType();
                if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                {
                    const float * src = _outputValues->at(i).GetTensorMutableData<float>();
                    if (_decoderName == "rtdetr")
                    {
                        assert(shape[0] == 1 && shape[2] == 6);
                        Vector tmp;
                        for (size_t j = 0, n = shape[1]; j < n; ++j, src += 6)
                        {
                            if (src[4] <= _regionThreshold)
                                continue;
                            size_t offset = tmp.size();
                            tmp.resize(offset + 6);
                            tmp[offset + 0] = src[0];
                            tmp[offset + 1] = src[1];
                            tmp[offset + 2] = src[2];
                            tmp[offset + 3] = src[3];
                            tmp[offset + 4] = src[4];
                            tmp[offset + 5] = src[5];
                        }
                        SortRtdetr(tmp.data(), tmp.size());
                        _output[i].Reshape(Shp(1, tmp.size() / 6, 6));
                        memcpy(_output[i].CpuData(), tmp.data(), _output[i].Size() * sizeof(float));
                    }
                    else
                    {
                        for (size_t j = 0; j < size; ++j)
                            dst[j] = src[j];
                    }
                }
                else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
                {
                    const int64_t * src = _outputValues->at(i).GetTensorMutableData<int64_t>();
                    for (size_t j = 0; j < size; ++j)
                        dst[j] = (float)src[j];
                }
                else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
                {
                    const uint8_t* src = _outputValues->at(i).GetTensorMutableData<uint8_t>();
                    for (size_t j = 0; j < size; ++j)
                        dst[j] = (float)src[j];
                }
                else
                {
                    CPL_LOG_SS(Error, "OnnxRuntime: unknown format of output tensor: " << type << " !");
                    assert(0);
                }
                _output[i].SetName(_outputNames[i]);
            }
        }

        bool IsDynamicOutput()
        {
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                Shape shape = Convert<size_t, int64_t>(_outputValues->at(i).GetTensorTypeAndShapeInfo().GetShape());
                size_t size = Synet::Detail::Size(shape);
                if (size == 0)
                    return true;
            }
            return false;
        }

        void ClearOutputValues()
        {
            _outputValues->clear();
            for (size_t i = 0; i < _outputNames.size(); i++)
                _outputValues->emplace_back(nullptr);
        }

        void ReshapeOutput()
        {
            _output.resize(_outputNames.size());
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                Shape shape = Convert<size_t, int64_t>(_outputValues->at(i).GetTensorTypeAndShapeInfo().GetShape());
                if (shape.empty())
                    shape = Shp(1);
                if (_batchSize > 1)
                    shape[0] = _batchSize;
                _output[i].Reshape(shape);
            }
        }
    };
}

#endif

