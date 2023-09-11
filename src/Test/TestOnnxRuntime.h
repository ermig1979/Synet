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

        virtual size_t SrcSize(size_t index) const
        {
            return Synet::Detail::Size(SrcShape(index));
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
            CPL_PERF_FUNC();
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

            _inputNameBuffers.clear();
            _inputNames.clear();
            _inputNameBuffers.reserve(_session->GetInputCount());
            for (size_t i = 0; i < _session->GetInputCount(); i++)
            {
                _inputNameBuffers.push_back(String(_session->GetInputNameAllocated(i, s_env.allocator).get()));
                _inputNames.push_back(_inputNameBuffers[i].c_str());
                _inputShapes.push_back(Convert<size_t, int64_t>(_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
            }

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

            if (_inputShapes[0][0] == -1)
            {
                _inputShapes[0][0] = options.batchSize;
                _batchSize = 1;
            }
            else
            {
                _batchSize = options.batchSize;
                if (_batchSize > 1 && !options.consoleSilence)
                    CPL_LOG_SS(Warning, "OnnxRuntime model can't be reshaped, try to emulate batch > 1.");
            }

            _outputNames.clear();
            _outputNameBuffers.clear();
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

            if (_inputShapes.size() != 1)
                SYNET_ERROR("Current implementation of OnnxRuntimeNetwork supports only 1 input!");

            Tensor inputTensor(_inputShapes[0]);
            Dim inputDim = Convert<int64_t, size_t>(_inputShapes[0]);
            Ort::Value inputValue = Ort::Value::CreateTensor<float>(s_env.memoryInfo, inputTensor.CpuData(), inputTensor.Size(), inputDim.data(), inputDim.size());
            if (!inputValue.IsTensor())
                return false;

            _outputValues = std::make_shared<Values>();
            ClearOutputValues();

            _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), &inputValue, _inputNames.size(),
                _outputNames.data(), _outputValues->data(), _outputNames.size());

            if (!(_outputValues->size() == _outputNames.size() && _outputValues->front().IsTensor()))
                return false;

            _dynamicOutput = IsDynamicOutput();
            if(!_dynamicOutput)
                ReshapeOutput();

            if (param.detection().decoder() == "ultraface")
                _ultraface.Init(param.detection().ultraface());
            if (param.detection().decoder() == "yoloV5")
                _yoloV5.Init(param.detection().yoloV5());
            if (param.detection().decoder() == "yoloV7")
                _yoloV7.Init();
            if (param.detection().decoder() == "yoloV8")
                _yoloV8.Init();            
            if (param.detection().decoder() == "iim")
                _iim.Init(param.detection().iim());

            return true;
        }

        virtual void Free()
        {
            _session.reset();
            _inputNames.clear();
            _inputShapes.clear();
            _outputNames.clear();
            _outputValues->clear();
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            Values inputValues;
            if (_batchSize == 1)
            {
                SetInput(src, 0, inputValues);
                if (_dynamicOutput)
                    ClearOutputValues();
                {
                    CPL_PERF_FUNC();
                    _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), inputValues.data(), _inputNames.size(),
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
                    SetInput(src, b, inputValues);
                    _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), inputValues.data(), _inputNames.size(),
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
            if (_ultraface.Enable())
                return _ultraface.GetRegions(_output[0].CpuData(), _output[1].CpuData(), _output[0].Axis(1), size.x, size.y, threshold, overlap);
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(_output[0].CpuData(), _output[0].Axis(1), _inputShapes[0][3], _inputShapes[0][2], size.x, size.y, threshold, overlap);
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(_output[0].CpuData(), _output[0].Axis(1), _inputShapes[0][3], _inputShapes[0][2], size.x, size.y, threshold, overlap);
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(_output[0].CpuData(), _output[0].Axis(2), _inputShapes[0][3], _inputShapes[0][2], size.x, size.y, threshold, overlap);
            else if (_iim.Enable())
            {
                const float* bin = NULL;
                for (size_t i = 0; i < _outputNames.size(); ++i)
                    if (_outputNames[i] == _iim.Name())
                        bin = _output[0].CpuData();
                return _iim.GetRegions(bin, _inputShapes[0][3], _inputShapes[0][2], size.x, size.y);
            }
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

        std::vector<const char*> _inputNames;
        Shapes _inputShapes;

        Strings _inputNameBuffers, _outputNameBuffers;
        std::vector<const char*> _outputNames;
        ValuesPtr _outputValues;

        size_t _batchSize;
        bool _dynamicOutput;

        Synet::UltrafaceDecoder _ultraface;
        Synet::YoloV5Decoder _yoloV5;
        Synet::YoloV7Decoder _yoloV7;
        Synet::YoloV8Decoder _yoloV8;
        Synet::IimDecoder _iim;

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

        template<class D, class S> std::vector<D> Convert(const std::vector<S>& src)
        {
            std::vector<D> dst(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                dst[i] = (D)src[i];
            return dst;
        }

        void SetInput(const Tensors& src, size_t b, Values & inputs)
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
                inputs.emplace_back(nullptr);
                Dim dim = Convert<int64_t, size_t>(_inputShapes[i]);
                inputs[i] = Ort::Value::CreateTensor<float>(s_env.memoryInfo, (float*)src[i].CpuData(index), size, dim.data(), dim.size());
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
                float* dst = _output[i].CpuData(index);
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

