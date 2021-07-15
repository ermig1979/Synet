/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
            return _inputShapes[index];
        }

        virtual size_t SrcSize(size_t index) const
        {
            return Synet::Detail::Size(SrcShape(index));
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
            TEST_PERF_FUNC();

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            sessionOptions.SetInterOpNumThreads((int)options.workThreads);
            sessionOptions.SetIntraOpNumThreads((int)options.workThreads);
            if (OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, 0) != nullptr)
            {
                std::cout << "Can not Initialize ONNXRT CPU Session!" << std::endl;
                return false;
            }
            std::stringstream logName;
            logName << "log_";
            logName << std::hex << std::this_thread::get_id();
            logName << std::endl;
            sessionOptions.SetLogId(logName.str().c_str());
            sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

            _session.reset(new Ort::Session(s_env.env, weight.c_str(), sessionOptions));

            for (size_t i = 0, n = _session->GetInputCount(); i < n; i++)
            {
                _inputNames.push_back(_session->GetInputName(i, s_env.allocator));
                _inputShapes.push_back(Convert<size_t, int64_t>(_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
            }

            _batchSize = options.batchSize;

            if (_inputShapes[0][0] == -1)
            {
                //std::cout << weight << " has dynamic input[0] : " << Synet::Detail::DebugPrint(_inputShapes[0]);
                //std::cout << "}. Try to set batch " << _batchSize << "." << std::endl;
                _inputShapes[0][0] = _batchSize;
            }

            for (size_t i = 0, n = _session->GetOutputCount(); i < n; i++)
            {
                _outputNames.push_back(_session->GetOutputName(i, s_env.allocator));
                _outputShapes.push_back(Convert<size_t, int64_t>(_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()));
            }

            if (_inputShapes.size() != 1)
            {
                std::cout << "Current implementation of OnnxRuntimeNetwork supports only 1 input!" << std::endl;
                return false;
            }

            Tensor inputTensor(_inputShapes[0]);
            Dim inputDim = Convert<int64_t, size_t>(_inputShapes[0]);
            Ort::Value inputValue = Ort::Value::CreateTensor<float>(s_env.memoryInfo, inputTensor.CpuData(), inputTensor.Size(), inputDim.data(), inputDim.size());

            _outputValues = std::make_shared<Values>();
            Values outputValues;
            for (size_t i = 0; i < _outputValues->size(); i++)
                _outputValues->emplace_back(nullptr);

            _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), &inputValue, _inputNames.size(),
                _outputNames.data(), _outputValues->data(), _outputNames.size());

            if (!(_outputValues->size() == _outputNames.size() && _outputValues->front().IsTensor()))
                return false;

            _output.resize(_outputNames.size());
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                _outputShapes[i] = Convert<size_t, int64_t>(_outputValues->at(i).GetTensorTypeAndShapeInfo().GetShape());
                _output[i].Reshape(_outputShapes[i]);
            }

            return true;
        }

        virtual void Free()
        {
            _session.reset();
            _inputNames.clear();
            _inputShapes.clear();
            _outputNames.clear();
            _outputShapes.clear();
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            assert(src.size() == _inputNames.size());

            Values inputValues;
            for (size_t i = 0; i < src.size(); i++)
            {
                inputValues.emplace_back(nullptr);
                assert(src[i].Shape() == _inputShapes[i]);
                Dim dim = Convert<int64_t, size_t>(_inputShapes[i]);
                inputValues[i] = Ort::Value::CreateTensor<float>(s_env.memoryInfo, (float*)src[i].CpuData(), src[i].Size(), dim.data(), dim.size());
            }

            {
                TEST_PERF_FUNC();

                _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), inputValues.data(), _inputNames.size(),
                    _outputNames.data(), _outputValues->data(), _outputNames.size());
            }

            SetOutput();

            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream & os, int flag, int first, int last, int precision)
        {
        }

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            return Regions();
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

        std::vector<const char*> _outputNames;
        Shapes _outputShapes;
        ValuesPtr _outputValues;

        size_t _batchSize;

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

        void SetOutput()
        {
            _output.resize(_outputNames.size());
            for (size_t i = 0; i < _outputNames.size(); i++)
            {
                _outputShapes[i] = Convert<size_t, int64_t>(_outputValues->at(i).GetTensorTypeAndShapeInfo().GetShape());
                _output[i].Reshape(_outputShapes[i]);

                const float * src = _outputValues->at(i).GetTensorMutableData<float>();
                float* dst = _output[i].CpuData();
                size_t size = _output[i].Size();
                for (size_t j = 0; j < size; ++j)
                    dst[i] = src[i];
            }
        }
    };
}

#endif

