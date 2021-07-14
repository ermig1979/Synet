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
            return 0;
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape;
            return shape;
        }

        virtual size_t SrcSize(size_t index) const
        {
            return 0;
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
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

            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0, n = _session->GetInputCount(); i < n; i++)
            {
                _inputNames.push_back(_session->GetInputName(i, allocator));
                _inputDims.push_back(_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            }

        //    if (_inputDim[0] == -1)
        //    {
        //        std::cout << fullModelPath << " has dynamic input[0] : { ";
        //        for (int j = 0; j < _inputDim.size(); j++)
        //            std::cout << _inputDim[j] << " ";
        //        std::cout << "}. Try to set batch " << _batchSize << "." << std::endl;
        //        _inputDim[0] = _batchSize;
        //}

        //    for (size_t i = 0, n = _session->GetOutputCount(); i < n; i++)
        //    {
        //        _outputNames.push_back(_session->GetOutputName(i, allocator));
        //        _outputDims.push_back(_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        //    }

        //    Synet::Shape inputShape;
        //    for (int i = 0; i < _inputDim.size(); i++)
        //        inputShape.push_back((size_t)_inputDim[i]);
        //    _inputTensor.Reshape(inputShape);
        //    Ort::Value inputValue = Ort::Value::CreateTensor<float>(_memoryInfo, _inputTensor.CpuData(), _inputTensor.Size(), _inputDim.data(), _inputDim.size());

        //    for (size_t i = 0; i < _outputNames.size(); i++)
        //        _outputValues.emplace_back(nullptr);

        //    _session->Run(Ort::RunOptions{ nullptr }, _inputNames.data(), &inputValue, _inputNames.size(),
        //        _outputNames.data(), _outputValues.data(), _outputNames.size());
        //    assert(_outputValues.size() == _outputNames.size() && _outputValues.front().IsTensor());

            return false;
        }

        virtual void Free()
        {
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream & os, int flag, int first, int last, int precision)
        {
        };

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

        std::shared_ptr<Ort::MemoryInfo> _memoryInfo;
        std::shared_ptr<Ort::Session> _session;

        std::vector<const char*> _inputNames;
        Dims _inputDims;
        Tensors _input;

        std::vector<const char*> _outputNames;
        //std::vector<Ort::Value> _outputValues;
        Dims _outputDims;

        size_t _batchSize;

        struct Env
        {
            Ort::MemoryInfo memoryInfo; 
            Ort::Env env;

            Env()
                : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
            {
                env = Ort::Env{ ORT_LOGGING_LEVEL_WARNING, "RootLogger" };
            }
        };
        static Env s_env;
    };
}

#endif

