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

#include <openvino/openvino.hpp>

#if defined(SYNET_TEST_OPENVINO_EXTENSIONS)
#ifndef WITH_INF_ENGINE
#define WITH_INF_ENGINE
#endif
#include "OpenvinoExtensions/priorbox_v2.hpp"
#endif

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
            return _ov->model->inputs().size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _ov->model->input(index).get_shape();
            if (_ov->batchSize > 1)
            {
                assert(shape.size() >= 2);
                shape[0] = _ov->batchSize;
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
                if (!InitCore(options))
                    return false;

                if (!ReadNetwork(model, weight, param.model() == "onnx"))
                    return false;

                if (!InitInput(param))
                    return false;

                if (!InitOutput(param))
                    return false;

                _ov->batchSize = 1;
                if (options.batchSize > 1)
                {
                    if (_ov->model->is_dynamic())
                    {
                        SYNET_ERROR("Inference Engine model is dynamic. This case is not implemented!");
                    }
                    else
                    {
                        if (!options.consoleSilence)
                            CPL_LOG_SS(Warning, "Inference Engine model is static. Try to emulate batch > 1.");
                        _ov->batchSize = options.batchSize;
                        CreateCompiledModelAndInferRequest();
                    }
                }
                else
                    CreateCompiledModelAndInferRequest();
                GetTensors();
                StubInfer();
            }
            catch (std::exception& e)
            {
                SYNET_ERROR("Inference Engine init error: " << e.what());
            }
            return true;
        }

        virtual const Tensors& Predict(const Tensors& src)
        {
            if (_ov->batchSize == 1)
            {
                SetInput(src, 0);
                {
                    CPL_PERF_FUNC();
                    _ov->inferRequest.infer();
                }
                SetOutput(0);
            }
            else
            {
                CPL_PERF_BEG("batch emulation");
                for (size_t b = 0; b < _ov->batchSize; ++b)
                {
                    SetInput(src, b);
                    _ov->inferRequest.infer();
                    SetOutput(b);
                }
            }
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream& os, int flag, int first, int last, int precision)
        {
            for (size_t o = 0; o < _ov->output.size(); ++o)
                DebugPrint(os, _ov->output[o], _ov->outputNames[o], flag, first, last, precision);
        }

        virtual Regions GetRegions(const Size& size, float threshold, float overlap) const
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

        virtual void Free()
        {
            Network::Free();
            _ov.reset();
        }

    private:
        typedef InferenceEngine::SizeVector Sizes;
        typedef std::map<std::string, std::string> StringMap;
        struct Ov
        {
            ov::Core core;
            std::shared_ptr<ov::Model> model;
            ov::CompiledModel compiledModel;
            ov::InferRequest inferRequest;
            std::vector<ov::Tensor> input, output;
            Strings inputNames, outputNames;
            size_t batchSize;
        };
        typedef std::shared_ptr<Ov> OvPtr;
        OvPtr _ov;
        const std::string _ieDeviceName = "CPU";

        bool InitCore(const Options& options)
        {
            _ov = std::make_shared<Ov>();
#if defined(SYNET_TEST_OPENVINO_EXTENSIONS)
            _ov->core.add_extension(ov::OpExtension<OpenvinoCustomExtension::PriorBoxV2>());
            if (!options.consoleSilence && 0)
                std::cout << "Inference Engine uses PriorBoxV2 extension." << std::endl;
#endif
            _ov->core.set_property(_ieDeviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
            return true;
        }

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
                {
                    std::cout << "Can't copy file form '" << src << "' to '" << dst << "' !" << std::endl;
                    return false;
                }
            } 
            _ov->model = _ov->core.read_model(model, weight);
            return true;
        }

        bool InitInput(const TestParam& param)
        {
            _ov->inputNames.clear();
            if (param.input().size())
            {
                if (_ov->model->inputs().size() != param.input().size())
                    SYNET_ERROR("Incorrect input count :" << param.input().size());
                for (size_t i = 0; i < param.input().size(); ++i)
                {
                    const String & name = param.input()[i].name();
                    _ov->inputNames.push_back(name);
                    bool found = false;
                    for (size_t j = 0; j < _ov->model->inputs().size(); ++j)
                    {
                        if (_ov->model->inputs()[j].get_any_name() == name)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        SYNET_ERROR("Input with name '" << name << "' is not exist! ");
                }
            }
            else
            {
                for (size_t i = 0; i < _ov->model->inputs().size(); ++i)
                    _ov->inputNames.push_back(_ov->model->inputs()[i].get_any_name());
            }
            return true;
        }

        bool InitOutput(const TestParam& param)
        {
            _ov->outputNames.clear();
            if (param.output().size())
            {
                for (size_t i = 0; i < param.output().size(); ++i)
                    _ov->outputNames.push_back(param.output()[i].name());
            }
            else
            {
                for (size_t i = 0; i < _ov->model->outputs().size(); ++i)
                    _ov->outputNames.push_back(_ov->model->outputs()[i].get_any_name());
                std::sort(_ov->outputNames.begin(), _ov->outputNames.end());
            }
            return true;
        }

        void CreateCompiledModelAndInferRequest()
        {
            _ov->compiledModel = _ov->core.compile_model(_ov->model, _ieDeviceName,
                ov::inference_num_threads(1),
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                ov::num_streams(1),
                ov::affinity(ov::Affinity::CORE));
            _ov->inferRequest = _ov->compiledModel.create_infer_request();
        }

        void GetTensors()
        {
            _ov->input.resize(_ov->inputNames.size());
            for (size_t i = 0; i < _ov->inputNames.size(); ++i)
                _ov->input[i] = _ov->inferRequest.get_tensor(_ov->inputNames[i]);
            _ov->output.resize(_ov->outputNames.size());
            for (size_t i = 0; i < _ov->outputNames.size(); ++i)
                _ov->output[i] = _ov->inferRequest.get_tensor(_ov->outputNames[i]);
        }

        void StubInfer()
        {
            Tensors stub(SrcCount());
            for (size_t i = 0; i < SrcCount(); ++i)
                stub[i].Reshape(SrcShape(i));
            SetInput(stub, 0);
            _ov->inferRequest.infer();
        }

        void SetInput(const Tensors& x, size_t b)
        {
            assert(_ov->input.size() == x.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                const float* src = x[i].CpuData() + b * x[i].Size() / _ov->batchSize;
                const ov::Shape& dims = _ov->input[i].get_shape();
                const ov::Strides & strides = _ov->input[i].get_strides();
                uint8_t* dst = (uint8_t*)_ov->input[i].data();
                SetInput(dims, strides, 0, src, dst);
            }
        }

        void SetInput(const ov::Shape& dims, const ov::Strides& strides, size_t current, const float* src, uint8_t* dst)
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
            _output.resize(_ov->output.size());
            for (size_t o = 0; o < _ov->output.size(); ++o)
            {
                const ov::Shape& dims = _ov->output[o].get_shape();
                const ov::Strides& strides = _ov->output[o].get_strides();
                ov::element::Type_t type = _ov->output[o].get_element_type();
                if (dims.size() == 4 && dims[3] == 7 && _ov->outputNames[o].find("Yolo") == std::string::npos)
                {
                    assert(dims[0] == 1);
                    Vector tmp;
                    const float* pOut = (float*)_ov->output[o].data();
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
                    memcpy(_output[o].CpuData(), tmp.data(), _output[o].Size() * sizeof(float));
                }
                else
                {
                    if (b == 0)
                    {
                        Shape shape = dims;
                        if (_ov->batchSize != 1)
                        {
                            if (shape[0] == 1)
                                shape[0] = _ov->batchSize;
                            else
                                shape.insert(shape.begin(), _ov->batchSize);
                        }
                        _output[o].Reshape(shape, Synet::TensorFormatNchw);
                    }
                    size_t size = 1;
                    for (size_t i = 0; i < dims.size(); ++i)
                        size *= dims[i];
                    switch (type)
                    {
                    case ov::element::Type_t::f32:
                        SetOutput(dims, strides, 0, _ov->output[o].data<float>(), _output[o].CpuData() + b * size);
                        break;
                    case ov::element::Type_t::i32:
                        SetOutput(dims, strides, 0, _ov->output[o].data<int32_t>(), _output[o].CpuData() + b * size);
                        break;
                    case ov::element::Type_t::i64:
                        SetOutput(dims, strides, 0, _ov->output[o].data<int64_t>(), _output[o].CpuData() + b * size);
                        break;
                    default:
                        CPL_LOG_SS(Error, "OpenVino wrapper: unknown type of output tensor!");
                        assert(0);
                    }
                }
            }
        }

        template<class S, class D> void SetOutput(const ov::Shape& dims, const ov::Strides& strides, size_t current, const S* src, D* dst)
        {
            if (current == dims.size() - 1)
            {
                for (size_t i = 0; i < dims[current]; ++i)
                    dst[i] = (D)src[i];
            }
            else
            {
                size_t srcStride = strides[current] / sizeof(S);
                size_t dstStride = 1;
                for (size_t i = current + 1; i < dims.size(); ++i)
                    dstStride *= dims[i];
                for (size_t i = 0; i < dims[current]; ++i)
                    SetOutput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }

        void DebugPrint(std::ostream& os, const ov::Tensor & src, const String & name, int flag, int first, int last, int precision)
        {
            os << "Layer: " << name;
            os << " : " << std::endl;
            Shape dims = src.get_shape();
            Shape strides = src.get_strides();
            Synet::TensorFormat format = Synet::TensorFormatNchw;
            if(dims[0] == 1)
                dims[0] = _ov->batchSize;
            ov::element::Type_t type = src.get_element_type();
            switch (type)
            {
            case ov::element::Type_t::f32:
            {
                Synet::Tensor<float> tensor(dims, format);
                const float* pOut = (float*)src.data();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case ov::element::Type_t::i32:
            {
                Synet::Tensor<int32_t> tensor(dims, format);
                const int32_t* pOut = (int32_t*)src.data();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case ov::element::Type_t::i64:
            {
                Synet::Tensor<int64_t> tensor(dims, format);
                const int64_t* pOut = (int64_t*)src.data();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case ov::element::Type_t::u8:
            {
                Synet::Tensor<uint8_t> tensor(dims, format);
                const uint8_t* pOut = (uint8_t*)src.data();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            case ov::element::Type_t::i8:
            {
                Synet::Tensor<int8_t> tensor(dims, format);
                const int8_t* pOut = (int8_t*)src.data();
                SetOutput(dims, strides, 0, pOut, tensor.CpuData());
                tensor.DebugPrint(os, "dst[0]", false, first, last, precision);
                break;
            }
            default:
                CPL_LOG_SS(Error, "Can't debug print for layer '" << name << "' , unknown type: " << src.get_element_type());
                break;
            }
        }
    };
}



