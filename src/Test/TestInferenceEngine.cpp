/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

        virtual bool Init(const String & model, const String & weight, size_t threadNumber, const TestParam & param)
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
                InferenceEngine::ExecutableNetwork executableNet = _iePlugin.LoadNetwork(network, config);
                _ieInferRequest = executableNet.CreateInferRequest();

                InferenceEngine::InputsDataMap inputsInfo = network.getInputsInfo();
                assert(inputsInfo.size() == 1);
                InferenceEngine::InputsDataMap::iterator input = inputsInfo.begin();
                auto inputName = input->first;
                input->second->setPrecision(InferenceEngine::Precision::FP32);
                _ieInput = _ieInferRequest.GetBlob(inputName);

                InferenceEngine::OutputsDataMap outputsInfo = network.getOutputsInfo();
                for (InferenceEngine::OutputsDataMap::iterator it = outputsInfo.begin(); it != outputsInfo.end(); ++it)
                {
                    //std::cout << "init " << it->first << std::endl;
                    _names.push_back(it->first);
                    it->second->setPrecision(InferenceEngine::Precision::FP32);
                    _ieOutput.push_back(_ieInferRequest.GetBlob(it->first));
                    //std::cout << "init " << it->first << std::endl;
                }
             }
            catch (std::exception & e) 
            {
                std::cout << "Inference Engine init error: " << e.what() << std::endl;
                return false;
            }

            num = param.input()[0].shape()[0].size();
            channels = param.input()[0].shape()[1].size();
            height = param.input()[0].shape()[2].size();
            width = param.input()[0].shape()[3].size();

            if (param.output().size())
            {
                for (size_t i = 0; i < param.output().size(); ++i)
                    _names.push_back(param.output()[i].name());
            }
            else
            {
                //std::vector<int> outLayers = _net.getUnconnectedOutLayers();
                //for (size_t i = 0; i < outLayers.size(); ++i)
                //    _names.push_back(_net.getLayer(outLayers[i])->name);
            }

            {
                Vector stub(num*channels*height*width);
                SetInput(stub);
                _ieInferRequest.Infer();
                //Ints sizes{ int(num), int(channels), int(height), int(width) };
                //cv::Mat ins(sizes, CV_32F, (void*)stub.data());
                //_net.setInput(ins);
                //if (_names.empty())
                //    _net.forward(_out);
                //else
                //    _net.forward(_out, _names);
            }

            return true;
        }

        virtual const Vector & Predict(const Vector & src)
        {
            SetInput(src);
            {
                TEST_PERF_FUNC();
                _ieInferRequest.Infer();
            }
            SetOutput();
            return _output;
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        virtual void DebugPrint(std::ostream & os)
        {
            _ieInferRequest.Infer();
            for (size_t o = 0; o < _ieOutput.size(); ++o)
            {
                const InferenceEngine::SizeVector & dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                size_t size = 1;
                for (size_t i = 0; i < dims.size(); ++i)
                    size *= dims[i];
                Synet::Tensor<float> tensor(dims);
                SetOutput(dims, strides, 0, _ieOutput[o]->buffer(), tensor.CpuData());
                tensor.DebugPrint(os, _names.empty() ? String("???") : String(_names[o]), false);
            }
        }
#endif

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            //int nboxes = 0;
            //float hier_thresh = 0.5;
            //::layer l = _net->layers[_net->n - 1];
            //::detection *dets = get_network_boxes((::network*)_net, (int)size.x, (int)size.y, threshold, hier_thresh, 0, 1, &nboxes);
            //if (overlap)
            //    do_nms_sort(dets, nboxes, l.classes, overlap);
            //Regions regions;
            //for (size_t i = 0; i < nboxes; ++i)
            //{
            //    box b = dets[i].bbox;
            //    int const obj_id = max_index(dets[i].prob, l.classes);
            //    float const prob = dets[i].prob[obj_id];

            //    if (prob > threshold)
            //    {
            //        Region region;
            //        region.x = b.x*size.x;
            //        region.y = b.y*size.y;
            //        region.w = b.w*size.x;
            //        region.h = b.h*size.y;
            //        region.id = obj_id;
            //        region.prob = prob;
            //        regions.push_back(region);
            //    }
            //}
            //free_detections(dets, nboxes);
            //return regions;
        }

    private:
        InferenceEngine::InferencePlugin _iePlugin;
        InferenceEngine::InferRequest _ieInferRequest;
        InferenceEngine::Blob::Ptr _ieInput;
        std::vector<InferenceEngine::Blob::Ptr> _ieOutput;
        Vector _output;
        Strings _names;

        void SetInput(const Vector & x)
        {
            assert(_ieInput->getTensorDesc().getLayout() == InferenceEngine::Layout::NCHW);
            const InferenceEngine::SizeVector & strides = _ieInput->getTensorDesc().getBlockingDesc().getStrides();
            //for (size_t i = 0; i < strides.size(); ++i)
            //    std::cout << "i strides[" << i << "]=" << strides[i] << std::endl;

            const float * src = x.data();
            float * dst = (float*)_ieInput->buffer();
            for (size_t i = 0; i < channels; ++i)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    memcpy(dst + row*strides[2], src, sizeof(float)*width);
                    src += width;
                }
                dst += strides[1];
            }
        }

        typedef InferenceEngine::SizeVector Sizes;

        void SetOutput()
        {
            size_t offset = 0;
            for (size_t o = 0; o < _ieOutput.size(); ++o)
            {
                const InferenceEngine::SizeVector & dims = _ieOutput[o]->getTensorDesc().getDims();
                const InferenceEngine::SizeVector & strides = _ieOutput[o]->getTensorDesc().getBlockingDesc().getStrides();
                size_t size = 1;
                for (size_t i = 0; i < dims.size(); ++i)
                    size *= dims[i];
                _output.resize(offset + size);
                SetOutput(dims, strides, 0, _ieOutput[o]->buffer(), _output.data() + offset);
                //for (size_t i = 0; i < strides.size(); ++i)
                //    std::cout << "strides[" << i << "]=" << strides[i] << std::endl;
                //for (size_t i = 0; i < dims.size(); ++i)
                //    std::cout << "dims[" << i << "]=" << dims[i] << std::endl;
                offset += size;
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
                //std::cout << "s " << srcStride << " d " << dstStride << std::endl;
                for(size_t i = 0; i < dims[current]; ++i)
                    SetOutput(dims, strides, current + 1, src + i * srcStride, dst + i * dstStride);
            }
        }
    };
}

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