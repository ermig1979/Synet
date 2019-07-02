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

#define SYNET_DARKNET_ENABLE
#define SYNET_DARKNET_PATH ../../../3rd/darknet/include
#include "Synet/Converters/Darknet.h"

namespace Test
{
    struct DarknetNetwork : public Network
    {
        DarknetNetwork()
            : _net(0)
        {
        }

        virtual ~DarknetNetwork()
        {
            if(_net)
                ::free_network(_net);
        }

        virtual String Name() const
        {
            return "Darknet";
        }

        virtual bool Init(const String & model, const String & weight, size_t threadNumber, const TestParam & param)
        {
            TEST_PERF_FUNC();
            _net = ::parse_network_cfg((char*)model.c_str());
            ::load_weights(_net, (char*)weight.c_str());
            //::set_batch_network(_net, 1);
            num = _net->batch;
            channels = _net->c;
            height = _net->h;
            width = _net->w;
            return true;
        }

        virtual const Vector & Predict(const Vector & src)
        {
            {
                TEST_PERF_FUNC();
                ::network_predict(_net, (float*)src.data());
            }
            SetOutput();
            return _output;
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        virtual void DebugPrint(std::ostream & os)
        {
            for (int i = 0; i < _net->n; ++i)
            {
                os << "Layer: " << i << " : " << std::endl;
                ::layer l = _net->layers[i];
                if (l.type == CONVOLUTIONAL)
                {
                    Synet::Tensor<float> weight({ (size_t)l.out_c, (size_t)l.c, (size_t)l.size, (size_t)l.size });
                    memcpy(weight.CpuData(), l.weights, weight.Size() * sizeof(float));
                    weight.DebugPrint(os, String("weight"), true);
                    if (l.batch_normalize)
                    {
                        Synet::Tensor<float> mean({ (size_t)l.out_c });
                        memcpy(mean.CpuData(), l.rolling_mean, mean.Size() * sizeof(float));
                        mean.DebugPrint(os, String("mean"), true);

                        Synet::Tensor<float> variance({ (size_t)l.out_c });
                        memcpy(variance.CpuData(), l.rolling_variance, variance.Size() * sizeof(float));
                        variance.DebugPrint(os, String("variance"), true);

                        Synet::Tensor<float> scale({ (size_t)l.out_c });
                        memcpy(scale.CpuData(), l.scales, scale.Size() * sizeof(float));
                        scale.DebugPrint(os, String("scale"), true);
                    }
                    Synet::Tensor<float> bias({ (size_t)l.out_c });
                    memcpy(bias.CpuData(), l.biases, bias.Size() * sizeof(float));
                    bias.DebugPrint(os, String("bias"), true);
                }
                Synet::Tensor<float> dst({ size_t(1), (size_t)l.out_c, (size_t)l.out_h, (size_t)l.out_w });
                memcpy(dst.CpuData(), l.output, dst.Size() * sizeof(float));
                dst.DebugPrint(os, String("dst[0]"), false);
            }
        }
#endif

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            int nboxes = 0;
            float hier_thresh = 0.5;
            ::layer l = _net->layers[_net->n - 1];
            ::detection *dets = get_network_boxes((::network*)_net, (int)size.x, (int)size.y, threshold, hier_thresh, 0, 1, &nboxes);
            if (overlap)
                do_nms_sort(dets, nboxes, l.classes, overlap);
            Regions regions;
            for (size_t i = 0; i < nboxes; ++i)
            {
                box b = dets[i].bbox;
                int const obj_id = max_index(dets[i].prob, l.classes);
                float const prob = dets[i].prob[obj_id];

                if (prob > threshold)
                {
                    Region region;
                    region.x = b.x*size.x;
                    region.y = b.y*size.y;
                    region.w = b.w*size.x;
                    region.h = b.h*size.y;
                    region.id = obj_id;
                    region.prob = prob;
                    regions.push_back(region);
                }
            }
            free_detections(dets, nboxes);
            return regions;
        }

    private:
        ::network * _net;

        void SetOutput()
        {
            size_t offset = 0;
            for (size_t i = 0; i < _net->n; ++i)
            {
                const layer & l = _net->layers[i];
                if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
                    AddToOutput(l, offset);
            }
            if (offset == 0)
                AddToOutput(_net->layers[_net->n - 1], offset);
        }

        void AddToOutput(const layer & l, size_t & offset)
        {
            size_t size = l.outputs*l.batch;
            if (offset + size > _output.size())
                _output.resize(offset + size);
            memcpy(_output.data() + offset, l.output, size * sizeof(float));
            offset += size;
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
        std::cout << "Convert network from Yolo to Synet :" << std::endl;
        options.result = Synet::ConvertDarknetToSynet(options.otherModel, options.otherWeight, options.tensorFormat == 1, options.synetModel, options.synetWeight);
        std::cout << "Conversion is finished " << (options.result ? "successfully." : "with errors.") << std::endl;
    }
    else if (options.mode == "compare")
        options.result = Test::CompareOtherAndSynet<Test::DarknetNetwork>(options);
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}