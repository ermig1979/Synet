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

#include "TestCompare.h"
#include "TestReport.h"

#ifdef SYNET_TEST_FIRST_RUN

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

        virtual size_t SrcCount() const
        {
            return 1;
        }

        virtual Shape SrcShape(size_t index) const
        {
            return Shape({ size_t(_net->batch), size_t(_net->c), size_t(_net->h), size_t(_net->w) });
        }

        virtual size_t SrcSize(size_t index) const
        {
            return size_t(_net->batch) * _net->c * _net->h * _net->w;
        }

        virtual bool Init(const String & model, const String & weight, const Options & options, const TestParam & param)
        {
            TEST_PERF_FUNC();
            _regionThreshold = options.regionThreshold;
            _net = ::parse_network_cfg((char*)PatchCfg(model, options.batchSize).c_str());
            ::load_weights(_net, (char*)weight.c_str());
            return true;
        }

        virtual const Tensors & Predict(const Tensors & src)
        {
            {
                TEST_PERF_FUNC();
                ::network_predict(_net, (float*)src[0].CpuData());
            }
            SetOutput();
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream& os, int flag, int first, int last, int precision)
        {
            if (!flag)
                return;
            bool output = flag & (1 << Synet::DebugPrintOutput);
            bool weight = flag & (1 << Synet::DebugPrintLayerWeight);
            bool interim = flag & (1 << Synet::DebugPrintLayerDst);
            for (int i = 0; i < _net->n; ++i)
            {
                const ::layer& l = _net->layers[i];
                if (((i == _net->n - 1 || l.type == YOLO || l.type == REGION || l.type == DETECTION) && output) || interim || weight)
                {
                    os << "Layer: " << i << " : " << ToString(l.type) << " ." << std::endl;
                    if (l.type == CONVOLUTIONAL && weight)
                    {
                        Synet::Tensor<float> weight({ (size_t)l.out_c, (size_t)l.c, (size_t)l.size, (size_t)l.size });
                        memcpy(weight.CpuData(), l.weights, weight.Size() * sizeof(float));
                        weight.DebugPrint(os, String("weight"), true, first, last, precision);
                        if (l.batch_normalize)
                        {
                            Synet::Tensor<float> mean({ (size_t)l.out_c });
                            memcpy(mean.CpuData(), l.rolling_mean, mean.Size() * sizeof(float));
                            mean.DebugPrint(os, String("mean"), true, first, last, precision);

                            Synet::Tensor<float> variance({ (size_t)l.out_c });
                            memcpy(variance.CpuData(), l.rolling_variance, variance.Size() * sizeof(float));
                            variance.DebugPrint(os, String("variance"), true, first, last, precision);

                            Synet::Tensor<float> scale({ (size_t)l.out_c });
                            memcpy(scale.CpuData(), l.scales, scale.Size() * sizeof(float));
                            scale.DebugPrint(os, String("scale"), true, first, last, precision);
                        }
                        Synet::Tensor<float> bias({ (size_t)l.out_c });
                        memcpy(bias.CpuData(), l.biases, bias.Size() * sizeof(float));
                        bias.DebugPrint(os, String("bias"), true, first, last, precision);
                    }
                    if (((i == _net->n - 1 || l.type == YOLO || l.type == REGION || l.type == DETECTION) && output) || interim)
                    {
                        Synet::Tensor<float> dst;
                        OutputToTensor(l, dst);
                        dst.DebugPrint(os, String("dst[0]"), false, first, last, precision);
                    }
                }
            }
        }

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
            _output.clear();
            for (size_t i = 0; i < _net->n; ++i)
            {
                const layer& l = _net->layers[i];
                if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
                    AddToOutput(l);
            }
            if (_output.empty())
                AddToOutput(_net->layers[_net->n - 1]);
        }

        void AddToOutput(const layer& l)
        {
            _output.push_back(Tensor());
            OutputToTensor(l, _output.back());
        }

        String ToString(LAYER_TYPE type) const
        {
            switch (type)
            {
            case CONVOLUTIONAL: return "Convolution";
            case DECONVOLUTIONAL: return "Deconvolution";
            case CONNECTED: return "Connected";
            case MAXPOOL: return "Maxpool";
            case SOFTMAX: return "Softmax";
            case DETECTION: return "Detection";
            case DROPOUT: return "Dropout";
            case CROP: return "Crop";
            case ROUTE: return "Route";
            case COST: return "Cost";
            case NORMALIZATION: return "Normalization";
            case AVGPOOL: return "Avgpool";
            case LOCAL: return "Local";
            case SHORTCUT: return "Shortcut";
            case ACTIVE: return "Active";
            case RNN: return "Rnn";
            case GRU: return "Gru";
            case CRNN: return "Crnn";
            case BATCHNORM: return "Batchnorm";
            case NETWORK: return "Network";
            case XNOR: return "Xnor";
            case REGION: return "Region";
            case YOLO: return "Yolo";
            case REORG: return "Reorg";
            case UPSAMPLE: return "Upsample";
            case BLANK: return "Blank";
            default:
                return "";
            }
        }

        void OutputToTensor(const layer& l, Tensor& t)
        {
            if (l.type == ::SOFTMAX)
                t.Reshape(Shp(l.batch, l.outputs, 1, 1), Synet::TensorFormatNchw);
            else if (l.type == ::YOLO || l.type == ::REGION || l.type == DETECTION)
                t.Reshape(Shp(l.batch, l.outputs / l.h / l.w, l.h, l.w), Synet::TensorFormatNchw);
            else
                t.Reshape(Shp(l.batch, l.out_c, l.out_h, l.out_w), Synet::TensorFormatNchw);
            memcpy(t.CpuData(), l.output, t.Size() * sizeof(float));
        }

        String PatchCfg(const String & src, size_t batchSize)
        {
            String dst = src + ".patched.txt";
            std::ifstream ifs(src.c_str());
            std::ofstream ofs(dst.c_str());
            String line;
            while(std::getline(ifs, line))
            {
                if (line.substr(0, 6) == "batch=")
                    ofs << "batch=" << batchSize << std::endl;
                else
                    ofs << line << std::endl;
            }
            ifs.close();
            ofs.close();
            return dst;
        }
    };
}
#else //SYNET_FIRST_RUN
namespace Test
{
    struct DarknetNetwork : public Network
    {
    };
}
#endif//SYNET_FIRST_RUN

Test::PerformanceMeasurerStorage Test::PerformanceMeasurerStorage::s_storage;

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    if (options.mode == "convert")
    {
        SYNET_PERF_FUNC();
#ifdef SYNET_TEST_FIRST_RUN
        std::cout << "Convert network from Darkent to Synet :" << std::endl;
        options.result = Synet::ConvertDarknetToSynet(options.firstModel, options.firstWeight, options.tensorFormat == 1, options.secondModel, options.secondWeight);
        std::cout << "Conversion is finished " << (options.result ? "successfully." : "with errors.") << std::endl;
#else
        std::cout << "Conversion of Darkent to Synet is not available!" << std::endl;
        options.result = false;
#endif
    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::DarknetNetwork, Test::SynetNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}