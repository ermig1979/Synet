/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#pragma once

#include "Synet/Common.h"
#include "Synet/Params.h"

#if defined(SYNET_YOLO_ENABLE)

#ifdef SYNET_YOLO_PATH
#define SYNET_INCLUDE_YOLO(file) SYNET_INCLUDE(SYNET_YOLO_PATH, file)
#else
#error yolo library path is not defined!
#endif

#ifdef _MSC_VER
#define _TIMESPEC_DEFINED
#pragma warning (push)
#pragma warning (disable: 4244 4305)
#endif

#include SYNET_INCLUDE_YOLO(network.h)
extern "C" 
{
#include SYNET_INCLUDE_YOLO(region_layer.h)
#include SYNET_INCLUDE_YOLO(utils.h)
#include SYNET_INCLUDE_YOLO(parser.h)
#include SYNET_INCLUDE_YOLO(box.h)
#include SYNET_INCLUDE_YOLO(stb_image.h)
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif

namespace Synet
{
    class YoloToSynet
    {
    public:
        bool Convert(const String & srcModelPath, const String & srcWeightPath, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcModelPath))
            {
                std::cout << "File '" << srcModelPath << "' is not exist!" << std::endl;
                return false;
            }

            if (!Synet::FileExist(srcWeightPath))
            {
                std::cout << "File '" << srcWeightPath << "' is not exist!" << std::endl;
                return false;
            }

            ::network net = ::parse_network_cfg((char*)srcModelPath.c_str());

            ::load_weights(&net, (char*)srcWeightPath.c_str());

            Synet::NetworkParamHolder holder;
            Tensors weight;
            if (!ConvertNetwork(net, holder(), weight))
            {
                ::free_network(net);
                return false;
            }

            ::free_network(net);

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(weight, dstWeightPath))
                return false;

            return true;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;

        bool ConvertNetwork(const ::network & net, Synet::NetworkParam & network, Tensors & weight)
        {
            _id = 0;
            _dst.clear();
            _dst.reserve(net.n);

            network.layers().reserve(net.n*3);
            weight.reserve(net.n * 2);
            network.name() = String("yolo_unknown");

            if (!ConvertInputLayer(net, network.layers()))
                return false;

            for (int i = 0; i < net.n; ++i)
            {
                if (!ConvertLayer(net.layers[i], network.layers(), weight))
                    return false;
                _dst.push_back(network.layers().back().dst()[0]);
            }
            return true;
        }

        bool ConvertInputLayer(const ::network & src, LayerParams & dst)
        {
            Synet::LayerParam input;
            input.type() = Synet::LayerTypeInput;
            input.name() = UniqueName("Input");
            input.dst().resize(1, input.name());
            input.input().shape().resize(1);
            input.input().shape()[0].dim() = Shape({ (size_t)src.batch, (size_t)src.c, (size_t)src.h, (size_t)src.w });
            dst.push_back(input);
            return true;
        }

        bool ConvertLayer(const ::layer & src, LayerParams & dst, Tensors & weight)
        {
            switch (src.type)
            {
            case ::CONVOLUTIONAL:
                if (!ConvertConvolitionLayer(src, dst, weight))
                    return false;
                if (src.batch_normalize)
                {
                    if (!ConvertBatchNormLayer(src, dst, weight))
                        return false;
                    if (!ConvertScaleLayer(src, dst, weight))
                        return false;
                }
                else
                {
                    if (!ConvertBiasLayer(src, dst, weight))
                        return false;
                }
                if (!ConvertActivationLayer(src, dst))
                    return false;
                break;
            case ::MAXPOOL:
                if (!ConvertMaxPoolingLayer(src, dst))
                    return false;
                break;
            case ::REGION:
                if (!ConvertRegionLayer(src, dst))
                    return false;
                break;
            case ::REORG:
                if (!ConvertReorgLayer(src, dst))
                    return false;
                break;
            case ::ROUTE:
                if (!ConvertConcatLayer(src, dst))
                    return false;
                break;
            default:
                assert(0);
                return false;
            }
            return true;
        }

        bool ConvertBatchNormLayer(const ::layer & src, LayerParams & dst, Tensors & weight)
        {
            Synet::LayerParam batchNorm;
            batchNorm.type() = Synet::LayerTypeBatchNorm;
            batchNorm.name() = UniqueName("BatchNorm");
            batchNorm.src() = dst.back().dst();
            batchNorm.dst() = dst.back().dst();
            batchNorm.batchNorm().eps() = 0.000001f;
            batchNorm.weight().resize(2);
            batchNorm.weight()[0].dim() = Shape({ (size_t)src.out_c });
            batchNorm.weight()[1].dim() = Shape({ (size_t)src.out_c });
            weight.push_back(Tensor());
            weight.back().Reshape(batchNorm.weight()[0].dim());
            memcpy(weight.back().Data(), src.rolling_mean, weight.back().Size() * sizeof(float));
            weight.push_back(Tensor());
            weight.back().Reshape(batchNorm.weight()[1].dim());
            memcpy(weight.back().Data(), src.rolling_variance, weight.back().Size() * sizeof(float));
            dst.push_back(batchNorm);
            return true;
        }

        bool ConvertBiasLayer(const ::layer & src, LayerParams & dst, Tensors & weight)
        {
            Synet::LayerParam bias;
            bias.type() = Synet::LayerTypeBias;
            bias.name() = UniqueName("Bias");
            bias.src() = dst.back().dst();
            bias.dst() = dst.back().dst();
            bias.weight().resize(1);
            bias.weight()[0].dim() = Shape({ (size_t)src.out_c});
            weight.push_back(Tensor());
            weight.back().Reshape(bias.weight()[0].dim());
            memcpy(weight.back().Data(), src.biases, weight.back().Size() * sizeof(float));
            dst.push_back(bias);
            return true;
        }

        bool ConvertConcatLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam concat;
            concat.type() = Synet::LayerTypeConcat;
            concat.name() = UniqueName("Concat");
            for (int i = 0; i < src.n; ++i)
                concat.src().push_back(_dst[src.input_layers[i]]);
            concat.dst().resize(1, concat.name());
            dst.push_back(concat);
            return true;
        }

        bool ConvertConvolitionLayer(const ::layer & src, LayerParams & dst, Tensors & weight)
        {
            Synet::LayerParam convolution;
            convolution.type() = Synet::LayerTypeConvolution;
            convolution.name() = UniqueName("Conv");
            convolution.src() = dst.back().dst();
            convolution.dst().resize(1, convolution.name());
            convolution.convolution().outputNum() = src.out_c;
            convolution.convolution().kernel().resize(1, src.size);
            if (src.stride != 1)
                convolution.convolution().stride().resize(1, src.stride);
            if (src.pad != 0)
                convolution.convolution().pad().resize(1, src.pad);
            convolution.convolution().biasTerm() = false;
            convolution.weight().resize(1);
            convolution.weight()[0].dim() = Shape({ (size_t)src.out_c, (size_t)src.c, (size_t)src.size, (size_t)src.size });
            weight.push_back(Tensor());
            weight.back().Reshape(convolution.weight()[0].dim());
            memcpy(weight.back().Data(), src.weights, weight.back().Size() * sizeof(float));
            dst.push_back(convolution);
            return true;
        }

        bool ConvertMaxPoolingLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam pooling;
            pooling.type() = Synet::LayerTypePooling;
            pooling.name() = UniqueName("MaxPool");
            pooling.src() = dst.back().dst();
            pooling.dst().resize(1, pooling.name());
            pooling.pooling().method() = Synet::PoolingMethodTypeMax;
            pooling.pooling().kernel().resize(1, src.size);
            if (src.stride != 1)
                pooling.pooling().stride().resize(1, src.stride);
            if (src.pad != 0)
                pooling.pooling().pad().resize(1, src.pad);
            pooling.pooling().yoloCompatible() = true;
            dst.push_back(pooling);
            return true;
        }

        bool ConvertRegionLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam region;
            region.type() = Synet::LayerTypeRegion;
            region.name() = UniqueName("Region");
            region.src() = dst.back().dst();
            region.dst().resize(1, region.name());
            region.region().coords() = src.coords;
            region.region().classes() = src.classes;
            region.region().num() = src.n;
            region.region().softmax() = src.softmax != 0;
            region.region().anchors().resize(src.n * 2);
            for (size_t i = 0; i < region.region().anchors().size(); ++i)
                region.region().anchors()[i] = src.biases[i];
            dst.push_back(region);
            return true;
        }

        bool ConvertReorgLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam reorg;
            reorg.type() = Synet::LayerTypeReorg;
            reorg.name() = UniqueName("Reorg");
            reorg.reorg().reverse() = (src.reverse != 0);
            reorg.reorg().stride() = src.stride;
            reorg.src() = dst.back().dst();
            reorg.dst().resize(1, reorg.name());
            dst.push_back(reorg);
            return true;
        }

        bool ConvertScaleLayer(const ::layer & src, LayerParams & dst, Tensors & weight)
        {
            Synet::LayerParam scale;
            scale.type() = Synet::LayerTypeScale;
            scale.name() = UniqueName("Scale");
            scale.src() = dst.back().dst();
            scale.dst() = dst.back().dst();
            scale.scale().biasTerm() = true;
            scale.weight().resize(2);
            scale.weight()[0].dim() = Shape({ (size_t)src.out_c });
            scale.weight()[1].dim() = Shape({ (size_t)src.out_c });
            weight.push_back(Tensor());
            weight.back().Reshape(scale.weight()[0].dim());
            memcpy(weight.back().Data(), src.scales, weight.back().Size() * sizeof(float));            
            weight.push_back(Tensor());
            weight.back().Reshape(scale.weight()[1].dim());
            memcpy(weight.back().Data(), src.biases, weight.back().Size() * sizeof(float));
            dst.push_back(scale);
            return true;
        }

        bool ConvertActivationLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam activation;
            activation.src() = dst.back().dst();
            activation.dst() = dst.back().dst();
            switch (src.activation)
            {
            case ::LEAKY:
                activation.type() = Synet::LayerTypeRelu;
                activation.name() = UniqueName("ReLU");
                activation.relu().negativeSlope() = 0.1f;
                break;
            case ::LINEAR:
                return true;
            case ::LOGISTIC:
                activation.type() = Synet::LayerTypeSigmoid;
                activation.name() = UniqueName("Sigmoid");
                break;
            default:
                assert(0);
                return false;
            }
            dst.push_back(activation);
            return true;
        }

        bool SaveWeight(const Tensors & weight, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (ofs.is_open())
            {
                for (size_t i = 0; i < weight.size(); ++i)
                {
                    ofs.write((const char*)weight[i].Data(), weight[i].Size()*sizeof(float));
                }
                ofs.close();
                return true;
            }
            return false;
        }

        String UniqueName(const String & prefix)
        {
            return prefix + "_" + Synet::ValueToString<size_t>(_id++);
        }

        size_t _id;
        Strings _dst;
    };

    bool ConvertYoloToSynet(const String & srcData, const String & srcWeights, const String & dstXml, const String & dstBin)
    {
        YoloToSynet yoloToSynet;
        return yoloToSynet.Convert(srcData, srcWeights, dstXml, dstBin);
    }
}

#endif