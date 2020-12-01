/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"
#include "Synet/Utils/FileUtils.h"

#if defined(SYNET_DARKNET_ENABLE)

#ifdef SYNET_DARKNET_PATH
#define SYNET_INCLUDE_DARKNET(file) SYNET_INCLUDE(SYNET_DARKNET_PATH, file)
#else
#error darknet library path is not defined!
#endif

#ifdef _MSC_VER
#define _TIMESPEC_DEFINED
#pragma warning (push)
#pragma warning (disable: 4244 4305)
#endif

#ifdef SYNET_DARKNET_CUSTOM
#include SYNET_INCLUDE_DARKNET(network.h)
extern "C" 
{
#include SYNET_INCLUDE_DARKNET(region_layer.h)
#include SYNET_INCLUDE_DARKNET(utils.h)
#include SYNET_INCLUDE_DARKNET(parser.h)
#include SYNET_INCLUDE_DARKNET(box.h)
#include SYNET_INCLUDE_DARKNET(stb_image.h)
}
#else
#include SYNET_INCLUDE_DARKNET(darknet.h)
#endif

#ifdef _MSC_VER
#pragma warning (pop)
#endif

namespace Synet
{
    class DarknetToSynet
    {
    public:
        bool Convert(const String & srcModelPath, const String & srcWeightPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
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

            Synet::NetworkParamHolder holder;
            Vector weight;
#ifdef SYNET_DARKNET_CUSTOM
            ::network net = ::parse_network_cfg((char*)srcModelPath.c_str());
            ::load_weights(&net, (char*)srcWeightPath.c_str());
            if (!ConvertNetwork(net, trans, holder(), weight))
#else
            ::network * net = ::parse_network_cfg((char*)srcModelPath.c_str());
            ::load_weights(net, (char*)srcWeightPath.c_str());
            if (!ConvertNetwork(*net, trans, holder(), weight))
#endif
            {
                ::free_network(net);
                return false;
            }
            ::free_network(net);

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
                return false;

            if (!param.Save(dstModelPath, false))
            {
                std::cout << "Can't save Synet model '" << dstModelPath << "' !" << std::endl;
                return false;
            }

            if (!SaveBinaryData(weight, dstWeightPath))
            {
                std::cout << "Can't save Synet weight '" << dstWeightPath << "' !" << std::endl;
                return false;
            }

            return true;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<float> Vector;

        bool ConvertNetwork(const ::network & net, bool trans, Synet::NetworkParam & network, Vector & weight)
        {
            _id = 0;
            _dst.clear();
            _dst.reserve(net.n);

            network.version() = 1;
            network.name() = String("darknet_unknown");
            network.layers().reserve(net.n*3);
            weight.resize(WeightSize(net));

            if (!ConvertInputLayer(net, trans, network.layers()))
                return false;

            for (size_t i = 0, offset = 0; i < (size_t)net.n; ++i)
            {
                if (!ConvertLayer(net.layers[i], trans, network.layers(), weight, offset))
                    return false;
                _dst.push_back(network.layers().back().dst()[0]);
            }
            return true;
        }

        bool ConvertInputLayer(const ::network & src, bool trans, LayerParams & dst)
        {
            Synet::LayerParam input;
            input.type() = Synet::LayerTypeInput;
            input.name() = UniqueName("Input");
            input.dst().resize(1, input.name());
            input.input().shape().resize(1);
            if (trans)
            {
                input.input().shape()[0].dim() = Shape({ (size_t)src.batch, (size_t)src.h, (size_t)src.w, (size_t)src.c });
                input.input().shape()[0].format() = Synet::TensorFormatNhwc;
            }
            else
                input.input().shape()[0].dim() = Shape({ (size_t)src.batch, (size_t)src.c, (size_t)src.h, (size_t)src.w });
            dst.push_back(input);
            return true;
        }

        bool ConvertLayer(const ::layer & src, bool trans, LayerParams & dst, Vector & weight, size_t & offset)
        {
            switch (src.type)
            {
            case ::AVGPOOL:
                if (!ConvertAveragePoolingLayer(src, dst))
                    return false;
                break;
            case ::CONVOLUTIONAL:
                if (!ConvertConvolitionLayer(src, trans, dst, weight, offset))
                    return false;
                if (src.batch_normalize)
                {
                    if (!ConvertBatchNormLayer(src, dst, weight, offset))
                        return false;
                    if (!ConvertScaleLayer(src, dst, weight, offset))
                        return false;
                }
                else
                {
                    if (!ConvertBiasLayer(src, dst, weight, offset))
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
                if (!ConvertRegionLayer(src, trans, dst))
                    return false;
                break;
            case ::REORG:
                if (!ConvertReorgLayer(src, dst))
                    return false;
                break;
            case ::ROUTE:
                if (!ConvertConcatLayer(src, trans, dst))
                    return false;
                break;
            case ::SHORTCUT:
                if (!ConvertShortcutLayer(src, dst))
                    return false;
                if (!ConvertActivationLayer(src, dst))
                    return false;
                break;
            case ::SOFTMAX:
                if (!ConvertSoftmaxLayer(src, trans, dst))
                    return false;
                break;
            case ::UPSAMPLE:
                if (!ConvertUpsampleLayer(src, dst))
                    return false;
                break;
            case ::YOLO:
                if (!ConvertYoloLayer(src, trans, dst))
                    return false;
                break;
            default:
                assert(0);
                return false;
            }
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

        bool ConvertAveragePoolingLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam pooling;
            pooling.type() = Synet::LayerTypePooling;
            pooling.name() = UniqueName("AvgPool");
            pooling.src() = dst.back().dst();
            pooling.dst().resize(1, pooling.name());
            pooling.pooling().method() = Synet::PoolingMethodTypeAverage;
            pooling.pooling().globalPooling() = true;
            dst.push_back(pooling);
            return true;
        }

        bool ConvertBatchNormLayer(const ::layer & src, LayerParams & dst, Vector & weight, size_t & offset)
        {
            Synet::LayerParam batchNorm;
            batchNorm.type() = Synet::LayerTypeBatchNorm;
            batchNorm.name() = UniqueName("BatchNorm");
            batchNorm.src() = dst.back().dst();
            batchNorm.dst() = dst.back().dst();
            batchNorm.batchNorm().eps() = 0.000001f;
            batchNorm.batchNorm().yoloCompatible() = true;
            batchNorm.weight().resize(2);
            if (!ConvertWeight(src.rolling_mean, Shape({ (size_t)src.out_c }), false, batchNorm.weight()[0], weight, offset))
                return false;
            if (!ConvertWeight(src.rolling_variance, Shape({ (size_t)src.out_c }), false, batchNorm.weight()[1], weight, offset))
                return false;
            dst.push_back(batchNorm);
            return true;
        }

        bool ConvertBiasLayer(const ::layer & src, LayerParams & dst, Vector & weight, size_t & offset)
        {
            Synet::LayerParam bias;
            bias.type() = Synet::LayerTypeBias;
            bias.name() = UniqueName("Bias");
            bias.src() = dst.back().dst();
            bias.dst() = dst.back().dst();
            bias.weight().resize(1);
            if (!ConvertWeight(src.biases, Shape({ (size_t)src.out_c}), false, bias.weight()[0], weight, offset))
                return false;
            dst.push_back(bias);
            return true;
        }

        bool ConvertConcatLayer(const ::layer & src, bool trans, LayerParams & dst)
        {
            Synet::LayerParam concat;
            concat.type() = Synet::LayerTypeConcat;
            concat.name() = UniqueName("Concat");
            for (int i = 0; i < src.n; ++i)
                concat.src().push_back(_dst[src.input_layers[i]]);
            concat.dst().resize(1, concat.name());
            concat.concat().axis() = trans ? 3 : 1;
            dst.push_back(concat);
            return true;
        }

        bool ConvertConvolitionLayer(const ::layer & src, bool trans, LayerParams & dst, Vector & weight, size_t & offset)
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
            if (!ConvertWeight(src.weights, Shape({ (size_t)src.out_c, (size_t)src.c, (size_t)src.size, (size_t)src.size }), 
                trans, convolution.weight()[0], weight, offset))
                return false;
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
#ifdef SYNET_DARKNET_CUSTOM
            if (src.pad != 0)
                pooling.pooling().pad().resize(1, src.pad);
            pooling.pooling().yoloCompatible() = 1;
#else
            if (src.pad != 0)
            {
                pooling.pooling().pad().resize(4, 0);
                pooling.pooling().pad()[2] = src.pad;
                pooling.pooling().pad()[3] = src.pad;
            }
            pooling.pooling().yoloCompatible() = 2;
#endif
            dst.push_back(pooling);
            return true;
        }

        void ToNchwPermuteLayer(LayerParams & dst)
        {
            Synet::LayerParam permute;
            permute.type() = Synet::LayerTypePermute;
            permute.name() = UniqueName("Permute");
            permute.src() = dst.back().dst();
            permute.dst().resize(1, permute.name());
            permute.permute().order() = Shape({ 0, 3, 1, 2 });
            permute.permute().format() = TensorFormatNchw;
            dst.push_back(permute);
        }

        bool ConvertRegionLayer(const ::layer & src, bool trans, LayerParams & dst)
        {
            if (trans)
                ToNchwPermuteLayer(dst);
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

        bool ConvertScaleLayer(const ::layer & src, LayerParams & dst, Vector & weight, size_t & offset)
        {
            Synet::LayerParam scale;
            scale.type() = Synet::LayerTypeScale;
            scale.name() = UniqueName("Scale");
            scale.src() = dst.back().dst();
            scale.dst() = dst.back().dst();
            scale.scale().biasTerm() = true;
            scale.weight().resize(2);
            if (!ConvertWeight(src.scales, Shape({ (size_t)src.out_c }), false, scale.weight()[0], weight, offset))
                return false;
            if (!ConvertWeight(src.biases, Shape({ (size_t)src.out_c }), false, scale.weight()[1], weight, offset))
                return false;
            dst.push_back(scale);
            return true;
        }

        bool ConvertShortcutLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam shortcut;
            shortcut.type() = Synet::LayerTypeShortcut;
            shortcut.name() = UniqueName("Shortcut");
            shortcut.src().push_back(_dst.back());
            shortcut.src().push_back(_dst[src.index]);
            shortcut.dst().resize(1, shortcut.name());
            dst.push_back(shortcut);
            return true;
        }

        bool ConvertSoftmaxLayer(const ::layer & src, bool trans, LayerParams & dst)
        {
            Synet::LayerParam softmax;
            softmax.type() = Synet::LayerTypeSoftmax;
            softmax.name() = UniqueName("Softmax");
            softmax.src().push_back(_dst.back());
            softmax.dst().push_back(softmax.name());
            softmax.softmax().axis() = trans ? 3 : 1;
            dst.push_back(softmax);
            return true;
        }

        bool ConvertUpsampleLayer(const ::layer & src, LayerParams & dst)
        {
            Synet::LayerParam upsample;
            upsample.type() = Synet::LayerTypeUpsample;
            upsample.name() = UniqueName("Upsample");
            upsample.src() = dst.back().dst();
            upsample.dst().resize(1, upsample.name());
            upsample.upsample().stride() = src.stride;
            upsample.upsample().scale() = src.scale;
            dst.push_back(upsample);
            return true;
        }

        bool ConvertYoloLayer(const ::layer & src, bool trans, LayerParams & dst)
        {
            if (trans)
                ToNchwPermuteLayer(dst);
            Synet::LayerParam yolo;
            yolo.type() = Synet::LayerTypeYolo;
            yolo.name() = UniqueName("Yolo");
            yolo.src() = dst.back().dst();
            yolo.dst().resize(1, yolo.name());
            yolo.yolo().classes() = src.classes;
            yolo.yolo().num() = src.n;
            yolo.yolo().total() = src.total;
            yolo.yolo().max() = src.max_boxes;
            yolo.yolo().jitter() = src.jitter;
            yolo.yolo().ignoreThresh() = src.ignore_thresh;
            yolo.yolo().truthThresh() = src.truth_thresh;
            yolo.yolo().mask().resize(src.n);
            for (size_t i = 0; i < yolo.yolo().mask().size(); ++i)
                yolo.yolo().mask()[i] = src.mask[i];
            yolo.yolo().anchors().resize(src.total * 2);
            for (size_t i = 0; i < yolo.yolo().anchors().size(); ++i)
                yolo.yolo().anchors()[i] = src.biases[i];
            dst.push_back(yolo);
            return true;
        }

        bool ConvertWeight(const float * src, Shape shape, bool trans, WeightParam & param, Vector & weight, size_t & offset)
        {
            size_t size = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                size *= shape[i];
            if (offset + size > weight.size())
            {
                std::cout << "Can't convert weight: buffer overflow!" << std::endl;
                return false;
            }
            if (shape.size() == 4 && trans)
            {
                shape = Shape({shape[2], shape[3], shape[1], shape[0]});
                param.format() = TensorFormatNhwc;
                Tensor dst(weight.data() + offset, size, shape, param.format());
                for (size_t o = 0; o < shape[3]; ++o)
                    for (size_t i = 0; i < shape[2]; ++i)
                        for (size_t y = 0; y < shape[0]; ++y)
                            for (size_t x = 0; x < shape[1]; ++x)
                                dst.CpuData(Shape({ y, x, i, o }))[0] = *src++;
            }
            else
                memcpy(weight.data() + offset, src, size * sizeof(float));
            param.dim() = shape;
            param.offset() = offset*sizeof(float);
            param.size() = size * sizeof(float);
            offset += size;
            return true;
        }

        size_t WeightSize(const ::network & net) const
        {
            size_t size = 0;
            for (size_t i = 0; i < net.n; ++i)
            {
                const ::layer & l = net.layers[i];
                size += l.size*l.size*l.c*l.out_c;
                if (l.batch_normalize)
                    size += l.out_c*4;
                else
                    size += l.out_c;
            }
            return size;
        }

        String UniqueName(const String & prefix)
        {
            return prefix + "_" + Synet::ValueToString<size_t>(_id++);
        }

        size_t _id;
        Strings _dst;
    };

    bool ConvertDarknetToSynet(const String & srcData, const String & srcWeights, bool trans, const String & dstXml, const String & dstBin)
    {
        DarknetToSynet darknetToSynet;
        return darknetToSynet.Convert(srcData, srcWeights, trans, dstXml, dstBin);
    }
}

#endif