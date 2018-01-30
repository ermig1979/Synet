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

#if defined(SYNET_CAFFE_ENABLE)

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4996)
#endif

#include "caffe/caffe.hpp"

namespace Synet
{
    class CaffeToSynet
    {
    public:
        bool Convert(const String & srcModelPath, const String & srcWeightPath, const String & dstModelPath, const String & dstWeightPath)
        {
            if (::_access(srcModelPath.c_str(), 0) == -1)
                return false;
            if (::_access(srcWeightPath.c_str(), 0) == -1)
                return false;

            caffe::NetParameter srcModel;
            if (!caffe::ReadProtoFromBinaryFile(srcModelPath, &srcModel))
                return false;

            Synet::NetworkConfig dstModel;
            dstModel().name() = srcModel.name();
            dstModel().layers().reserve(srcModel.layers_size());
            for (int i = 0; i < srcModel.layer_size(); ++i)
            {
                const caffe::LayerParameter & srcLayer = srcModel.layer(i);
                Synet::LayerParam dstLayer;
                dstLayer.name() = srcLayer.name();
                Synet::StringToValue<LayerType>(srcLayer.type(), dstLayer.type());
                for (int j = 0; j < srcLayer.bottom_size(); ++j)
                    dstLayer.src().push_back(srcLayer.bottom(j));
                for (int j = 0; j < srcLayer.top_size(); ++j)
                    dstLayer.dst().push_back(srcLayer.top(j));
                switch (dstLayer.type())
                {
                case Synet::LayerTypeInput:
                    dstLayer.inputLayer().shape().resize(srcLayer.input_param().shape_size());
                    for (int j = 0; j < srcLayer.top_size(); ++j)
                    {
                        dstLayer.inputLayer().shape()[j].dim().resize(srcLayer.input_param().shape(j).dim_size());
                        for (int k = 0; k < srcLayer.input_param().shape(j).dim_size(); ++k)
                            dstLayer.inputLayer().shape()[j].dim()[k] = srcLayer.input_param().shape(j).dim(k);
                    }
                    break;
                case Synet::LayerTypeInnerProduct:
                    dstLayer.innerProductLayer().outputNum() = srcLayer.inner_product_param().num_output();
                    dstLayer.innerProductLayer().biasTerm() = srcLayer.inner_product_param().bias_term();
                    dstLayer.innerProductLayer().transpose() = srcLayer.inner_product_param().transpose();
                    dstLayer.innerProductLayer().axis() = srcLayer.inner_product_param().axis();
                    break;
                case Synet::LayerTypeRelu:
                    dstLayer.reluLayer().negativeSlope() = srcLayer.relu_param().negative_slope();
                    break;
                case Synet::LayerTypeSigmoid:
                    break;
                case Synet::LayerTypePooling:
                {
                    dstLayer.poolingLayer().method() = (Synet::PoolingMethodType)srcLayer.pooling_param().pool();
                    dstLayer.poolingLayer().globalPooling() = srcLayer.pooling_param().global_pooling();
                    if (srcLayer.pooling_param().has_kernel_size())
                        dstLayer.poolingLayer().kernel() = Shape({ srcLayer.pooling_param().kernel_size() });
                    if (srcLayer.pooling_param().has_kernel_h() && srcLayer.pooling_param().has_kernel_w())
                        dstLayer.poolingLayer().kernel() = Shape({ srcLayer.pooling_param().kernel_h(), srcLayer.pooling_param().kernel_w() });
                    if (srcLayer.pooling_param().has_pad())
                        dstLayer.poolingLayer().pad() = Shape({ srcLayer.pooling_param().pad() });
                    if (srcLayer.pooling_param().has_pad_h() && srcLayer.pooling_param().has_pad_w())
                        dstLayer.poolingLayer().pad() = Shape({ srcLayer.pooling_param().pad_h(), srcLayer.pooling_param().pad_w() });
                    if (srcLayer.pooling_param().has_stride())
                        dstLayer.poolingLayer().stride() = Shape({ srcLayer.pooling_param().stride() });
                    if (srcLayer.pooling_param().has_stride_h() && srcLayer.pooling_param().has_stride_w())
                        dstLayer.poolingLayer().stride() = Shape({ srcLayer.pooling_param().stride_h(), srcLayer.pooling_param().stride_w() });
                    break;
                }
                case Synet::LayerTypeConvolution:
                {
                    dstLayer.convolutionLayer().outputNum() = srcLayer.convolution_param().num_output();
                    dstLayer.convolutionLayer().biasTerm() = srcLayer.convolution_param().bias_term();
                    dstLayer.convolutionLayer().axis() = srcLayer.convolution_param().axis();
                    dstLayer.convolutionLayer().group() = srcLayer.convolution_param().group();
                    dstLayer.convolutionLayer().kernel().resize(srcLayer.convolution_param().kernel_size_size());
                    for (int j = 0; j < srcLayer.convolution_param().kernel_size_size(); ++j)
                        dstLayer.convolutionLayer().kernel()[j] = srcLayer.convolution_param().kernel_size(j);
                    dstLayer.convolutionLayer().pad().resize(srcLayer.convolution_param().pad_size());
                    for (int j = 0; j < srcLayer.convolution_param().pad_size(); ++j)
                        dstLayer.convolutionLayer().pad()[j] = srcLayer.convolution_param().pad(j);
                    dstLayer.convolutionLayer().stride().resize(srcLayer.convolution_param().stride_size());
                    for (int j = 0; j < srcLayer.convolution_param().stride_size(); ++j)
                        dstLayer.convolutionLayer().stride()[j] = srcLayer.convolution_param().stride(j);
                    dstLayer.convolutionLayer().dilation().resize(srcLayer.convolution_param().dilation_size());
                    for (int j = 0; j < srcLayer.convolution_param().dilation_size(); ++j)
                        dstLayer.convolutionLayer().dilation()[j] = srcLayer.convolution_param().dilation(j);
                    break;
                }
                case Synet::LayerTypeLrn:
                    dstLayer.lrnLayer().localSize() = srcLayer.lrn_param().local_size();
                    dstLayer.lrnLayer().alpha() = srcLayer.lrn_param().alpha();
                    dstLayer.lrnLayer().beta() = srcLayer.lrn_param().beta();
                    dstLayer.lrnLayer().normRegion() = (Synet::NormRegionType)srcLayer.lrn_param().norm_region();
                    dstLayer.lrnLayer().k() = srcLayer.lrn_param().k();
                    break;
                case Synet::LayerTypeConcat:
                    break;
                case Synet::LayerTypeDropout:
                    break;
                default:
                    assert(0);
                    break;
                }
                dstModel().layers().push_back(dstLayer);

            }
            if (!dstModel.Save(dstModelPath, false))
                return false;

            return true;
        }
    };

    bool ConvertCaffeToSynet(const String & srcData, const String & srcWeights, const String & dstXml, const String & dstBin)
    {
        CaffeToSynet caffeToSynet;
        return caffeToSynet.Convert(srcData, srcWeights, dstXml, dstBin);
    }
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif