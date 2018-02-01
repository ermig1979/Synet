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
#include <io.h>
#endif

#define CPU_ONLY

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

            Synet::NetworkParamHolder holder;
            if (!ConvertNetwork(srcModel, holder()))
                return false;

            caffe::NetParameter srcWeight;
            if (!caffe::ReadProtoFromBinaryFile(srcWeightPath, &srcWeight))
                return false;

            Tensors data;
            data.reserve(holder().layers().size() * 2);
            if (!ConvertWeight(srcWeight, holder(), data))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(data, dstWeightPath))
                return false;

            return true;
        }

    private:

        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;

        bool ConvertNetwork(const caffe::NetParameter & src, Synet::NetworkParam & dst)
        {
            dst.name() = src.name();
            dst.layers().reserve(src.layers_size());
            for (int i = 0; i < src.layer_size(); ++i)
            {
                Synet::LayerParam dstLayer;
                if(ConvertLayer(src.layer(i), dstLayer))
                    dst.layers().push_back(dstLayer);
            }
            return true;
        }

        bool ConvertLayer(const caffe::LayerParameter & src, Synet::LayerParam & dst)
        {
            for(int i = 0; i < src.exclude_size(); ++i)
            {
                if (src.exclude(i).has_phase() && src.exclude(i).phase() == caffe::TEST)
                    return false;
            }

            dst.name() = src.name();
            Synet::StringToValue<LayerType>(src.type(), dst.type());
            for (int j = 0; j < src.bottom_size(); ++j)
                dst.src().push_back(src.bottom(j));
            for (int j = 0; j < src.top_size(); ++j)
                dst.dst().push_back(src.top(j));
            switch (dst.type())
            {
            case Synet::LayerTypeInput:
                dst.input().shape().resize(src.input_param().shape_size());
                for (int j = 0; j < src.top_size(); ++j)
                {
                    dst.input().shape()[j].dim().resize(src.input_param().shape(j).dim_size());
                    for (int k = 0; k < src.input_param().shape(j).dim_size(); ++k)
                        dst.input().shape()[j].dim()[k] = src.input_param().shape(j).dim(k);
                }
                break;
            case Synet::LayerTypeInnerProduct:
                dst.innerProduct().outputNum() = src.inner_product_param().num_output();
                dst.innerProduct().biasTerm() = src.inner_product_param().bias_term();
                dst.innerProduct().transpose() = src.inner_product_param().transpose();
                dst.innerProduct().axis() = src.inner_product_param().axis();
                break;
            case Synet::LayerTypeRelu:
                dst.relu().negativeSlope() = src.relu_param().negative_slope();
                break;
            case Synet::LayerTypeSigmoid:
                break;
            case Synet::LayerTypePooling:
                dst.pooling().method() = (Synet::PoolingMethodType)src.pooling_param().pool();
                dst.pooling().globalPooling() = src.pooling_param().global_pooling();
                if (src.pooling_param().has_kernel_size())
                    dst.pooling().kernel() = Shape({ src.pooling_param().kernel_size() });
                if (src.pooling_param().has_kernel_h() && src.pooling_param().has_kernel_w())
                    dst.pooling().kernel() = Shape({ src.pooling_param().kernel_h(), src.pooling_param().kernel_w() });
                if (src.pooling_param().has_pad())
                    dst.pooling().pad() = Shape({ src.pooling_param().pad() });
                if (src.pooling_param().has_pad_h() && src.pooling_param().has_pad_w())
                    dst.pooling().pad() = Shape({ src.pooling_param().pad_h(), src.pooling_param().pad_w() });
                if (src.pooling_param().has_stride())
                    dst.pooling().stride() = Shape({ src.pooling_param().stride() });
                if (src.pooling_param().has_stride_h() && src.pooling_param().has_stride_w())
                    dst.pooling().stride() = Shape({ src.pooling_param().stride_h(), src.pooling_param().stride_w() });
                break;
            case Synet::LayerTypeConvolution:
                dst.convolution().outputNum() = src.convolution_param().num_output();
                dst.convolution().biasTerm() = src.convolution_param().bias_term();
                dst.convolution().axis() = src.convolution_param().axis();
                dst.convolution().group() = src.convolution_param().group();
                dst.convolution().kernel().resize(src.convolution_param().kernel_size_size());
                for (int j = 0; j < src.convolution_param().kernel_size_size(); ++j)
                    dst.convolution().kernel()[j] = src.convolution_param().kernel_size(j);
                dst.convolution().pad().resize(src.convolution_param().pad_size());
                for (int j = 0; j < src.convolution_param().pad_size(); ++j)
                    dst.convolution().pad()[j] = src.convolution_param().pad(j);
                dst.convolution().stride().resize(src.convolution_param().stride_size());
                for (int j = 0; j < src.convolution_param().stride_size(); ++j)
                    dst.convolution().stride()[j] = src.convolution_param().stride(j);
                dst.convolution().dilation().resize(src.convolution_param().dilation_size());
                for (int j = 0; j < src.convolution_param().dilation_size(); ++j)
                    dst.convolution().dilation()[j] = src.convolution_param().dilation(j);
                break;
            case Synet::LayerTypeLrn:
                dst.lrn().localSize() = src.lrn_param().local_size();
                dst.lrn().alpha() = src.lrn_param().alpha();
                dst.lrn().beta() = src.lrn_param().beta();
                dst.lrn().normRegion() = (Synet::NormRegionType)src.lrn_param().norm_region();
                dst.lrn().k() = src.lrn_param().k();
                break;
            case Synet::LayerTypeConcat:
                if (src.concat_param().has_concat_dim())
                    dst.concat().axis() = src.concat_param().concat_dim();
                else
                    dst.concat().axis() = src.concat_param().axis();
                break;
            case Synet::LayerTypeDropout:
                break;
            default:
                assert(0);
                break;
            }
            return true;
        }

        bool ConvertWeight(const caffe::NetParameter & src, Synet::NetworkParam & dst, Tensors & data)
        {
            for (int i = 0; i < src.layer_size(); ++i)
            {
                for (int l = 0; l < dst.layers().size(); ++l)
                {
                    if (src.layer(i).name() == dst.layers()[l].name())
                    {
                        dst.layers()[l].data().resize(src.layer(i).blobs_size());
                        for (int j = 0; j < src.layer(i).blobs_size(); ++j)
                        {
                            dst.layers()[l].data()[j].dim().resize(src.layer(i).blobs(j).shape().dim_size());
                            for (int k = 0; k < src.layer(i).blobs(j).shape().dim_size(); ++k)
                                dst.layers()[l].data()[j].dim()[k] = src.layer(i).blobs(j).shape().dim(k);
                            data.push_back(Tensor());
                            data.back().Reshape(dst.layers()[l].data()[j].dim(), 0);
                            for (int k = 0; k < src.layer(i).blobs(j).data_size(); ++k)
                                data.back().Data()[k] = src.layer(i).blobs(j).data(k);
                        }
                    }
                }
            }
            return true;
        }

        bool SaveWeight(const Tensors & data, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (ofs.is_open())
            {
                for (size_t i = 0; i < data.size(); ++i)
                {
                    ofs.write((const char*)data[i].Data(), data[i].Size()*sizeof(float));
                }
                ofs.close();
                return true;
            }
            return false;
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