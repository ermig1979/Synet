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

#define CPU_ONLY

#include "caffe/caffe.hpp"

namespace Synet
{
    class CaffeToSynet
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

            caffe::NetParameter srcModel;
            if(!(caffe::ReadProtoFromTextFile(srcModelPath, &srcModel) || caffe::ReadProtoFromBinaryFile(srcModelPath, &srcModel)))
                return false;

            Synet::NetworkParamHolder holder;
            if (!ConvertNetwork(srcModel, holder()))
                return false;

            caffe::NetParameter srcWeight;
            if (!caffe::ReadProtoFromBinaryFile(srcWeightPath, &srcWeight))
                return false;

            Tensors weight;
            weight.reserve(holder().layers().size() * 2);
            if (!ConvertWeight(srcWeight, holder(), weight))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(weight, dstWeightPath))
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
            case Synet::LayerTypeBatchNorm: 
                ConvertBatchNorm(src.batch_norm_param(), dst.batchNorm()); 
                break;
            case Synet::LayerTypeConcat:
                if (src.concat_param().has_concat_dim())
                    dst.concat().axis() = src.concat_param().concat_dim();
                else
                    dst.concat().axis() = src.concat_param().axis();
                break;
            case Synet::LayerTypeConvolution: 
                ConvertConvolution(src.convolution_param(), dst.convolution()); 
                break;
            case Synet::LayerTypeEltwise:
                dst.eltwise().operation() = (EltwiseOperationType)src.eltwise_param().operation();
                dst.eltwise().coefficients().resize(src.eltwise_param().coeff_size());
                for (int j = 0; j < src.eltwise_param().coeff_size(); ++j)
                    dst.eltwise().coefficients()[j] = src.eltwise_param().coeff(j);
                break;
            case Synet::LayerTypeFlatten:
                dst.flatten().axis() = src.flatten_param().axis();
                dst.flatten().endAxis() = src.flatten_param().end_axis();
                break;
            case Synet::LayerTypeDetectionOutput:
                dst.detectionOutput().numClasses() = src.detection_output_param().num_classes();
                dst.detectionOutput().shareLocation() = src.detection_output_param().share_location();
                dst.detectionOutput().backgroundLabelId() = src.detection_output_param().background_label_id();
                dst.detectionOutput().nms().nmsThreshold() = src.detection_output_param().nms_param().nms_threshold();
                dst.detectionOutput().nms().topK() = src.detection_output_param().nms_param().top_k();
                dst.detectionOutput().nms().eta() = src.detection_output_param().nms_param().eta();
                dst.detectionOutput().codeType() = (PriorBoxCodeType)src.detection_output_param().code_type();
                dst.detectionOutput().varianceEncodedInTarget() = src.detection_output_param().variance_encoded_in_target();
                dst.detectionOutput().keepTopK() = src.detection_output_param().keep_top_k();
                dst.detectionOutput().confidenceThreshold() = src.detection_output_param().confidence_threshold();
                dst.detectionOutput().keepMaxClassScoresOnly() = src.detection_output_param().keep_max_class_scores_only();
                break;
            case Synet::LayerTypeDropout:
                break;
            case Synet::LayerTypeInnerProduct:
                dst.innerProduct().outputNum() = src.inner_product_param().num_output();
                dst.innerProduct().biasTerm() = src.inner_product_param().bias_term();
                dst.innerProduct().transposeB() = src.inner_product_param().transpose();
                dst.innerProduct().axis() = src.inner_product_param().axis();
                break;
            case Synet::LayerTypeInput:
                dst.input().shape().resize(src.input_param().shape_size());
                for (int j = 0; j < src.top_size(); ++j)
                {
                    dst.input().shape()[j].dim().resize(src.input_param().shape(j).dim_size());
                    for (int k = 0; k < src.input_param().shape(j).dim_size(); ++k)
                        dst.input().shape()[j].dim()[k] = src.input_param().shape(j).dim(k);
                }
                break;
            case Synet::LayerTypeInterp:
                ConvertInterp(src.interp_param(), dst.interp());
                break;
            case Synet::LayerTypeLog:
                dst.log().base() = src.log_param().base();
                dst.log().scale() = src.log_param().scale();
                dst.log().shift() = src.log_param().shift();
                break;
            case Synet::LayerTypeLrn:
                dst.lrn().localSize() = src.lrn_param().local_size();
                dst.lrn().alpha() = src.lrn_param().alpha();
                dst.lrn().beta() = src.lrn_param().beta();
                dst.lrn().normRegion() = (Synet::NormRegionType)src.lrn_param().norm_region();
                dst.lrn().k() = src.lrn_param().k();
                break;
            case Synet::LayerTypeNormalize:
                dst.normalize().acrossSpatial() = src.norm_param().across_spatial();
                dst.normalize().channelShared() = src.norm_param().channel_shared();
                dst.normalize().eps() = src.norm_param().eps();
                break;
            case Synet::LayerTypePermute:
                dst.permute().order().resize(src.permute_param().order_size());
                for (int i = 0; i < src.permute_param().order_size(); ++i)
                    dst.permute().order()[i] = src.permute_param().order(i);
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
            case Synet::LayerTypePriorBox:
                dst.priorBox().minSize().resize(src.prior_box_param().min_size_size());
                for (int i = 0; i < src.prior_box_param().min_size_size(); ++i)
                    dst.priorBox().minSize()[i] = src.prior_box_param().min_size(i);
                dst.priorBox().maxSize().resize(src.prior_box_param().max_size_size());
                for (int i = 0; i < src.prior_box_param().max_size_size(); ++i)
                    dst.priorBox().maxSize()[i] = src.prior_box_param().max_size(i);
                dst.priorBox().aspectRatio().resize(src.prior_box_param().aspect_ratio_size());
                for (int i = 0; i < src.prior_box_param().aspect_ratio_size(); ++i)
                    dst.priorBox().aspectRatio()[i] = src.prior_box_param().aspect_ratio(i);
                dst.priorBox().flip() = src.prior_box_param().flip();
                dst.priorBox().clip() = src.prior_box_param().clip();
                dst.priorBox().variance().resize(src.prior_box_param().variance_size());
                for (int i = 0; i < src.prior_box_param().variance_size(); ++i)
                    dst.priorBox().variance()[i] = src.prior_box_param().variance(i);
                if (src.prior_box_param().has_img_size())
                    dst.priorBox().imgSize() = Shape({ src.prior_box_param().img_size() });
                if (src.prior_box_param().has_img_h() && src.prior_box_param().has_img_w())
                    dst.priorBox().imgSize() = Shape({ src.prior_box_param().img_h(), src.prior_box_param().img_w() });
                if (src.prior_box_param().has_step())
                    dst.priorBox().step() = Floats({ src.prior_box_param().step() });
                if (src.prior_box_param().has_step_h() && src.prior_box_param().has_step_w())
                    dst.priorBox().step() = Floats({ src.prior_box_param().step_h(), src.prior_box_param().step_w() });
                dst.priorBox().offset() = src.prior_box_param().offset();
                break;
            case Synet::LayerTypeRelu:
                dst.relu().negativeSlope() = src.relu_param().negative_slope();
                break;
            case Synet::LayerTypeReshape:
                dst.reshape().shape().resize(src.reshape_param().shape().dim_size());
                for (int i = 0; i < src.reshape_param().shape().dim_size(); ++i)
                    dst.reshape().shape()[i] = src.reshape_param().shape().dim(i);
                dst.reshape().axis() = src.reshape_param().axis();
                dst.reshape().numAxes() = src.reshape_param().num_axes();
                break;
            case Synet::LayerTypeScale:
                dst.scale().axis() = src.scale_param().axis();
                dst.scale().numAxes() = src.scale_param().num_axes();
                dst.scale().biasTerm() = src.scale_param().bias_term();
                break;
            case Synet::LayerTypeSigmoid:
                break;
            case Synet::LayerTypeSlice:
                if (src.slice_param().has_slice_dim())
                    dst.slice().axis() = src.slice_param().slice_dim();
                else
                    dst.slice().axis() = src.slice_param().axis();
                dst.slice().slicePoint().resize(src.slice_param().slice_point_size());
                for (int j = 0; j < src.slice_param().slice_point_size(); ++j)
                    dst.slice().slicePoint()[j] = src.slice_param().slice_point(j);
                break;
            case Synet::LayerTypeSoftmax:
                dst.softmax().axis() = src.softmax_param().axis();
                break;
            default:
                assert(0);
                break;
            }
            return true;
        }

        void ConvertBatchNorm(const caffe::BatchNormParameter & src, Synet::BatchNormParam & dst)
        {
            if (src.has_use_global_stats())
                dst.useGlobalStats() = src.use_global_stats();
            dst.movingAverageFraction() = src.moving_average_fraction();
            dst.eps() = src.eps();
        }

        void ConvertConvolution(const caffe::ConvolutionParameter & src, Synet::ConvolutionParam & dst)
        {
            dst.outputNum() = src.num_output();
            dst.biasTerm() = src.bias_term();
            dst.axis() = src.axis();
            dst.group() = src.group();
            dst.kernel().resize(src.kernel_size_size());
            for (int j = 0; j < src.kernel_size_size(); ++j)
                dst.kernel()[j] = src.kernel_size(j);
            dst.pad().resize(src.pad_size());
            for (int j = 0; j < src.pad_size(); ++j)
                dst.pad()[j] = src.pad(j);
            dst.stride().resize(src.stride_size());
            for (int j = 0; j < src.stride_size(); ++j)
                dst.stride()[j] = src.stride(j);
            dst.dilation().resize(src.dilation_size());
            for (int j = 0; j < src.dilation_size(); ++j)
                dst.dilation()[j] = src.dilation(j);
        }

        void ConvertInterp(const caffe::InterpParameter & src, Synet::InterpParam & dst)
        {
            dst.height() = src.height();
            dst.width() = src.width();
            dst.zoomFactor() = src.zoom_factor();
            dst.shrinkFactor() = src.shrink_factor();
            dst.cropBeg() = -src.pad_beg();
            dst.cropEnd() = -src.pad_end();
            dst.useTensorSize() = src.use_blob_size();
        }

        bool ConvertWeight(const caffe::NetParameter & src, Synet::NetworkParam & dst, Tensors & weight)
        {
            for (int i = 0; i < src.layer_size(); ++i)
            {
                for (int l = 0; l < dst.layers().size(); ++l)
                {
                    if (src.layer(i).name() == dst.layers()[l].name())
                    {
                        dst.layers()[l].weight().resize(src.layer(i).blobs_size());
                        for (int j = 0; j < src.layer(i).blobs_size(); ++j)
                        {
                            dst.layers()[l].weight()[j].dim().resize(src.layer(i).blobs(j).shape().dim_size());
                            for (int k = 0; k < src.layer(i).blobs(j).shape().dim_size(); ++k)
                                dst.layers()[l].weight()[j].dim()[k] = src.layer(i).blobs(j).shape().dim(k);
                            weight.push_back(Tensor());
                            weight.back().Reshape(dst.layers()[l].weight()[j].dim(), 0);
                            for (int k = 0; k < src.layer(i).blobs(j).data_size(); ++k)
                                weight.back().CpuData()[k] = src.layer(i).blobs(j).data(k);
                        }
                    }
                }
            }
            return true;
        }

        bool SaveWeight(const Tensors & weight, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (ofs.is_open())
            {
                for (size_t i = 0; i < weight.size(); ++i)
                {
                    ofs.write((const char*)weight[i].CpuData(), weight[i].Size()*sizeof(float));
                }
                ofs.close();
                return true;
            }
            return false;
        }
    };

    bool ConvertCaffeToSynet(const String & srcModel, const String & srcWeight, const String & dstXml, const String & dstBin)
    {
        CaffeToSynet caffeToSynet;
        return caffeToSynet.Convert(srcModel, srcWeight, dstXml, dstBin);
    }
}

#endif