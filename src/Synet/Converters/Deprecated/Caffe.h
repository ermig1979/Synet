/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Converters/Optimizer.h"

#if defined(SYNET_CAFFE_ENABLE)

#if !defined(SYNET_LEGACY_2020_ENABLE)
#error The support of Caffe to Synet model conversion is stopped!
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
#endif

#define CPU_ONLY
#include "caffe/caffe.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace Synet
{
    class CaffeToSynet
    {
    public:
        bool Convert(const String & srcModelPath, const String & srcWeightPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Cpl::FileExists(srcModelPath))
            {
                std::cout << "File '" << srcModelPath << "' is not exist!" << std::endl;
                return false;
            }

            if (!Cpl::FileExists(srcWeightPath))
            {
                std::cout << "File '" << srcWeightPath << "' is not exist!" << std::endl;
                return false;
            }

            caffe::NetParameter srcModel;
            if(!(caffe::ReadProtoFromTextFile(srcModelPath, &srcModel) || caffe::ReadProtoFromBinaryFile(srcModelPath, &srcModel)))
                return false;

            caffe::NetParameter srcWeight;
            if (!caffe::ReadProtoFromBinaryFile(srcWeightPath, &srcWeight))
                return false;

            Synet::NetworkParamHolder holder;
            if (!ConvertNetwork(srcModel, trans, holder()))
                return false;

            Vector weight;
            if (!ConvertWeight(srcWeight, trans, holder(), weight))
                return false;

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
                return false;

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
        typedef std::vector<float> Vector;

        bool ConvertNetwork(const caffe::NetParameter & src, bool trans, Synet::NetworkParam & dst)
        {
            dst.version() = 1;
            dst.name() = src.name();
            dst.layers().reserve(src.layers_size());
            for (int i = 0; i < src.layer_size(); ++i)
            {
                Synet::LayerParam dstLayer;
                if(ConvertLayer(src.layer(i), trans, dstLayer, dst.layers()))
                    dst.layers().push_back(dstLayer);
            }
            return true;
        }

        bool ConvertLayer(const caffe::LayerParameter & src, bool trans, Synet::LayerParam & dst, const LayerParams & layers)
        {
            for(int i = 0; i < src.exclude_size(); ++i)
            {
                if (src.exclude(i).has_phase() && src.exclude(i).phase() == caffe::TEST)
                    return false;
            }
            if (src.type() == "Dropout")
                return false;

            dst.name() = src.name();
            Cpl::ToVal<LayerType>(src.type(), dst.type());
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
                ConvertConcat(src.concat_param(), trans, dst.concat(), layers);
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
            case Synet::LayerTypeInnerProduct:
                ConvertInnerProduct(src.inner_product_param(), trans, dst.innerProduct());
                break;
            case Synet::LayerTypeInput:
                ConvertInput(src.input_param(), trans, dst.input());
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
                ConvertPermute(src.permute_param(), trans, dst.permute());
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

        bool PermutedToNchw(const LayerParams & layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                if (layers[i].type() == LayerTypePermute && layers[i].permute().format() == TensorFormatNchw)
                    return true;
            }
            return false;
        }

        void ConvertBatchNorm(const caffe::BatchNormParameter & src, Synet::BatchNormParam & dst)
        {
            if (src.has_use_global_stats())
                dst.useGlobalStats() = src.use_global_stats();
            dst.movingAverageFraction() = src.moving_average_fraction();
            dst.eps() = src.eps();
        }

        void ConvertConcat(const caffe::ConcatParameter & src, bool trans, Synet::ConcatParam & dst, const LayerParams & layers)
        {
            if (src.has_concat_dim())
                dst.axis() = src.concat_dim();
            else
                dst.axis() = src.axis();
            if (trans && dst.axis() == 1 && !PermutedToNchw(layers))
                dst.axis() = 3;
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

        void ConvertInput(const caffe::InputParameter & src, bool trans, Synet::InputParam & dst)
        {
            dst.shape().resize(src.shape_size());
            for (int j = 0; j < src.shape_size(); ++j)
            {
                Shape shape(src.shape(j).dim_size());
                for (int k = 0; k < src.shape(j).dim_size(); ++k)
                    shape[k] = src.shape(j).dim(k);
                if (trans && shape.size() == 4)
                {
                    shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                    dst.shape()[j].format() = TensorFormatNhwc;
                }
                dst.shape()[j].dim() = shape;
            }
        }

        void ConvertInnerProduct(const caffe::InnerProductParameter & src, bool trans, Synet::InnerProductParam & dst)
        {
            dst.outputNum() = src.num_output();
            dst.biasTerm() = src.bias_term();
            dst.transposeB() = src.transpose();
            dst.axis() = src.axis();
            //if (trans && dst.axis() == 1)
            //    dst.axis() = 3;
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

        void ConvertPermute(const caffe::PermuteParameter & src, bool trans, Synet::PermuteParam & dst)
        {
            Shape order(src.order_size());
            for (int i = 0; i < src.order_size(); ++i)
                order[i] = src.order(i);
            if (trans && order.size() == 4)
            {
                if (order == Shape({ 0, 2, 3, 1 }))
                {
                    order = Shape({ 0, 1, 2, 3 });
                    dst.format() = TensorFormatNchw;
                }
            }
            dst.order() = order;
        }

        bool ConvertTransInnerProductWeight(const LayerParams & layers, size_t current, const caffe::BlobProto & blob, Tensor & tensor)
        {
            const LayerParam & layer = layers[current];
            if (layer.type() != LayerTypeInnerProduct || layer.src().size() != 1 || blob.shape().dim_size() != 2)
                return false;
            ptrdiff_t curr = current, prev;
            while (curr)
            {
                for (prev = curr - 1; prev >= 0; --prev)
                    if (layers[prev].name() == layers[curr].src()[0])
                        break;
                if (prev < 0 || layers[prev].type() == LayerTypeInnerProduct || layers[prev].type() == LayerTypePermute || layers[prev].type() == LayerTypeConcat)
                    return false;
                if (layers[prev].type() == LayerTypeConvolution)
                {
                    Shape shape;
                    shape.push_back(blob.shape().dim(0));
                    shape.push_back(blob.shape().dim(1));
                    size_t channel = layers[prev].convolution().outputNum();
                    size_t spatial = shape[1] / channel;
                    tensor.Reshape(shape);
                    for (size_t d = 0; d < shape[0]; d++)
                    {
                        for (size_t c = 0; c < channel; c++)
                        {
                            for (size_t s = 0; s < spatial; s++)
                            {
                                size_t srcOffset = d*shape[1] + spatial * c + s;
                                size_t dstOffset = d*shape[1] + channel * s + c;
                                tensor.CpuData()[dstOffset] = blob.data(srcOffset);
                            }
                        }
                    }
                    return true;
                }
                curr = prev;
            }
            return false;
        }

        bool ConvertWeight(const caffe::NetParameter & src, bool trans, Synet::NetworkParam & dst, Vector & weight)
        {
            size_t offset = 0;
            for (int i = 0; i < src.layer_size(); ++i)
            {
                for (int l = 0; l < dst.layers().size(); ++l)
                {
                    if (src.layer(i).name() == dst.layers()[l].name())
                    {
                        const caffe::LayerParameter & srcLayer = src.layer(i);
                        Synet::LayerParam & dstLayer = dst.layers()[l];
                        dstLayer.weight().resize(srcLayer.blobs_size());
                        for (int j = 0; j < srcLayer.blobs_size(); ++j)
                        {
                            const caffe::BlobProto & blob = srcLayer.blobs(j);
                            Synet::WeightParam & param = dstLayer.weight()[j];
                            Shape shape(blob.shape().dim_size());
                            for (int k = 0; k < blob.shape().dim_size(); ++k)
                                shape[k] = blob.shape().dim(k);
                            Tensor tensor;
                            if (trans && shape.size() == 4)
                            {
                                shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
                                tensor.Reshape(shape);
                                for (size_t d = 0, o = 0; d < shape[3]; ++d)
                                    for (size_t c = 0; c < shape[2]; ++c)
                                        for (size_t y = 0; y < shape[0]; ++y)
                                            for (size_t x = 0; x < shape[1]; ++x)
                                                tensor.CpuData(Shape({ y, x, c, d }))[0] = blob.data((int)(o++));
                                param.format() = TensorFormatNhwc;
                            }
                            else if (trans && ConvertTransInnerProductWeight(dst.layers(), l, blob, tensor))
                            {
                                shape = tensor.Shape();
                            }
                            else
                            {
                                tensor.Reshape(shape);
                                for (int k = 0; k < blob.data_size(); ++k)
                                    tensor.CpuData()[k] = blob.data(k);
                            }
                            param.dim() = shape;
                            param.offset() = offset * sizeof(float);
                            param.size() = tensor.Size() * sizeof(float);
                            if (offset + tensor.Size() > weight.size())
                                weight.resize(offset + tensor.Size());
                            memcpy(weight.data() + offset, tensor.CpuData(), param.size());
                            offset += tensor.Size();
                        }
                    }
                }
            }
            return true;
        }

        bool SaveWeight(const Vector & weight, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (!ofs.is_open())
                return false;
            ofs.write((const char*)weight.data(), weight.size() * sizeof(float));
            bool result = (bool)ofs;
            ofs.close();
            return result;
        }
    };

    bool ConvertCaffeToSynet(const String & srcModel, const String & srcWeight, bool trans, const String & dstXml, const String & dstBin)
    {
        CaffeToSynet caffeToSynet;
        return caffeToSynet.Convert(srcModel, srcWeight, trans, dstXml, dstBin);
    }
}

#endif