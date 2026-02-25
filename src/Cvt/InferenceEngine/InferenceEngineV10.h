/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Cvt/InferenceEngine/InferenceEngineBase.h"

namespace Synet
{
    class InferenceEngineConverterV10 : public InferenceEngineConverter
    {
    public:
        bool Convert(const XmlNode& srcXml, const Bytes& srcBin, bool trans, const OnnxParam &onnxParam, Synet::NetworkParam & dstXml, Bytes& dstBin)
        {
            Edges edges;
            if (!ParseEdges(srcXml, edges))
                return false;

            IndexMap index;
            TensorInfoMap info;

            const XmlNode* pLayers = srcXml.FirstNode("layers");
            if (pLayers == NULL)
                return false;
            const XmlNode* pLayer = pLayers->FirstNode("layer"), * pPrevLayer = NULL, * pNextLayer = NULL;
            while (pLayer)
            {
                pNextLayer = pLayer->NextSibling("layer");

                LayerParams children;
                LayerParam layer;
                if (!ParseInputOutput(*pLayer, edges, dstXml.layers(), layer, index, info))
                    return false;

                String type = pLayer->FirstAttribute("type")->Value();
                if (type == "Add" && !ConvertAddLayer(pLayer, dstXml.layers(), srcBin, layer))
                    return ErrorMessage(pLayer);
                if(type == "AvgPool" && !ConvertAvgPoolLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Broadcast" && !ConvertBroadcastLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Clamp" && !ConvertClampLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Concat" && !ConvertConcatLayer(pLayer, dstXml.layers(), trans, layer, index))
                    return ErrorMessage(pLayer);
                if (type == "Const" && !ConvertConstLayer(pLayer, srcBin, layer))
                    return ErrorMessage(pLayer);
                if (type == "Convert" && !ConvertConvertLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "Convolution" || type == "GroupConvolution") && !ConvertConvolutionLayer(pLayer, trans, dstXml.layers(), srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "CTCGreedyDecoder" && !ConvertCtcGreedyDecoderLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "DetectionOutput" && !ConvertDetectionOutputLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Divide" && !ConvertDivideLayer(pLayer, dstXml.layers(), srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Equal" && !ConvertEqualLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Exp" && !ConvertExpLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Floor" && !ConvertFloorLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Gather" && !ConvertGatherLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Interpolate" && !ConvertInterpolateLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Log" && !ConvertLogLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if ((type == "MatMul") && !ConvertMatMulLayer(pLayer, trans, dstXml.layers(), srcBin, layer, dstBin, info))
                    return ErrorMessage(pLayer);
                if (type == "MaxPool" && !ConvertMaxPoolLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Mish" && !ConvertMishLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Multiply" && !ConvertMultiplyLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "MVN") && !ConvertMvnLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "NonMaxSuppression") && !ConvertNonMaxSuppressionLayer(pLayer, dstXml.layers(), srcBin, layer))
                    return ErrorMessage(pLayer);
                if (type == "NonZero" && !ConvertNonZeroLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Parameter" && !ConvertParameterLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Power" && !ConvertPowerLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "PReLU" && !ConvertPreluLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBox" && !ConvertPriorBoxLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxClustered" && !ConvertPriorBoxClusteredLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxV2" && !ConvertPriorBoxV2Layer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Range" && !ConvertRangeLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceMean" && !ConvertReduceMeanLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceMin" && !ConvertReduceMinLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceProd" && !ConvertReduceProdLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "ReduceMax" || type == "ReduceSum") && !ConvertReduceMaxOrSumLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "RegionYolo" && !ConvertRegionYoloLayer(pLayer, dstXml.layers(), trans, layer, index))
                    return ErrorMessage(pLayer);
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Reshape" && !ConvertReshapeLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Result" && !ConvertResultLayer(pLayer, layer, &dstXml))
                    return ErrorMessage(pLayer);
                if (type == "ReverseSequence" && !ConvertReverseSequenceLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Select" && !ConvertSelectLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "ShapeOf" && !ConvertShapeOfLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Sigmoid" && !ConvertSigmoidLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if ((type == "SoftMax" || type == "LogSoftmax") && !ConvertSoftmaxOrLogSoftmaxLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Split" && !ConvertSplitLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Sqrt" && !ConvertSqrtLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Squeeze" && !ConvertSqueezeLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "StridedSlice" && !ConvertStridedSliceLayer(pLayer, dstXml.layers(), srcBin, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Subtract" && !ConvertSubtractLayer(pLayer, dstXml.layers(), srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Swish" && !ConvertSwishLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Tanh" && !ConvertTanhLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if ((type == "TensorIterator") && !ConvertTensorIteratorLayer(pLayer, trans, dstXml.layers(), srcBin, layer, dstBin, info, children))
                    return ErrorMessage(pLayer);
                if (type == "Tile" && !ConvertTileLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "TopK" && !ConvertTopKLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Transpose" && !ConvertTransposeLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "VariadicSplit" && !ConvertVariadicSplitLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Unsqueeze" && !ConvertUnsqueezeLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
#if defined(SYNET_IE_PARSE_STOP_ON_ERROR)
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(pLayer);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(pLayer, layer);
                    std::cout << "Not implemented layer : name = " << layer.name() << " ; type = " << type << " ; id = " << pLayer->FirstAttribute("id")->Value() << std::endl;
                }
#endif
                dstXml.layers().push_back(layer);
                for(size_t c = 0; c < children.size(); ++c)
                    dstXml.layers().push_back(children[c]);
                pPrevLayer = pLayer;
                pLayer = pNextLayer;

                if (trans && !ManualInsertToNchwPermute(onnxParam, dstXml.layers()))
                    return false;
            }

            if (!RemoveUnusedConst(dstXml.layers()))
                return false;

            return true;
        }

    private:

        bool ConvertAddLayer(const XmlNode* pLayer, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape src0 = ConvertInputShape(pLayer, "0");
            Shape src1 = ConvertInputShape(pLayer, "1");
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (first == NULL || second == NULL)
                return false;
            if (first->type() == LayerTypeMeta && (second->type() == LayerTypeMeta || second->type() == LayerTypeConst))
            {
                if (second->type() == LayerTypeConst)
                {
                    LayerParam* change = (LayerParam*)second;
                    change->type() = Synet::LayerTypeMeta;
                    change->meta().type() = Synet::MetaTypeConst;
                    change->meta().alpha().type() = TensorType32f;
                    change->meta().alpha().shape() = second->weight()[0].dim();
                    size_t size = TensorSize(second->weight()[0].dim());
                    change->meta().alpha().f32().resize(size);
                    const float* src = GetWeight<float>(srcBin, second->weight()[0].offset());
                    for (size_t i = 0; i < size; ++i)
                        change->meta().alpha().f32()[i] = src[i];
                    change->weight().clear();
                }
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeAdd;
            }
            else if (second->type() == LayerTypeConst && TensorSize(src0) >= TensorSize(src1))
            {
                if (TensorSize(src1) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    const float * pShift = GetWeight<float>(srcBin, second->weight()[0]);
                    layer.power().shift() = pShift[0];
                }
                else
                {
                    layer.type() = Synet::LayerTypeBias;
                    layer.weight() = second->weight();
                    if (!CompactShape(layer.weight()[0].dim()))
                        return false;
                }
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeSum;
                if (TensorSize(src0) < TensorSize(src1))
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertAvgPoolLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            if (!ConvertVector(pData->FirstAttribute("kernel"), layer.pooling().kernel()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("strides"), layer.pooling().stride()))
                return false;
            if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.pooling().pad()))
                return false;
            const XmlAttr* pAutoPad = pData->FirstAttribute("auto_pad");
            if (pAutoPad && String(pAutoPad->Value()) == "valid")
                layer.pooling().roundingType() = RoundingTypeFloor;
            return true;
        }

        bool ConvertBroadcastLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst &&
                src1->meta().alpha().type() == TensorType64i && AllEqualTo(src1->meta().alpha().i64(), int64_t(1)))
            {
                layer.type() = Synet::LayerTypeStub;
                layer.src().resize(1);
                return true;
            }
            layer.type() = Synet::LayerTypeBroadcast;
            //if (src0->type() == LayerTypeConst && src1->type() == LayerTypeMeta)
            //    layer.broadcast().fixed() = true;
            return true;
        }

        bool ConvertClampLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            Cpl::ToVal(pData->FirstAttribute("min")->Value(), layer.restrictRange().lower());
            Cpl::ToVal(pData->FirstAttribute("max")->Value(), layer.restrictRange().upper());
            return true;
        }

        bool ConvertConcatLayer(const XmlNode* pLayer, LayerParams& layers, bool trans, LayerParam& layer, IndexMap& index)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == Synet::LayerTypeMeta || (src0->type() == Synet::LayerTypeConst && src0->weight()[0].dim().size() == 1))
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypePack;
            }
            else
            {
                layer.type() = Synet::LayerTypeConcat;
                const XmlNode* pData = pLayer->FirstNode("data");
                if (pData == NULL)
                    return false;
                Cpl::ToVal(pData->FirstAttribute("axis")->Value(), layer.concat().axis());
                if (trans)
                {
                    Ints perm;
                    int count = PermutedToNchw(layers, layer.src(), true, true, false, perm);
                    if (count == 0)
                    {
                        Shape input = ConvertInputShape(pLayer);
                        if (input.size() == 4)
                        {
                            Shape nchw = Shape({ 0, 3, 1, 2 });
                            layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                        }
                        else if (input.size() == 3)
                        {
                            Shape ncs = Shape({ 0, 2, 1});
                            layer.concat().axis() = (uint32_t)ncs[layer.concat().axis()];
                        }
                        else
                            return false;
                    }
                    else if(count < perm.size())
                    {
                        for (size_t i = 0; i < perm.size(); ++i)
                        {
                            Shape input = ConvertInputShape(pLayer, Cpl::ToStr(i));
                            if (perm[i] == 0 && input.size() == 4)
                            {
                                LayerParam permute;
                                permute.type() = LayerTypePermute;
                                permute.src().push_back(layer.src()[i]);
                                permute.name() = layer.src()[i] + "_permute_to_nchw";
                                permute.dst().push_back(permute.name());
                                permute.permute().order() = Shape({ 0, 3, 1, 2});
                                permute.permute().format() = TensorFormatNchw;
                                size_t layerId;
                                Cpl::ToVal(pLayer->FirstAttribute("id")->Value(), layerId);
                                index[layerId]++;
                                layers.push_back(permute);
                                layer.src()[i] = permute.name();
                            }
                        }
                    }
                }
            }
            return true;
        }

        bool ConvertConstLayer(const XmlNode* pLayer, const Bytes & srcBin, LayerParam & layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData)
            {
                String type = pData->FirstAttribute("element_type")->Value();
                Shape shape;
                if(!ConvertVector(pData->FirstAttribute("shape"), shape))
                    return false;
                if (shape == Shp(0))
                    shape[0] = 1;
                else
                {
                    if (shape != ConvertOutputShape(pLayer))
                        return false;
                }
                size_t offset, size;
                Cpl::ToVal(pData->FirstAttribute("offset")->Value(), offset);
                if (type == "f32")
                {
                    layer.type() = Synet::LayerTypeConst;
                    layer.weight().resize(1);
                    layer.weight()[0].type() = TensorType32f;
                    layer.weight()[0].dim() = shape;
                    layer.weight()[0].offset() = offset;
                    Cpl::ToVal(pData->FirstAttribute("size")->Value(), layer.weight()[0].size());
                }
                else if (type == "i32")
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeConst;
                    layer.meta().alpha().type() = TensorType32i;
                    layer.meta().alpha().shape() = shape;
                    size = TensorSize(shape);
                    layer.meta().alpha().i32().resize(size);
                    const int32_t* src = GetWeight<int32_t>(srcBin, offset);
                    for (size_t i = 0; i < size; ++i)
                        layer.meta().alpha().i32()[i] = src[i];
                }
                else if (type == "i64")
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeConst;
                    layer.meta().alpha().type() = TensorType64i;
                    layer.meta().alpha().shape() = shape;
                    size = TensorSize(shape);
                    layer.meta().alpha().i64().resize(size);
                    const int64_t* src = GetWeight<int64_t>(srcBin, offset);
                    for (size_t i = 0; i < size; ++i)
                        layer.meta().alpha().i64()[i] = src[i];
                }
                else if (type == "u64")
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeConst;
                    layer.meta().alpha().type() = TensorType64u;
                    layer.meta().alpha().shape() = shape;
                    size = TensorSize(shape);
                    layer.meta().alpha().u64().resize(size);
                    const uint64_t* src = GetWeight<uint64_t>(srcBin, offset);
                    for (size_t i = 0; i < size; ++i)
                        layer.meta().alpha().u64()[i] = src[i];
                }
                else
                {
                    std::cout << "Unknown element_type = " << type << " !" << std::endl;
                    return false;
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertConvertLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam * source = GetLayer(layers, layer.src()[0]);
            if (source == NULL)
                return false;
            if (source->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeCast;
                const XmlNode* pData = pLayer->FirstNode("data");
                if (pData == NULL)
                    return false;
                String type = pData->FirstAttribute("destination_type")->Value();
                if (type == "i32")
                    layer.meta().alpha().type() = TensorType32i;
                else if (type == "f32")
                    layer.meta().alpha().type() = TensorType32f;
                else if (type == "i64")
                    layer.meta().alpha().type() = TensorType64i;
                else if (type == "u64")
                    layer.meta().alpha().type() = TensorType64u;
                else
                    return false;
            }
            else
            {
                layer.type() = LayerTypeCast;
                const XmlNode* pData = pLayer->FirstNode("data");
                if (pData == NULL)
                    return false;
                String type = pData->FirstAttribute("destination_type")->Value();
                if (type == "i32")
                    layer.cast().type() = TensorType32i;
                else if (type == "f32")
                    layer.cast().type() = TensorType32f;
                else if (type == "i64")
                    layer.cast().type() = TensorType64i;
                else if (type == "u64")
                    layer.cast().type() = TensorType64u;
                else
                    return false;
            }
            return true;
        }

        bool ConvertConvolutionLayer(const XmlNode* pLayer, bool trans, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin)
        {
            layer.type() = Synet::LayerTypeConvolution;
            layer.convolution().biasTerm() = false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (!ConvertVector(pData->FirstAttribute("strides"), layer.convolution().stride()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("dilations"), layer.convolution().dilation()))
                return false;
            if (pData->FirstAttribute("auto_pad"))
            {
                String autoPad = pData->FirstAttribute("auto_pad")->Value();
                if (autoPad == "same_upper")
                    layer.convolution().autoPad() = true;
            }
            if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.convolution().pad()))
                return false;
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            const Shape & shape = second->weight()[0].dim();
            layer.weight() = second->weight();
            if(String(pLayer->FirstAttribute("type")->Value()) == "Convolution")
            { 
                if (!CheckDims(shape, 4, "convolution weight"))
                    return false;
                layer.convolution().kernel() = Shape({ shape[2], shape[3] });
                layer.convolution().outputNum() = (uint32_t)shape[0];
            }
            else
            {
                if (!CheckDims(shape, 5, "convolution weight"))
                    return false;
                layer.convolution().kernel() = Shape({ shape[3], shape[4] });
                layer.convolution().group() = (uint32_t)shape[0];
                layer.convolution().outputNum() = (uint32_t)(shape[0] * shape[1]);
                layer.weight()[0].dim() = Shape({ shape[0] * shape[1] , shape[2], shape[3], shape[4] });
            }
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, false))
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }

        bool ConvertCtcGreedyDecoderLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeCtcGreedyDecoder;
            return true;
        }

        bool ConvertDetectionOutputLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            int opset = GetOpsetVersion(pLayer);
            layer.type() = Synet::LayerTypeDetectionOutput;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String codeType = pData->FirstAttribute("code_type")->Value();
            if (codeType == "caffe.PriorBoxParameter.CENTER_SIZE")
                layer.detectionOutput().codeType() = PriorBoxCodeTypeCenterSize;
            else
                assert(0);
            Cpl::ToVal(pData->FirstAttribute("confidence_threshold")->Value(), layer.detectionOutput().confidenceThreshold());
            Cpl::ToVal(pData->FirstAttribute("keep_top_k")->Value(), layer.detectionOutput().keepTopK());
            Cpl::ToVal(pData->FirstAttribute("nms_threshold")->Value(), layer.detectionOutput().nms().nmsThreshold());
            if (opset < 8)
                Cpl::ToVal(pData->FirstAttribute("num_classes")->Value(), layer.detectionOutput().numClasses());
            Cpl::ToVal(pData->FirstAttribute("variance_encoded_in_target")->Value(), layer.detectionOutput().varianceEncodedInTarget());
            Cpl::ToVal(pData->FirstAttribute("top_k")->Value(), layer.detectionOutput().nms().topK());
            Cpl::ToVal(pData->FirstAttribute("share_location")->Value(), layer.detectionOutput().shareLocation());
            ConvertValue(pData->FirstAttribute("clip"), layer.detectionOutput().clip());
            ConvertValue(pData->FirstAttribute("background_label_id"), layer.detectionOutput().backgroundLabelId());
            return true;
        }

        bool ConvertDivideLayer(const XmlNode* pLayer, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape is0 = ConvertInputShape(pLayer, "0");
            Shape is1 = ConvertInputShape(pLayer, "1");
            const LayerParam* sp0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            if (sp0 == NULL || sp1 == NULL)
                return false;
            if (sp1->type() == LayerTypeConst && (TensorSize(is1) == 1 || TensorSize(is1) == 0))
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(srcBin, sp1->weight()[0]);
                layer.power().scale() = 1.0f / pScale[0];
                layer.src().resize(1);
            }
            else if (sp1->type() == LayerTypeConst && SignificantDimsCount(is1) == 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.weight() = sp1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                const float* pSrc = GetWeight<float>(srcBin, layer.weight()[0]);
                float* pDst = GetWeight<float>(dstBin, layer.weight()[0]);
                size_t size = TensorSize(layer.weight()[0].dim());
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = 1.0f / pSrc[i];
                layer.src().resize(1);
            }
            else if (sp0->type() == LayerTypeMeta && sp1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeDiv;
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeDiv;
            }
            return true;
        }

        bool ConvertEqualLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeEqual;
            return true;
        }

        bool ConvertExpLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeExp;
            return true;
        }  

        bool ConvertFloorLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam * source = GetLayer(layers, layer.src()[0]);
            if (source == NULL)
                return false;
            if (source->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeFloor;
            }
            else
                return false;
            return true;
        }
        
        bool ConvertGatherLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2, 3))
                return false;
            const LayerParam* sp0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            if (sp0 == NULL || sp1 == NULL)
                return false;
            if (sp0->type() == LayerTypeMeta && sp1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeGather;
            }
            else
            {
                layer.type() = LayerTypeGather;
                const XmlNode* pData = pLayer->FirstNode("data");
                if (pData == NULL)
                    return false;
                if (!ConvertValue(pData->FirstAttribute("batch_dims"), layer.gather().batchDims()))
                    return false;
                if (layer.src().size() > 2)
                {
                    const LayerParam* sp2 = GetLayer(layers, layer.src()[2]);
                    if (sp2 == NULL || sp2->type() != LayerTypeMeta || sp2->meta().type() != MetaTypeConst)
                        return false;
                    if (sp2->meta().alpha().type() == TensorType64i)
                        layer.gather().axis() = (int)sp2->meta().alpha().i64()[0];
                    else if (sp2->meta().alpha().type() == TensorType32i)
                        layer.gather().axis() = (int)sp2->meta().alpha().i32()[0];
                    else
                        return false;
                }
            }
            layer.src().resize(2);
            return true;
        }

        bool ConvertInterpolateLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            //if (!CheckSourceNumber(layer, 2))
            //    return false;
            layer.type() = LayerTypeInterp;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr* pMode = pData->FirstAttribute("mode");
            if (pMode)
            {
                String mode = pMode->Value();
                if (mode == "nearest")
                    layer.interp().interpolationType() = InterpolationTypeNearest;
                else if (mode == "linear_onnx")
                    layer.interp().interpolationType() = InterpolationTypeBilinear;
                else
                    return false;
            }
            const XmlAttr* pCoordTranf = pData->FirstAttribute("coordinate_transformation_mode");
            if (pCoordTranf)
            {
                String coordTransf = pCoordTranf->Value();
                if (coordTransf == "pytorch_half_pixel")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypeHalfPixel;
                else if (coordTransf == "asymmetric")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypePytorch;
                else
                    return false;
            }
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
                return false;
            if (second->meta().type() == MetaTypeConst)
            {
                if (second->meta().alpha().shape().size() != 1 || second->meta().alpha().shape()[0] != 2)
                    return false;
                const int64_t* alpha = second->meta().alpha().i64().data();
                layer.interp().height() = (int32_t)alpha[0];
                layer.interp().width() = (int32_t)alpha[1];
            }
            else
            {
                Shape output = ConvertOutputShape(pLayer);
                if (output.size() != 4)
                    return false;
                layer.interp().height() = (int32_t)output[2];
                layer.interp().width() = (int32_t)output[3];
            }
            layer.src().resize(1);
            return true;
        }

        bool ConvertLogLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeLog;
            return true;
        }

        bool ConvertMatMulLayer(const XmlNode* pLayer, bool trans, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorInfoMap& info)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL || pData->FirstAttribute("transpose_a") == NULL || pData->FirstAttribute("transpose_b") == NULL)
                return false;
            bool transposeA, transposeB;
            Cpl::ToVal(pData->FirstAttribute("transpose_a")->Value(), transposeA);
            Cpl::ToVal(pData->FirstAttribute("transpose_b")->Value(), transposeB);
            layer.innerProduct().biasTerm() = false;
            layer.innerProduct().transposeA() = transposeA;
            layer.innerProduct().transposeB() = !transposeB;

            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            Shape input = ConvertInputShape(pLayer);
            if (!CheckDims(input, 2, "inner product input"))
                return false;
            const Shape & weight = second->weight()[0].dim();
            if (!CheckDims(weight, 2, "inner product weight"))
                return false;
            Shape output = ConvertOutputShape(pLayer);
            if (!CheckDims(output, 2, "output product weight"))
                return false;
            layer.weight() = second->weight();
            layer.innerProduct().outputNum() = (uint32_t)output[1];
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, false))
            {
                const LayerParam * first = GetLayer(layers, layer.src()[0]);
                if (first == NULL)
                    return false;
                if (first->type() == LayerTypePooling && first->pooling().globalPooling())
                    return true;
                if (first->type() != LayerTypeReshape)
                    return false;
                Shape origin = info[first->src()[0]].shape;
                return ReorderWeight(srcBin, origin, layer, dstBin);
            }
            return true;
        }

        bool ConvertMaxPoolLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeMax;
            if (!ConvertVector(pData->FirstAttribute("kernel"), layer.pooling().kernel()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("strides"), layer.pooling().stride()))
                return false;
            if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.pooling().pad()))
                return false;
            const XmlAttr * pRoundingType = pData->FirstAttribute("rounding_type");
            if (pRoundingType && String(pRoundingType->Value()) == "floor")
                layer.pooling().roundingType() = RoundingTypeFloor;
            if (layer.dst().size() > 1)
            {
                layer.dst().resize(1);
                layer.dst()[0] = layer.name();
            }
            return true;
        }

        bool ConvertMishLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeMish;
            return true;
        }

        bool ConvertMultiplyLayer(const XmlNode* pLayer, const Bytes& srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (first == NULL || second == NULL)
                return false;
            if (first->type() == LayerTypeMeta && (second->type() == LayerTypeMeta || second->type() == LayerTypeConst))
            {
                if (second->type() == LayerTypeConst)
                {
                    LayerParam* change = (LayerParam*)second;
                    change->type() = Synet::LayerTypeMeta;
                    change->meta().type() = Synet::MetaTypeConst;
                    change->meta().alpha().type() = TensorType32f;
                    change->meta().alpha().shape() = second->weight()[0].dim();
                    size_t size = TensorSize(second->weight()[0].dim());
                    change->meta().alpha().f32().resize(size);
                    const float* src = GetWeight<float>(srcBin, second->weight()[0].offset());
                    for (size_t i = 0; i < size; ++i)
                        change->meta().alpha().f32()[i] = src[i];
                    change->weight().clear();
                }
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeMul;
            }
            else if (first->type() == LayerTypeConst || second->type() == LayerTypeConst)
            {
                if (first->type() == LayerTypeConst)
                {
                    std::swap(layer.src()[0], layer.src()[1]);
                    std::swap(first, second);
                }
                if (TensorSize(second->weight()[0].dim()) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().power() = 1.0f;
                    layer.power().scale() = GetWeight<float>(srcBin, second->weight()[0])[0];
                    layer.power().shift() = 0.0f;
                }
                else
                {
                    layer.type() = Synet::LayerTypeScale;
                    layer.scale().biasTerm() = false;
                    layer.weight() = second->weight();
                    if (!CompactShape(layer.weight()[0].dim()))
                        return false;
                }
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
            }
            return true;
        }

        bool ConvertMvnLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape is0 = ConvertInputShape(pLayer, "0");
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            if (sp1 == NULL || sp1->type() != LayerTypeMeta || sp1->meta().type() != MetaTypeConst || sp1->meta().alpha().type() != TensorType64i)
                return false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypeNormalize;
            if (!ConvertValue(pData->FirstAttribute("eps"), layer.normalize().eps()))
                return false;
            bool normalizeVariance;
            if (!ConvertValue(pData->FirstAttribute("normalize_variance"), normalizeVariance))
                return false;
            if (!normalizeVariance)
                return false;
            String epsMode;
            if (!ConvertValue(pData->FirstAttribute("eps_mode"), epsMode) || epsMode != "INSIDE_SQRT")
                return false;
            if (is0.size() == 3 || is0.size() == 4)
            {
                int axis = (int)sp1->meta().alpha().i64()[0];
                layer.normalize().acrossSpatial() = axis == 2;
            }
            else
                return false;
            layer.normalize().channelShared() = true;
            layer.src().resize(1);
            return true;
        }

        bool ConvertNonMaxSuppressionLayer(const XmlNode* pLayer, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2, 6))
                return false;
            layer.type() = Synet::LayerTypeNonMaxSuppression;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String boxEncoding;
            if (!ConvertValue(pData->FirstAttribute("box_encoding"), boxEncoding))
                return false;
            if (boxEncoding == "corner")
                layer.nonMaxSuppression().boxEncoding() = BoxEncodingTypeCorner;
            else if (boxEncoding == "center")
                layer.nonMaxSuppression().boxEncoding() = BoxEncodingTypeCenter;
            else
                return false;
            if (!ConvertValue(pData->FirstAttribute("sort_result_descending"), layer.nonMaxSuppression().sortResultDescending()))
                return false;
            String outputType;
            if (!ConvertValue(pData->FirstAttribute("output_type"), outputType))
                return false;
            if (outputType == "i64")
                layer.nonMaxSuppression().outputType() = TensorType64i;
            else
                return false;
            if (layer.src().size() > 2)
            {
                const LayerParam * src = GetLayer(layers, layer.src()[2]);
                if (src == NULL || src->type() != LayerTypeMeta || src->meta().type() != MetaTypeConst)
                    return false;
                if (src->meta().alpha().type() == TensorType64i)
                    layer.nonMaxSuppression().maxOutputBoxesPerClass() = (int)src->meta().alpha().i64()[0];
                else
                    return false;
            }
            if (layer.src().size() > 3)
            {
                const LayerParam* src = GetLayer(layers, layer.src()[3]);
                if (src == NULL || src->type() != LayerTypeConst)
                    return false;
                layer.nonMaxSuppression().iouThreshold() = GetWeight<float>(srcBin, src->weight()[0])[0];
            }
            if (layer.src().size() > 4)
            {
                const LayerParam* src = GetLayer(layers, layer.src()[4]);
                if (src == NULL || src->type() != LayerTypeConst)
                    return false;
                layer.nonMaxSuppression().scoreThreshold() = GetWeight<float>(srcBin, src->weight()[0])[0];
            }  
            if (layer.src().size() > 5)
            {
                const LayerParam* src = GetLayer(layers, layer.src()[5]);
                if (src == NULL || src->type() != LayerTypeConst)
                    return false;
                layer.nonMaxSuppression().softNmsSigma() = GetWeight<float>(srcBin, src->weight()[0])[0];
            }
            layer.src().resize(2);
            return true;
        }

        bool ConvertNonZeroLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta || src0->type() == LayerTypeConstantOfShape || src0->type() == LayerTypeBroadcast)
            {
                layer.type() = Synet::LayerTypeNonZero;
            }
            else
            {
                SYNET_ERROR("Unsupported src type!");
            }
            return true;
        }

        bool ConvertParameterLayer(const XmlNode* pLayer, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeInput;
            const XmlNode* pData = pLayer->FirstNode("data");
            const XmlNode* pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode* pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.input().shape().resize(1);
                    Shape shape = ConvertShape(pPort);
                    if (trans)
                    {
                        if (shape.size() == 4)
                        {
                            shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                            layer.input().shape()[0].format() = TensorFormatNhwc;
                        }
                    }
                    layer.input().shape()[0].dim() = shape;
                }
            }
            else
            {
                std::cout << "Can't find layer output!" << std::endl;
                return false;
            }
            return true;
        }

        bool ConvertPowerLayer(const XmlNode* pLayer, const Bytes& srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            layer.type() = Synet::LayerTypePower;
            layer.power().power() = GetWeight<float>(srcBin, second->weight()[0])[0];
            layer.src().resize(1);
            return true;
        }

        bool ConvertPreluLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            layer.type() = Synet::LayerTypePrelu;
            layer.weight() = second->weight();
            layer.src().resize(1);
            if (!CompactShape(layer.weight()[0].dim()))
                return false;
            return true;
        }

        bool ConvertPriorBoxLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePriorBox;
            layer.priorBox().version() = 1;
            Cpl::ToVal(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            Cpl::ToVal(pData->FirstAttribute("flip")->Value(), layer.priorBox().flip());
            Cpl::ToVal(pData->FirstAttribute("offset")->Value(), layer.priorBox().offset());
            ConvertVector(pData->FirstAttribute("step"), layer.priorBox().step());
            if (pData->FirstAttribute("scale_all_sizes"))
                Cpl::ToVal(pData->FirstAttribute("scale_all_sizes")->Value(), layer.priorBox().scaleAllSizes());
            ConvertVector(pData->FirstAttribute("aspect_ratio"), layer.priorBox().aspectRatio());
            ConvertVector(pData->FirstAttribute("max_size"), layer.priorBox().maxSize());
            ConvertVector(pData->FirstAttribute("min_size"), layer.priorBox().minSize());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBox().variance());
            return true;
        }

        bool ConvertPriorBoxClusteredLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePriorBoxClustered;
            ConvertVector(pData->FirstAttribute("height"), layer.priorBoxClustered().heights());
            ConvertVector(pData->FirstAttribute("width"), layer.priorBoxClustered().widths());
            Cpl::ToVal(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBoxClustered().variance());
            if (pData->FirstAttribute("img_h"))
                Cpl::ToVal(pData->FirstAttribute("img_h")->Value(), layer.priorBoxClustered().imgH());
            if (pData->FirstAttribute("img_w"))
                Cpl::ToVal(pData->FirstAttribute("img_w")->Value(), layer.priorBoxClustered().imgW());
            Cpl::ToVal(pData->FirstAttribute("step")->Value(), layer.priorBoxClustered().step());
            if (pData->FirstAttribute("step_h"))
                Cpl::ToVal(pData->FirstAttribute("step_h")->Value(), layer.priorBoxClustered().stepH());
            if (pData->FirstAttribute("step_w"))
                Cpl::ToVal(pData->FirstAttribute("step_w")->Value(), layer.priorBoxClustered().stepW());
            Cpl::ToVal(pData->FirstAttribute("offset")->Value(), layer.priorBoxClustered().offset());
            return true;
        }

        bool ConvertPriorBoxV2Layer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePriorBox;
            layer.priorBox().version() = 2;
            Cpl::ToVal(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            Cpl::ToVal(pData->FirstAttribute("flip")->Value(), layer.priorBox().flip());
            Cpl::ToVal(pData->FirstAttribute("offset")->Value(), layer.priorBox().offset());
            if (pData->FirstAttribute("scale_all_sizes"))
                Cpl::ToVal(pData->FirstAttribute("scale_all_sizes")->Value(), layer.priorBox().scaleAllSizes());
            ConvertVector(pData->FirstAttribute("aspect_ratio"), layer.priorBox().aspectRatio());
            ConvertVector(pData->FirstAttribute("max_size"), layer.priorBox().maxSize());
            ConvertVector(pData->FirstAttribute("min_size"), layer.priorBox().minSize());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBox().variance());
            return true;
        }

        bool ConvertRangeLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() != LayerTypeMeta || src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta)
                return false;
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeRange;
            return true;
        }

        bool ConvertReduceMeanLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeMeta && second->meta().type() == MetaTypeConst)
            {
                const Longs & alpha = second->meta().alpha().i64();
                if (alpha.size() != 2 || alpha[0] != 2 || alpha[1] != 3)
                    return false;
                layer.src().resize(1);
            }
            Shape input = ConvertInputShape(pLayer);
            if (input.size() != 4)
                return false;
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            layer.pooling().globalPooling() = true;
            return true;
        }

        bool ConvertReduceMinLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            bool keepDims;
            if (!ConvertValue(pData->FirstAttribute("keep_dims"), keepDims))
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReduceMin;
                layer.meta().alpha().type() = TensorType32i;
                layer.meta().alpha().shape() = Shp(1);
                layer.meta().alpha().i32().resize(1, keepDims ? 1 : 0);
            }
            else
            {
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const Longs& alpha = src1->meta().alpha().i64();
                layer.type() = Synet::LayerTypeReduction;
                layer.reduction().type() = ReductionTypeMin;
                layer.reduction().keepDims() = keepDims;
                for (size_t i = 0; i < alpha.size(); ++i)
                    layer.reduction().axis().push_back((int)alpha[i]);
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertReduceProdLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            bool keepDims;
            if (!ConvertValue(pData->FirstAttribute("keep_dims"), keepDims))
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReduceProd;
                layer.meta().alpha().type() = TensorType32i;
                layer.meta().alpha().shape() = Shp(1);
                layer.meta().alpha().i32().resize(1, keepDims ? 1 : 0);
            }
            else
            {
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const Longs& alpha = src1->meta().alpha().i64();
                layer.type() = Synet::LayerTypeReduction;
                layer.reduction().type() = ReductionTypeProd;
                layer.reduction().keepDims() = keepDims;
                for (size_t i = 0; i < alpha.size(); ++i)
                    layer.reduction().axis().push_back((int)alpha[i]);
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertReduceMaxOrSumLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                return false;
            const Longs & alpha = second->meta().alpha().i64();
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypeReduction;
            String type = pLayer->FirstAttribute("type")->Value();
            if (type == "ReduceMax")
                layer.reduction().type() = ReductionTypeMax;
            else if (type == "ReduceSum")
                layer.reduction().type() = ReductionTypeSum;
            else
                return false;
            for (size_t i = 0; i < alpha.size(); ++i)
                layer.reduction().axis().push_back((int)alpha[i]);
            Cpl::ToVal(pData->FirstAttribute("keep_dims")->Value(), layer.reduction().keepDims());
            layer.src().resize(1);
            return true;
        }

        bool ConvertRegionYoloLayer(const XmlNode* pLayer, LayerParams& layers, bool trans, LayerParam& layer, IndexMap & index)
        {
            layer.type() = LayerTypeYolo;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (!ConvertVector(pData->FirstAttribute("anchors"), layer.yolo().anchors()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("classes"), layer.yolo().classes()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("mask"), layer.yolo().mask()))
                return false;
            Shape output = ConvertOutputShape(pLayer);
            layer.yolo().num() = (int)(output[1] / (layer.yolo().classes() + 5));
            if (trans)
            {
                LayerParam permute;
                permute.type() = LayerTypePermute;
                permute.src() = layer.src();
                permute.name() = layer.name() + "_auto_permute";
                permute.dst().push_back(permute.name());
                permute.permute().order() = Shp(0, 3, 1, 2);
                permute.permute().format() = TensorFormatNchw;
                layer.src() = permute.dst();
                layers.push_back(permute);
                size_t layerId;
                Cpl::ToVal(pLayer->FirstAttribute("id")->Value(), layerId);
                index[layerId]++;
            }
            return true;
        }

        bool ConvertReluLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertReshapeLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape input = ConvertInputShape(pLayer);
            Shape output = ConvertOutputShape(pLayer);
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
                return false;
            if(first->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReshape;
            }            
            else if (second->meta().type() == MetaTypeConst)
            {
                if (second->meta().alpha().shape().size() != 1)
                    return false;
                if (!CheckDims(output, second->meta().alpha().shape()[0], "output shape"))
                    return false;
                Shape & shape = layer.reshape().shape();
                const int64_t* alpha = second->meta().alpha().i64().data();
                layer.type() = LayerTypeReshape;
                shape.resize(output.size());
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    shape[i] = output[i];
                    //shape[i] = (size_t)alpha[i];
                }
                layer.src().resize(1);
                if (trans && !PermutedToNchw(layers, layer.src(), true, false, false))
                {
                    if (shape.size() == 3 && input.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2], shape[1] });
                    }
                    if (shape.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                    }
                    if (shape.size() == 5 && input[1] == output[1] * output[2])
                    {
                        shape = Shape({ shape[0], shape[3], shape[4], shape[1], shape[2] });
                    }
                }
                if (input.size() > 1 && output.size() > 1 && input[0] == 1 && output[0] == 1)
                {
                    layer.reshape().axis() = 1;
                    shape.erase(shape.begin(), shape.begin() + 1);
                }            
            }
            else
            {
                layer.type() = LayerTypeReshape;
            }
            return true;
        }

        bool ConvertResultLayer(const XmlNode* pLayer, LayerParam& layer, Synet::NetworkParam * network)
        {
            layer.type() = Synet::LayerTypeStub;
            if (layer.dst().empty())
                layer.dst().push_back(layer.src()[0]);
            if (network && layer.parent().empty())
                network->dst().push_back(layer.src()[0]);
            return true;
        }

        bool ConvertReverseSequenceLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReverseSequence;
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (!ConvertValue(pData->FirstAttribute("batch_axis"), layer.reverseSequence().batchAxis()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("seq_axis"), layer.reverseSequence().seqAxis()))
                return false;
            return true;
        }

        bool ConvertSelectLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* sp0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* sp2 = GetLayer(layers, layer.src()[2]);
            if (sp0 == NULL || sp1 == NULL || sp2 == NULL || 
                sp0->type() != LayerTypeMeta || sp1->type() != LayerTypeMeta || sp2->type() != LayerTypeMeta)
                return false;
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeSelect;
            return true;
        }

        bool ConvertShapeOfLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeShape;
            layer.meta().version() = 1;
            return true;
        }

        bool ConvertSigmoidLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        bool ConvertSoftmaxOrLogSoftmaxLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            Cpl::ToVal(pData->FirstAttribute("axis")->Value(), layer.softmax().axis());
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                Shape input = ConvertInputShape(pLayer);
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.softmax().axis() = (int32_t)nchw[layer.softmax().axis()];
                }
            }
            String type = pLayer->FirstAttribute("type")->Value();
            if (type == "LogSoftmax")
                layer.softmax().log() = true;
            return true;
        }

        bool ConvertSplitLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnpack;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            size_t numSplits;
            if (!ConvertValue(pData->FirstAttribute("num_splits"), numSplits))
                return false;
            if (!CheckDestinationNumber(layer, numSplits))
                return false;
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL || src1->type() != LayerTypeMeta)
                return false;
            if (src1->meta().type() == MetaTypeConst)
            {
                if (src1->meta().alpha().shape().size() != 1 || src1->meta().alpha().shape()[0] != 1)
                    return false;
                switch (src1->meta().alpha().type())
                {
                case TensorType64i:
                    layer.unpack().axis() = (int32_t)src1->meta().alpha().i64()[0];
                    break;
                default:
                    return false;
                }
                if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
                {
                    Shape input = ConvertInputShape(pLayer);
                    if (input.size() == 4)
                    {
                        Shape nchw = Shape({ 0, 3, 1, 2 });
                        layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                    }
                    if (input.size() == 3)
                    {
                        if (src0->type() == LayerTypePermute)
                        {
                            layer.unpack().axis() = 1;
                        }
                        else
                        {
                            Shape nchw = Shape({ 2, 0, 1 });
                            layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                        }
                    }
                } 
                layer.src().resize(1);
            }
            else
                return false;
            return true;
        }

        bool ConvertSqrtLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeSqrt;
            return true;
        }

        bool ConvertSqueezeLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSqueeze;
            }
            else
                layer.type() = Synet::LayerTypeSqueeze;
            return true;
        }

        bool ConvertStridedSliceLayer(const XmlNode* pLayer, const LayerParams& layers, const Bytes& srcBin, bool trans, LayerParam& layer)
        {
            bool meta = true;
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                const LayerParam * source = GetLayer(layers, layer.src()[s]);
                if (source == NULL)
                    return false;
                if (source->type() != LayerTypeMeta)
                    meta = false;
            }
            if (meta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeStridedSlice;
            }
            else
            {
                layer.type() = Synet::LayerTypeStridedSlice;
                const XmlNode * pData = pLayer->FirstNode("data");
                if (pData == NULL)
                    return false;
                if (!ConvertShapeParameter(pData, "begin_mask", trans, layer.stridedSlice().beginMask()))
                    return false;
                if (!ConvertShapeParameter(pData, "ellipsis_mask", trans, layer.stridedSlice().ellipsisMask()))
                    return false;
                if (!ConvertShapeParameter(pData, "end_mask", trans, layer.stridedSlice().endMask()))
                    return false;
                if (!ConvertShapeParameter(pData, "new_axis_mask", trans, layer.stridedSlice().newAxisMask()))
                    return false;
                if (!ConvertShapeParameter(pData, "shrink_axis_mask", trans, layer.stridedSlice().shrinkAxisMask()))
                    return false;
                for (size_t s = layer.src().size() - 1; s > 0; --s)
                {
                    const LayerParam * source = GetLayer(layers, layer.src()[s]);
                    bool result = false;
                    switch (s)
                    {
                    case 1:
                        result = GetShapeFromConst(*source, srcBin, trans, layer.stridedSlice().beginDims());
                        break;
                    case 2:
                        result = GetShapeFromConst(*source, srcBin, trans, layer.stridedSlice().endDims());
                        break;
                    case 3:
                        result = GetShapeFromConst(*source, srcBin, trans, layer.stridedSlice().strideDims());
                        break;
                    }
                    if (!result)
                        return false;
                }
            }
            return true;
        }

        bool ConvertSubtractLayer(const XmlNode* pLayer, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape is0 = ConvertInputShape(pLayer, "0");
            Shape is1 = ConvertInputShape(pLayer, "1");
            const LayerParam* sp0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            if (sp0 == NULL || sp1 == NULL)
                return false;
            if (sp1->type() == LayerTypeConst && (TensorSize(is1) == 1 || TensorSize(is1) == 0))
            {
                layer.type() = Synet::LayerTypePower;
                const float* pShift = GetWeight<float>(srcBin, sp1->weight()[0]);
                layer.power().shift() = -pShift[0];
                layer.src().resize(1);
            }
            else if (sp0->type() == LayerTypeConst && (TensorSize(is0) == 1 || TensorSize(is0) == 0))
            {
                layer.type() = Synet::LayerTypePower;
                layer.power().scale() = -1.0f;
                const float* pShift = GetWeight<float>(srcBin, sp0->weight()[0]);
                layer.power().shift() = pShift[0];
                layer.src()[0] = layer.src()[1];
                layer.src().resize(1);
            }
            else if (sp1->type() == LayerTypeConst && SignificantDimsCount(is1) == 1)
            {
                layer.type() = Synet::LayerTypeBias;
                layer.weight() = sp1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                const float* pSrc = GetWeight<float>(srcBin, layer.weight()[0]);
                float* pDst = GetWeight<float>(dstBin, layer.weight()[0]);
                size_t size = TensorSize(layer.weight()[0].dim());
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = -pSrc[i];
                layer.src().resize(1);
            }
            else
            {
                if (TensorSize(is0) != TensorSize(is1))
                    return false;
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeSum;
                layer.eltwise().coefficients() = Floats({1.0f, -1.0f});
            }
            return true;
        }

        bool ConvertSwishLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSwish;
            return true;
        }

        bool ConvertTanhLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeTanh;
            return true;
        }

        bool ConvertTileLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = Synet::LayerTypeTile;
            Shape input = ConvertInputShape(pLayer);
            Shape output = ConvertOutputShape(pLayer);
            if (input.size() != output.size())
                return false;
            for (size_t i = 0, already = 0; i < input.size(); ++i)
            {
                if (input[i] != output[i])
                {
                    layer.tile().axis() = int(i);
                    layer.tile().tiles() = int(output[i] / input[i]);
                }
            }
            if (trans && input.size() == 4 && !PermutedToNchw(layers, layer.src(), true, false, false))
            {
                uint32_t order[4] = { 0, 3, 1, 2 };
                layer.tile().axis() = order[layer.tile().axis()];
            }
            layer.src().resize(1);
            return true;
        }

        bool ConvertTopKLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* sp1 = GetLayer(layers, layer.src()[1]);
            if (sp1 == NULL || sp1->type() != LayerTypeMeta)
                return false;
            int64_t k = -1;
            if (sp1->meta().type() == MetaTypeConst && sp1->meta().alpha().type() == TensorType64i)
                k = sp1->meta().alpha().i64()[0];
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypeTopK;
            if (!ConvertValue(pData->FirstAttribute("axis"), layer.topK().axis()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("mode"), layer.topK().mode()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("sort"), layer.topK().sort()))
                return false;
            String indexElementType;
            if (!ConvertValue(pData->FirstAttribute("index_element_type"), indexElementType)) 
                return false;
            if (indexElementType == "i64")
                layer.topK().indexElementType() = TensorType64i;
            else if (indexElementType == "i32")
                layer.topK().indexElementType() = TensorType32i;
            else
                return false;
            if (k == 1 && layer.topK().mode() == TopKModeMax)
            {
                layer.type() = LayerTypeArgMax;
                layer.src().resize(1);
                layer.dst().resize(1);
                layer.dst()[0] = layer.name();
                layer.argMax().axis() = layer.topK().axis();
            }
            return true;
        }

        bool ConvertTensorIteratorLayer(const XmlNode* pParent, bool trans, const LayerParams& parents, const Bytes& srcBin, LayerParam& parent, Bytes& dstBin, TensorInfoMap& info, LayerParams & children)
        {
            if (trans)
                trans = !PermutedToNchw(parents, parent.src(), false, true, false);

            const XmlNode* pBody = pParent->FirstNode("body");
            if (pBody == NULL)
                return false;

            Edges edges;
            if (!ParseEdges(*pBody, edges))
                return false;

            IndexMap index;
            typedef std::map<String, String> NameMap;
            NameMap names;

            const XmlNode* pLayers = pBody->FirstNode("layers");
            if (pLayers == NULL)
                return false;
            const XmlNode* pChild= pLayers->FirstNode("layer"), * pPrevChild = NULL, * pNextChild = NULL;
            while (pChild)
            {
                pNextChild = pChild->NextSibling("layer");

                LayerParam child;
                child.parent() = parent.name();
                if (!ParseInputOutput(*pChild, edges, children, child, index, info))
                    return false;
                names[pChild->FirstAttribute("id")->Value()] = pChild->FirstAttribute("name")->Value();
                String type = pChild->FirstAttribute("type")->Value();

                if (type == "Add" && !ConvertAddLayer(pChild, children, srcBin, child))
                    return ErrorMessage(pChild);
                if (type == "Concat" && !ConvertConcatLayer(pChild, children, trans, child, index))
                    return ErrorMessage(pChild);
                if (type == "Const" && !ConvertConstLayer(pChild, srcBin, child))
                    return ErrorMessage(pChild);
                if ((type == "MatMul") && !ConvertMatMulLayer(pChild, trans, children, srcBin, child, dstBin, info))
                    return ErrorMessage(pChild);
                if (type == "Multiply" && !ConvertMultiplyLayer(pChild, srcBin, children, child))
                    return ErrorMessage(pChild);
                if (type == "Parameter" && !ConvertParameterLayer(pChild, trans, child))
                    return ErrorMessage(pChild);
                if (type == "Result" && !ConvertResultLayer(pChild, child, NULL))
                    return ErrorMessage(pChild);
                if (type == "Sigmoid" && !ConvertSigmoidLayer(pChild, child))
                    return ErrorMessage(pChild);
                if (type == "Split" && !ConvertSplitLayer(pChild, children, trans, child))
                    return ErrorMessage(pChild);
                if (type == "Squeeze" && !ConvertSqueezeLayer(pChild, children, child))
                    return ErrorMessage(pChild);
                if (type == "Subtract" && !ConvertSubtractLayer(pChild, children, srcBin, child, dstBin))
                    return ErrorMessage(pChild);
                if (type == "Tanh" && !ConvertTanhLayer(pChild, child))
                    return ErrorMessage(pChild);
                if (type == "Unsqueeze" && !ConvertUnsqueezeLayer(pChild,  children, child))
                    return ErrorMessage(pChild);

#if defined(SYNET_IE_PARSE_STOP_ON_ERROR)
                if (child.type() == LayerTypeUnknown)
                    return ErrorMessage(pChild);
#else
                if (child.type() == LayerTypeUnknown)
                {
                    NotImplemented(pChild, child);
                    std::cout << "TensorIterator " << parent.name() << " : not implemented layer : name = " << child.name() << " ; type = " << type << std::endl;
                }
#endif
                children.push_back(child);
                pPrevChild = pChild;
                pChild = pNextChild;
            }

            parent.type() = Synet::LayerTypeTensorIterator;

            const XmlNode* pPortMap = pParent->FirstNode("port_map");
            if (pPortMap == NULL)
                return false;

            const XmlNode* pInput = pPortMap->FirstNode("input");
            if (pInput == NULL)
                return false;
            while (pInput)
            {
                int port = -1, layer = -1, axis = -1;
                if (!ConvertValue(pInput->FirstAttribute("external_port_id"), port))
                    return false;
                if (!ConvertValue(pInput->FirstAttribute("internal_layer_id"), layer))
                    return false;
                ConvertValue(pInput->FirstAttribute("axis"), axis);

                ConnectionParam connection;
                connection.port() = port;
                connection.dst() = names[Cpl::ToStr(layer)];
                connection.axis() = axis;
                parent.tensorIterator().input().push_back(connection);

                pInput = pInput->NextSibling("input");
            }

            const XmlNode* pOutput = pPortMap->FirstNode("output");
            if (pOutput == NULL)
                return false;
            while (pOutput)
            {
                int port = -1, layer = -1, axis = -1;
                if (!ConvertValue(pOutput->FirstAttribute("external_port_id"), port))
                    return false;
                if (!ConvertValue(pOutput->FirstAttribute("internal_layer_id"), layer))
                    return false;
                ConvertValue(pOutput->FirstAttribute("axis"), axis);

                ConnectionParam connection;
                connection.src() = names[Cpl::ToStr(layer)];
                connection.port() = port - (int)parent.src().size();
                connection.axis() = axis;
                parent.tensorIterator().output().push_back(connection);

                pOutput = pOutput->NextSibling("output");
            }

            const XmlNode* pBackEdges = pParent->FirstNode("back_edges");
            if (pBackEdges)
            {
                const XmlNode* pEdge = pBackEdges->FirstNode("edge");
                while (pEdge)
                {
                    int src = -1, dst = -1;
                    if (!ConvertValue(pEdge->FirstAttribute("from-layer"), src))
                        return false;
                    if (!ConvertValue(pEdge->FirstAttribute("to-layer"), dst))
                        return false;

                    ConnectionParam connection;
                    connection.src() = names[Cpl::ToStr(src)];
                    connection.dst() = names[Cpl::ToStr(dst)];
                    parent.tensorIterator().back().push_back(connection);

                    pEdge = pEdge->NextSibling("edge");
                }
            }

            return true;
        }

        bool ConvertTransposeLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape first = ConvertInputShape(pLayer);
            const LayerParam * prev = GetLayer(layers, layer.src()[0]);
            const LayerParam * second = GetLayer(layers, layer.src()[1]);
            if (prev == NULL || second == NULL || second->type() != LayerTypeMeta)
                return false;
            if (second->meta().type() == MetaTypeConst)
            {
                if (second->meta().alpha().shape().size() != 1)
                    return false;
                if (!CheckDims(first, second->meta().alpha().shape()[0], "order size"))
                    return false;
                layer.type() = LayerTypePermute;
                Shape& order = layer.permute().order();
                order.resize(first.size());
                const int64_t * alpha = second->meta().alpha().i64().data();
                for (size_t i = 0; i < order.size(); ++i)
                    order[i] = (size_t)alpha[i];
                layer.src().resize(1);
                if (trans && !PermutedToNchw(layers, layer.src(), true, false, false))
                {
                    if (order.size() == 4)
                    {
                        if (order == Shape({ 0, 2, 3, 1 }))
                        {
                            order = Shape({ 0, 1, 2, 3 });
                            layer.permute().format() = TensorFormatNchw;
                        }
                        else
                        {
                            Shape nhwc = Shape({ 0, 2, 3, 1 });
                            Shape nchw = Shape({ 0, 3, 1, 2 });
                            order = Shape({ nchw[order[nhwc[0]]], nchw[order[nhwc[1]]], nchw[order[nhwc[2]]], nchw[order[nhwc[3]]] });
                        }
                    }
                    else if (order.size() == 5)
                    {
                        if (order == Shape({ 0, 2, 1, 3, 4 }))
                            order = Shape({ 0, 1, 2, 4, 3 });
                    }
                    else if (order.size() == 3 && prev->type() == LayerTypeReshape)
                    {
                        if (prev->reshape().shape().size() == 2 && prev->reshape().axis() == 1 && order == Shape({ 0, 2, 1 }))
                        {
                            order = Shape({ 0, 1, 2 });
                            layer.permute().format() = TensorFormatNchw;
                        }
                    }
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertVariadicSplitLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnpack;
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst
                || src2 == NULL || src2->type() != LayerTypeMeta || src2->meta().type() != MetaTypeConst)
                return false;

            if (src1->meta().alpha().shape().size() != 1 || src1->meta().alpha().shape()[0] != 1)
                return false;
            switch (src1->meta().alpha().type())
            {
            case TensorType64i:
                layer.unpack().axis() = (int32_t)src1->meta().alpha().i64()[0];
                break;
            default:
                return false;
            }
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                Shape input = ConvertInputShape(pLayer);
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                }
                if (input.size() == 3)
                {
                    if (src0->type() == LayerTypePermute)
                    {
                        layer.unpack().axis() = 1;
                    }
                    else
                    {
                        Shape nchw = Shape({ 2, 0, 1 });
                        layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                    }
                }
            }

            if (src2->meta().alpha().shape().size() != 1)
                return false;
            switch (src2->meta().alpha().type())
            {
            case TensorType64i:
            {
                const int64_t * src = src2->meta().alpha().i64().data();
                layer.unpack().parts().resize(src2->meta().alpha().i64().size());
                for (size_t i = 0; i < layer.unpack().parts().size(); ++i)
                    layer.unpack().parts()[i] = (size_t)(src[i]);
                break;
            }
            default:
                return false;
            }
            layer.src().resize(1);
            return true;
        }

        bool ConvertUnsqueezeLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            if (first->type() == LayerTypePriorBoxClustered || first->type() == LayerTypePriorBox)
            {
                layer.type() = Synet::LayerTypeStub;
                layer.src().resize(1);
            }
            else
            {
                const LayerParam* second = GetLayer(layers, layer.src()[1]);
                if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                    return false;
                if (first->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeExpandDims;
                }
                else
                {
                    const int64_t* alpha = second->meta().alpha().i64().data();
                    layer.type() = Synet::LayerTypeExpandDims;
                    layer.expandDims().axis() = (int32_t)alpha[0];
                    layer.src().resize(1);
                }
            }
            return true;
        }

        //---------------------------------------------------------------------

        template<class S, class D> bool GetShapeFromWeight(const WeightParam & weight, const Bytes& bin, std::vector<D> & shape)
        {
            size_t size = weight.dim()[0];
            if (size != weight.size() / sizeof(S))
                return false;
            const S * data = GetWeight<S>(bin, weight);
            shape.resize(size);
            for (size_t i = 0; i < size; ++i)
                shape[i] = (D)data[i];
            return true;
        }

        template<class D> bool GetShapeFromConst(const LayerParam & layer, const Bytes& bin, bool trans, std::vector<D>& shape)
        {
            if (layer.type() == LayerTypeConst)
            {
                if (layer.weight().size() != 1)
                    return false;
                const WeightParam & weight = layer.weight()[0];
                if (weight.dim().size() != 1)
                    return false;
                switch (weight.type())
                {
                case TensorType32i:
                    if (!GetShapeFromWeight<int32_t, D>(weight, bin, shape))
                        return false;
                case TensorType64i:
                    if (!GetShapeFromWeight<int64_t, D>(weight, bin, shape))
                        return false;
                default:
                    return false;
                }
            }
            else if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst)
            {
                const TensorParam & alpha = layer.meta().alpha();
                if (alpha.shape().size() != 1)
                    return false;
                shape.resize(alpha.shape()[0]);
                if (alpha.type() == TensorType32i)
                {
                    for (size_t i = 0; i < shape.size(); ++i)
                        shape[i] = (D)alpha.i32()[i];
                }
                else if (alpha.type() == TensorType64i)
                {
                    for (size_t i = 0; i < shape.size(); ++i)
                        shape[i] = (D)alpha.i64()[i];
                }
                else
                    return false;
            }
            else
                return false;
            if (trans)
            {
                if (shape.size() == 4)
                {
                    shape = std::vector<D>({ shape[0], shape[2], shape[3], shape[1] });
                }
            }
            return true;
        }

        bool ConvertShapeParameter(const XmlNode * data, const String & name, bool trans, Shape & shape)
        {
            if (!ConvertVector(data->FirstAttribute(name.c_str()), shape))
                return false;
            if (trans)
            {
                if (shape.size() == 4)
                {
                    shape = Shp(shape[0], shape[2], shape[3], shape[1]);
                }
            }
            return true;
        }

        int GetOpsetVersion(const XmlNode* pLayer)
        {
            String version = pLayer->FirstAttribute("version")->Value();
            if (version.substr(0, 5) == "opset")
                return Cpl::ToVal<int>(version.substr(5));
            else
                return -1;
        }

        bool ManualInsertToNchwPermute(const OnnxParam& onnxParam, LayerParams& layers)
        {
            LayerParam& layer = layers.back();
            for (size_t h = 0; h < onnxParam.toNchwHints().size(); ++h)
            {
                if (layer.name() == onnxParam.toNchwHints()[h])
                {
                    for (size_t d = 0; d < layer.dst().size(); ++d)
                    {
                        String old = layer.dst()[d];
                        layer.dst()[d] = old + "_before_permute_to_nchw";
                        LayerParam permute;
                        permute.type() = LayerTypePermute;
                        permute.src().push_back(layer.dst()[d]);
                        permute.name() = old + "_permute_to_nchw";
                        permute.dst().push_back(old);
                        permute.permute().order() = Shape({ 0, 3, 1, 2 });
                        permute.permute().format() = TensorFormatNchw;
                        layers.push_back(permute);
                    }
                }
            }
            return true;
        }
    };
}
