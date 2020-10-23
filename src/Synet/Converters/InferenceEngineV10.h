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

#include "Synet/Converters/InferenceEngineBase.h"

namespace Synet
{
    class InferenceEngineConverterV10 : public InferenceEngineConverter
    {
    public:
        bool Convert(const Xml::XmlNode<char>& srcXml, const std::vector<float>& srcBin, bool trans, Synet::NetworkParam & dstXml, std::vector<float>& dstBin)
        {
            dstXml.version() = 1;

            const XmlAttr* pName = srcXml.FirstAttribute("name");
            if (pName)
                dstXml.name() = pName->Value();

            Edges edges;
            if (!ParseEdges(srcXml, edges))
                return false;

            IndexMap index;

            const XmlNode* pLayers = srcXml.FirstNode("layers");
            if (pLayers == NULL)
                return false;
            const XmlNode* pLayer = pLayers->FirstNode("layer"), * pPrevLayer = NULL, * pNextLayer = NULL;
            while (pLayer)
            {
                pNextLayer = pLayer->NextSibling("layer");

                LayerParam layer;
                if (!ParseInputOutput(*pLayer, edges, dstXml.layers(), layer, index, _tensors))
                    return false;

                String type = pLayer->FirstAttribute("type")->Value();
                if (type == "Add" && !ConvertAddLayer(pLayer, dstXml.layers(), srcBin, layer))
                    return ErrorMessage(pLayer);
                if (type == "Clamp" && !ConvertClampLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Concat" && !ConvertConcatLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Const" && !ConvertConstLayer(pLayer, srcBin, layer))
                    return ErrorMessage(pLayer);
                if (type == "Convert" && !ConvertConvertLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "Convolution" || type == "GroupConvolution") && !ConvertConvolutionLayer(pLayer, trans, dstXml.layers(), srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Gather" && !ConvertGatherLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "DetectionOutput" && !ConvertDetectionOutputLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Interpolate" && !ConvertInterpolateLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if ((type == "MatMul") && !ConvertMatMulLayer(pLayer, trans, dstXml.layers(), srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "MaxPool" && !ConvertMaxPoolLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Multiply" && !ConvertMultiplyLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Parameter" && !ConvertParameterLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "PReLU" && !ConvertPreluLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBox" && !ConvertPriorBoxLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxClustered" && !ConvertPriorBoxClusteredLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxV2" && !ConvertPriorBoxV2Layer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceMean" && !ConvertReduceMeanLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceProd" && !ConvertReduceProdLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReduceSum" && !ConvertReduceSumLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "RegionYolo" && !ConvertRegionYoloLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Reshape" && !ConvertReshapeLayer(pLayer, srcBin, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Result" && !ConvertResultLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "ShapeOf" && !ConvertShapeOfLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Sigmoid" && !ConvertSigmoidLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "SoftMax" && !ConvertSoftmaxLayer(pLayer, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "StridedSlice" && !ConvertStridedSliceLayer(pLayer, dstXml.layers(), srcBin, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Transpose" && !ConvertTransposeLayer(pLayer, srcBin, dstXml.layers(), trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Unsqueeze" && !ConvertUnsqueezeLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
#if 1
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(pLayer);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(pLayer, layer);
                    std::cout << "Not implemented layer : name = " << layer.name() << " ; type = " << type << std::endl;
                }
#endif
                dstXml.layers().push_back(layer);
                _layers[layer.name()] = layer;
                pPrevLayer = pLayer;
                pLayer = pNextLayer;
            }

            if (!RemoveUnusedConst(dstXml.layers()))
                return false;

            return true;
        }

    private:

        TensorInfoMap _tensors;
        LayerParamMap _layers;

        struct Pin
        {
            String name;
            int index;
            Pin(const String& n = String(), int i = 0) : name(n), index(i) {}
        };

        bool ConvertAddLayer(const XmlNode* pLayer, const LayerParams& layers, const Vector & srcBin, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape src0 = ConvertInputShape(pLayer, "0");
            Shape src1 = ConvertInputShape(pLayer, "1");
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeConst && TensorSize(src0) >= TensorSize(src1))
            {
                if (TensorSize(src1) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    const float * pShift = srcBin.data() + second->weight()[0].offset() / sizeof(float);
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

        bool ConvertClampLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("min")->Value(), layer.restrictRange().lower());
            StringToValue(pData->FirstAttribute("max")->Value(), layer.restrictRange().upper());
            return true;
        }

        bool ConvertConcatLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == Synet::LayerTypeMeta)
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
                StringToValue(pData->FirstAttribute("axis")->Value(), layer.concat().axis());
                if (trans && !PermutedToNchw(layers, false, true))
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
            }
            return true;
        }

        bool ConvertConstLayer(const XmlNode* pLayer, const Vector & srcBin, LayerParam & layer)
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
                StringToValue(pData->FirstAttribute("offset")->Value(), offset);
                if (type == "f32")
                {
                    layer.type() = Synet::LayerTypeConst;
                    layer.weight().resize(1);
                    layer.weight()[0].type() = TensorType32f;
                    layer.weight()[0].dim() = shape;
                    layer.weight()[0].offset() = offset;
                    StringToValue(pData->FirstAttribute("size")->Value(), layer.weight()[0].size());
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
                else if (type == "i64")
                    layer.meta().alpha().type() = TensorType64i;
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        bool ConvertConvolutionLayer(const XmlNode* pLayer, bool trans, const LayerParams& layers, const Vector& srcBin, LayerParam& layer, Vector& dstBin)
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
            if (trans)
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }

        bool ConvertDetectionOutputLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeDetectionOutput;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String codeType = pData->FirstAttribute("code_type")->Value();
            if (codeType == "caffe.PriorBoxParameter.CENTER_SIZE")
                layer.detectionOutput().codeType() = PriorBoxCodeTypeCenterSize;
            else
                assert(0);
            StringToValue(pData->FirstAttribute("confidence_threshold")->Value(), layer.detectionOutput().confidenceThreshold());
            StringToValue(pData->FirstAttribute("keep_top_k")->Value(), layer.detectionOutput().keepTopK());
            StringToValue(pData->FirstAttribute("nms_threshold")->Value(), layer.detectionOutput().nms().nmsThreshold());
            StringToValue(pData->FirstAttribute("num_classes")->Value(), layer.detectionOutput().numClasses());
            StringToValue(pData->FirstAttribute("variance_encoded_in_target")->Value(), layer.detectionOutput().varianceEncodedInTarget());
            StringToValue(pData->FirstAttribute("top_k")->Value(), layer.detectionOutput().nms().topK());
            StringToValue(pData->FirstAttribute("share_location")->Value(), layer.detectionOutput().shareLocation());
            ConvertValue(pData->FirstAttribute("clip"), layer.detectionOutput().clip());
            ConvertValue(pData->FirstAttribute("background_label_id"), layer.detectionOutput().backgroundLabelId());
            return true;
        }

        bool ConvertGatherLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeGather;
            return true;
        }

        bool ConvertInterpolateLayer(const XmlNode* pLayer, const Vector& srcBin, const LayerParams& layers, LayerParam& layer)
        {
#if 0
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = LayerTypeInterp2;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (pData->FirstAttribute("factor"))
                StringToValue(pData->FirstAttribute("factor")->Value(), layer.interp2().factor());
            const XmlAttr* pPadBeg = pData->FirstAttribute("pad_beg");
            const XmlAttr* pPadEnd = pData->FirstAttribute("pad_end");
            if (pPadBeg && pPadEnd)
            {
                size_t padBeg, padEnd;
                StringToValue(pPadBeg->Value(), padBeg);
                StringToValue(pPadEnd->Value(), padEnd);
                layer.interp2().pad() = Shape({ padBeg, padBeg, padEnd, padEnd });
            }
            if (pData->FirstAttribute("align_corners"))
                StringToValue(pData->FirstAttribute("align_corners")->Value(), layer.interp2().alignCorners());

            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                return false;
            if (second->meta().alpha().shape().size() != 1 || second->meta().alpha().shape()[0] != 2)
                return false;
            const int64_t* alpha = second->meta().alpha().i64().data();
            layer.interp2().height() = (int32_t)alpha[0];
            layer.interp2().width() = (int32_t)alpha[1];
            layer.src().resize(1);
#else
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = LayerTypeInterp;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr* pMode = pData->FirstAttribute("mode");
            if (pMode && String(pMode->Value()) == "nearest")
                layer.interp().interpolationType() = InterpolationTypeNearest;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                return false;
            if (second->meta().alpha().shape().size() != 1 || second->meta().alpha().shape()[0] != 2)
                return false;
            const int64_t* alpha = second->meta().alpha().i64().data();
            layer.interp().height() = (int32_t)alpha[0];
            layer.interp().width() = (int32_t)alpha[1];
            layer.src().resize(1);
#endif
            return true;
        }

        bool ConvertMatMulLayer(const XmlNode* pLayer, bool trans, const LayerParams& layers, const Vector& srcBin, LayerParam& layer, Vector& dstBin)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL || pData->FirstAttribute("transpose_a") == NULL || pData->FirstAttribute("transpose_b") == NULL)
                return false;
            bool transposeA, transposeB;
            StringToValue(pData->FirstAttribute("transpose_a")->Value(), transposeA);
            StringToValue(pData->FirstAttribute("transpose_b")->Value(), transposeB);
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
            if (trans && !PermutedToNchw(layers, true, false))
            {
                const LayerParam * first = GetLayer(layers, layer.src()[0]);
                if (first == NULL)
                    return false;
                if (first->type() == LayerTypePooling && first->pooling().globalPooling())
                    return true;
                if (first->type() != LayerTypeReshape)
                    return false;
                Shape origin = _tensors[first->src()[0]].shape;
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
            return true;
        }

        bool ConvertMultiplyLayer(const XmlNode* pLayer, const Vector & srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (first == NULL || second == NULL)
                return false;
            if (first->type() == LayerTypeConst || second->type() == LayerTypeConst)
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
                            shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                        layer.input().shape()[0].format() = TensorFormatNhwc;
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
            StringToValue(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            StringToValue(pData->FirstAttribute("flip")->Value(), layer.priorBox().flip());
            StringToValue(pData->FirstAttribute("offset")->Value(), layer.priorBox().offset());
            ConvertVector(pData->FirstAttribute("step"), layer.priorBox().step());
            if (pData->FirstAttribute("scale_all_sizes"))
                StringToValue(pData->FirstAttribute("scale_all_sizes")->Value(), layer.priorBox().scaleAllSizes());
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
            StringToValue(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBoxClustered().variance());
            if (pData->FirstAttribute("img_h"))
                StringToValue(pData->FirstAttribute("img_h")->Value(), layer.priorBoxClustered().imgH());
            if (pData->FirstAttribute("img_w"))
                StringToValue(pData->FirstAttribute("img_w")->Value(), layer.priorBoxClustered().imgW());
            StringToValue(pData->FirstAttribute("step")->Value(), layer.priorBoxClustered().step());
            if (pData->FirstAttribute("step_h"))
                StringToValue(pData->FirstAttribute("step_h")->Value(), layer.priorBoxClustered().stepH());
            if (pData->FirstAttribute("step_w"))
                StringToValue(pData->FirstAttribute("step_w")->Value(), layer.priorBoxClustered().stepW());
            StringToValue(pData->FirstAttribute("offset")->Value(), layer.priorBoxClustered().offset());
            return true;
        }

        bool ConvertPriorBoxV2Layer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePriorBox;
            layer.priorBox().version() = 2;
            StringToValue(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            StringToValue(pData->FirstAttribute("flip")->Value(), layer.priorBox().flip());
            StringToValue(pData->FirstAttribute("offset")->Value(), layer.priorBox().offset());
            if (pData->FirstAttribute("scale_all_sizes"))
                StringToValue(pData->FirstAttribute("scale_all_sizes")->Value(), layer.priorBox().scaleAllSizes());
            ConvertVector(pData->FirstAttribute("aspect_ratio"), layer.priorBox().aspectRatio());
            ConvertVector(pData->FirstAttribute("max_size"), layer.priorBox().maxSize());
            ConvertVector(pData->FirstAttribute("min_size"), layer.priorBox().minSize());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBox().variance());
            return true;
        }

        bool ConvertReduceMeanLayer(const XmlNode* pLayer, const Vector & srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                return false;
            const Longs & alpha = second->meta().alpha().i64();
            if (alpha.size() != 2 || alpha[0] != 2 || alpha[1] != 3)
                return false;
            Shape input = ConvertInputShape(pLayer);
            if (input.size() != 4)
                return false;
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            layer.pooling().globalPooling() = true;
            layer.src().resize(1);
            return true;
        }

        bool ConvertReduceProdLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr* pKeepDims = pData->FirstAttribute("keep_dims");
            if (pKeepDims == NULL)
                return false;
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeReduceProd;
            layer.meta().alpha().type() = TensorType32i;
            layer.meta().alpha().shape() = Shp(1);
            layer.meta().alpha().i32().resize(1, String(pKeepDims->Value()) == "True" ? 1 : 0);
            return true;
        }

        bool ConvertReduceSumLayer(const XmlNode* pLayer, const Vector& srcBin, const LayerParams& layers, LayerParam& layer)
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
            layer.reduction().type() = ReductionTypeSum;
            for (size_t i = 0; i < alpha.size(); ++i)
                layer.reduction().axis().push_back((int)alpha[i]);
            StringToValue(pData->FirstAttribute("keep_dims")->Value(), layer.reduction().keepDims());
            layer.src().resize(1);
            return true;
        }

        bool ConvertRegionYoloLayer(const XmlNode* pLayer, LayerParams& layers, bool trans, LayerParam& layer)
        {
            layer.type() = LayerTypeYolo;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (!ConvertVector(pData->FirstAttribute("anchors"), layer.yolo().anchors()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("classes"), layer.yolo().classes()))
                return false;
            if (!ConvertValue(pData->FirstAttribute("num"), layer.yolo().num()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("mask"), layer.yolo().mask()))
                return false;
            layer.yolo().num() /= 2;
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
            }
            return true;
        }

        bool ConvertReluLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertReshapeLayer(const XmlNode* pLayer, const Vector & srcBin, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape input = ConvertInputShape(pLayer);
            Shape output = ConvertOutputShape(pLayer);
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
                return false;
            if (second->meta().type() == MetaTypeConst)
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
                    shape[i] = (size_t)alpha[i];
                layer.src().resize(1);
                if (input.size() > 1 && output.size() > 1 && input[0] == 1 && output[0] == 1)
                {
                    layer.reshape().axis() = 1;
                    shape.erase(shape.begin(), shape.begin() + 1);
                }
                if (trans && !PermutedToNchw(layers, false, false))
                {
                    if (shape.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                    }
                }
            }
            else if(first->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReshape;
            }
            else
            {
                layer.type() = LayerTypeReshape;
            }
            return true;
        }

        bool ConvertResultLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeStub;
            if (layer.dst().empty())
                layer.dst().push_back(layer.name());
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

        bool ConvertSoftmaxLayer(const XmlNode* pLayer, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.softmax().axis());
            if (trans && !PermutedToNchw(layers, false, false))
            {
                Shape input = ConvertInputShape(pLayer);
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.softmax().axis() = (int32_t)nchw[layer.softmax().axis()];
                }
            }
            return true;
        }

        bool ConvertStridedSliceLayer(const XmlNode* pLayer, const LayerParams& layers, const Vector& srcBin, bool trans, LayerParam& layer)
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

        bool ConvertTransposeLayer(const XmlNode* pLayer, const Vector& srcBin, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape first = ConvertInputShape(pLayer);
            const LayerParam * second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
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
                if (trans)
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
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertUnsqueezeLayer(const XmlNode* pLayer, const Vector& srcBin, const LayerParams& layers, LayerParam& layer)
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

        static String ShapeToStr(const Shape& shape)
        {
            std::stringstream ss;
            ss << "{";
            for (size_t i = 0; i < shape.size(); ++i)
                ss << " " << shape[i];
            ss << " }";
            return ss.str();
        }

        static bool CompactShape(Shape& shape)
        {
            size_t count = 0, value = 1;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] != 1)
                {
                    value = shape[i];
                    count++;
                }
            }
            if (count > 1)
            {
                std::cout << "Can't compact shape " << ShapeToStr(shape) << " !" << std::endl;
                return false;
            }
            shape = Shape( { value } );
            return true;
        }

        static size_t TensorSize(const Shape & shape)
        {
            if (shape.empty())
                return 0;
            else
            {
                size_t size = 1;
                for (size_t i = 0; i < shape.size(); ++i)
                    size *= shape[i];
                return size;
            }
        }

        static bool CheckSourceNumber(const LayerParam& layer, size_t size)
        {
            if (layer.src().size() != size)
            {
                std::cout << "Wrong number of sources (" << layer.src().size() << " instead of " << size << ") !" << std::endl;
                return false;
            }
            return true;
        }

        static bool CheckDims(const Shape& shape, size_t dims, const String& desc)
        {
            if (shape.size() != dims)
            {
                std::cout << "Wrong " << desc << " shape " << ShapeToStr(shape) << " !" << std::endl;
                return false;
            }
            return true;
        }

        template<class T> static const T* GetWeight(const Vector & bin, size_t offset)
        {
            return (const T*)((const uint8_t*)bin.data() + offset);
        }

        template<class T> static const T * GetWeight(const Vector & bin, const WeightParam & param)
        {
            return GetWeight<T>(bin, param.offset());
        }

        static Pin ParsePin(const String& name)
        {
            Pin pin(name);
            size_t delimiter = name.find_first_of(":");
            if (delimiter != std::string::npos)
            {
                pin.name = name.substr(0, delimiter);
                std::istringstream(name.substr(delimiter + 1)) >> pin.index;
            }
            return pin;
        }

        static const LayerParam* GetLayer(const LayerParams& layers, const String& name)
        {
            Pin pin = ParsePin(name);
            for (size_t i = 0; i < layers.size(); ++i)
                if (pin.name == layers[i].name())
                    return &layers[i];
            std::cout << "Can't found layer " << pin.name << " !" << std::endl;
            return NULL;
        }

        static bool ReorderWeight(const Vector& srcBin, const Shape & input, LayerParam & layer, Vector& dstBin)
        {
            if (layer.weight().size() < 1)
            {
                std::cout << "There is no weight to reorder!" << std::endl;
                return false;
            }
            WeightParam& weight = layer.weight()[0];
            const float * pSrc = srcBin.data() + weight.offset() / sizeof(float);
            float * pDst = dstBin.data() + weight.offset() / sizeof(float);
            Shape & shape = weight.dim();
            weight.format() = TensorFormatNhwc;
            switch (layer.type())
            {
            case LayerTypeConvolution:
            {
                shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
                Tensor dst(pDst, weight.size() / sizeof(float), shape, weight.format());
                for (size_t o = 0; o < shape[3]; ++o)
                    for (size_t i = 0; i < shape[2]; ++i)
                        for (size_t y = 0; y < shape[0]; ++y)
                            for (size_t x = 0; x < shape[1]; ++x)
                                dst.CpuData(Shape({ y, x, i, o }))[0] = *pSrc++;
                break;
            }
            case LayerTypeInnerProduct:
            {
                for (size_t n = 0; n < shape[0]; n++)
                {
                    for (size_t c = 0; c < input[1]; c++)
                    {
                        for (size_t y = 0; y < input[2]; y++)
                        {
                            for (size_t x = 0; x < input[3]; x++)
                            {
                                size_t srcOffset = input[2] * input[3] * c + input[3] * y + x;
                                size_t dstOffset = input[3] * input[1] * y + input[1] * x + c;
                                pDst[dstOffset] = pSrc[srcOffset];
                            }
                        }
                    }
                    pSrc += input[1] * input[2] * input[3];
                    pDst += input[1] * input[2] * input[3];
                }
                break;
            }
            default:
                std::cout << "Unknsupported layer type " << ValueToString(layer.type())  << " to convert weight !" << std::endl;
                return false;
            }
            return true;
        }

        static bool PermutedToNchw(const LayerParams & layers, size_t current, bool checkInnerProduct, bool checkPriorBox)
        {
            const LayerParam& layer = layers[current];
            if (layer.type() == LayerTypeConvolution && layer.weight()[0].format() == TensorFormatNhwc)
                return false;
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNchw)
                return true;
            if (checkInnerProduct && layer.type() == LayerTypeInnerProduct)
                return true;
            if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
                return true;
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                Pin src = ParsePin(layer.src()[s]);
                for (size_t l = 0; l < current; ++l)
                {
                    if (src.name == layers[l].name() && layers[l].type() != LayerTypeMeta &&
                        PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox))
                        return true;
                }
            }
            return false;
        }

        static bool PermutedToNchw(const LayerParams& layers, bool checkInnerProduct, bool checkPriorBox)
        {
            size_t start = layers.size() - 1;
            if (layers[start].type() == LayerTypeConst && start)
                start--;
            return PermutedToNchw(layers, start, checkInnerProduct, checkPriorBox);
        }

        template<class T> bool GetShapeFromWeight(const WeightParam & weight, const Vector & bin, Shape & shape)
        {
            size_t size = weight.dim()[0];
            if (size != weight.size() / sizeof(T))
                return false;
            T * data = (T*)((uint8_t*)bin.data() + weight.offset());
            shape.resize(size);
            for (size_t i = 0; i < size; ++i)
                shape[i] = (size_t)data[i];
            return true;
        }

        bool GetShapeFromConst(const LayerParam & layer, const Vector & bin, bool trans, Shape & shape)
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
                    if (!GetShapeFromWeight<int32_t>(weight, bin, shape))
                        return false;
                case TensorType64i:
                    if (!GetShapeFromWeight<int64_t>(weight, bin, shape))
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
                        shape[i] = (size_t)alpha.i32()[i];
                }
                else if (alpha.type() == TensorType64i)
                {
                    for (size_t i = 0; i < shape.size(); ++i)
                        shape[i] = (size_t)alpha.i64()[i];
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
                    shape = Shp(shape[0], shape[2], shape[3], shape[1]);
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
    };
}
