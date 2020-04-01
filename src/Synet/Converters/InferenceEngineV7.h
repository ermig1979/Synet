/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#pragma once

#include "Synet/Converters/InferenceEngineBase.h"

namespace Synet
{
    class InferenceEngineConverterV7 : public InferenceEngineConverter
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
                if (type == "Activation" && !ConvertActivationLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Clamp" && !ConvertClampLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Const" && !ConvertConstLayer(pLayer, srcBin, trans, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Concat" && !ConvertConcatLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if ((type == "Convolution" || type == "Deconvolution") && !ConvertConvolutionOrDeconvolutionLayer(pLayer, srcBin, trans, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "CTCGreedyDecoder" && !ConvertCtcGreedyDecoderLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "DetectionOutput" && !ConvertDetectionOutputLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Eltwise" && !ConvertEltwiseLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Flatten" && !ConvertFlattenLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "FullyConnected" && !ConvertFullyConnectedLayer(pLayer, pPrevLayer, srcBin, trans, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Input" && !ConvertInputLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Interp" && !ConvertInterpLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Log" && !ConvertLogLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Permute" && !ConvertPermuteLayer(pLayer, pPrevLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Pooling" && !ConvertPoolingLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Power" && !ConvertPowerLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PReLU" && !ConvertPreluLayer(pLayer, srcBin, trans, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "PriorBox" && !ConvertPriorBoxLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxClustered" && !ConvertPriorBoxClusteredLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "PriorBoxV2" && !ConvertPriorBoxV2Layer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Elu" && !ConvertEluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Resample" && !ConvertResampleLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Reshape" && !ConvertReshapeLayer(pLayer, trans, layer, dstXml.layers()))
                    return ErrorMessage(pLayer);
                if (type == "ScaleShift" && !ConvertScaleShiftLayer(pLayer, srcBin, layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "SoftMax" && !ConvertSoftmaxLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Split" && !ConvertSplitLayer(pLayer, trans, layer, dstXml.layers()))
                    return ErrorMessage(pLayer);
                if (type == "Squeeze" && !ConvertSqueezeLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "StridedSlice" && !ConvertStridedSliceLayer(pLayer, layer, dstXml.layers(), dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Tile" && !ConvertTileLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "Unsqueeze" && !ConvertUnsqueezeLayer(pLayer, layer))
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

            const XmlNode* pStatistics = srcXml.FirstNode("statistics");
            if (pStatistics)
            {
                const XmlNode* pLayer = pStatistics->FirstNode("layer");
                while (pLayer)
                {
                    StatisticParam statistic;
                    const XmlNode* pName = pLayer->FirstNode("name");
                    if (pName && ConvertVector(pLayer->FirstNode("min"), statistic.min()) && ConvertVector(pLayer->FirstNode("max"), statistic.max()))
                        statistic.name() = pName->Value();
                    else
                    {
                        std::cout << "Can't load statistics! " << std::endl;
                        return false;
                    }
                    dstXml.statistics().push_back(statistic);
                    pLayer = pLayer->NextSibling("layer");
                }
            }

            return true;
        }

    private:

        TensorInfoMap _tensors;
        LayerParamMap _layers;

        static void ConvertWeight(const XmlNode * pNode, const Vector & srcBin, int mode, const Shape & input, WeightParam & param, Vector & dstBin)
        {
            const Shape & shape = param.dim();
            StringToValue(pNode->FirstAttribute("offset")->Value(), param.offset());
            StringToValue(pNode->FirstAttribute("size")->Value(), param.size());
            const float * pSrc = srcBin.data() + param.offset() / sizeof(float);
            float * pDst = dstBin.data() + param.offset() / sizeof(float);
            Tensor dst(pDst, param.size() / sizeof(float), shape, param.format());
            switch (mode)
            {
            case 0:
                memcpy(pDst, pSrc, param.size());
                break;
            case 1:
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t c = 0; c < shape[3]; ++c)
                        for (size_t y = 0; y < shape[1]; ++y)
                            for (size_t x = 0; x < shape[2]; ++x)
                                dst.CpuData(Shape({ i, y, x, c }))[0] = *pSrc++;
                break;
            case 2:
                for (size_t o = 0; o < shape[3]; ++o)
                    for (size_t i = 0; i < shape[2]; ++i)
                        for (size_t y = 0; y < shape[0]; ++y)
                            for (size_t x = 0; x < shape[1]; ++x)
                                dst.CpuData(Shape({ y, x, i, o }))[0] = *pSrc++;
                break;
            case 3:
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
            default:
                assert(0);
            }
        }

        bool ConvertActivationLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String type = pData->FirstAttribute("type")->Value();
            if (type == "sigmoid")
                layer.type() = Synet::LayerTypeSigmoid;
            else if (type == "elu")
                layer.type() = Synet::LayerTypeElu;
            else if (type == "exp")
            {
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = UnaryOperationTypeExp;
                return true;
            }
            else if (type == "tanh")
            {
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = UnaryOperationTypeTanh;
                return true;
            }
            else
                return false;
            return true;
        }

        bool ConvertClampLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("min")->Value(), layer.restrictRange().lower());
            StringToValue(pData->FirstAttribute("max")->Value(), layer.restrictRange().upper());
            return true;
        }

        bool ConvertConcatLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeConcat;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.concat().axis());
            if (trans)
            {
                Shape input = ConvertInputShape(pLayer);
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                }
                //if (input.size() == 3 && layer.concat().axis() == 1)
                //    layer.concat().axis() = 2;
            }
            return true;
        }

        bool ConvertConstLayer(const XmlNode * pLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypeConst;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            Shape shape;
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.weight().resize(1);
                    shape = ConvertShape(pPort);
                    if (trans && shape.size() == 4)
                    {
                        shape = Shape({shape[0], shape[2], shape[3], shape[1]});
                        layer.weight()[0].format() = TensorFormatNhwc;
                    }
                    layer.weight()[0].dim() = shape;
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pCustom = pBlobs->FirstNode("custom");
                if (pCustom)
                    ConvertWeight(pCustom, srcBin, trans && shape.size() == 4 ? 1 : 0, Shape(), layer.weight()[0], dstBin);
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        bool ConvertConvolutionOrDeconvolutionLayer(const XmlNode * pLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            String type = pLayer->FirstAttribute("type")->Value();
            if (type == "Convolution")
                layer.type() = Synet::LayerTypeConvolution;
            else if (type == "Deconvolution")
                layer.type() = Synet::LayerTypeDeconvolution;
            else
                return false;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (pData->FirstAttribute("quantization_level") && pData->FirstAttribute("quantization_level")->Value() == String("I8"))
                layer.convolution().quantizationLevel() = TensorType8i;
            if (pData->FirstAttribute("kernel-y") && pData->FirstAttribute("kernel-x"))
            {
                layer.convolution().kernel().resize(2);
                StringToValue(pData->FirstAttribute("kernel-y")->Value(), layer.convolution().kernel()[0]);
                StringToValue(pData->FirstAttribute("kernel-x")->Value(), layer.convolution().kernel()[1]);
            }
            else if (!ConvertVector(pData->FirstAttribute("kernel"), layer.convolution().kernel()))
                return false;
            if (pData->FirstAttribute("stride-y") && pData->FirstAttribute("stride-x"))
            {
                layer.convolution().stride().resize(2);
                StringToValue(pData->FirstAttribute("stride-y")->Value(), layer.convolution().stride()[0]);
                StringToValue(pData->FirstAttribute("stride-x")->Value(), layer.convolution().stride()[1]);
            }
            else if (!ConvertVector(pData->FirstAttribute("strides"), layer.convolution().stride()))
                return false;
            if (pData->FirstAttribute("dilation-y") && pData->FirstAttribute("dilation-x"))
            {
                layer.convolution().dilation().resize(2);
                StringToValue(pData->FirstAttribute("dilation-y")->Value(), layer.convolution().dilation()[0]);
                StringToValue(pData->FirstAttribute("dilation-x")->Value(), layer.convolution().dilation()[1]);
            }
            else if (!ConvertVector(pData->FirstAttribute("dilations"), layer.convolution().dilation()))
                return false;
            StringToValue(pData->FirstAttribute("group")->Value(), layer.convolution().group());
            if (pData->FirstAttribute("pad-y") && pData->FirstAttribute("pad-x") && pData->FirstAttribute("pad-b") && pData->FirstAttribute("pad-r"))
            {
                layer.convolution().pad().resize(4);
                StringToValue(pData->FirstAttribute("pad-y")->Value(), layer.convolution().pad()[0]);
                StringToValue(pData->FirstAttribute("pad-x")->Value(), layer.convolution().pad()[1]);
                StringToValue(pData->FirstAttribute("pad-b")->Value(), layer.convolution().pad()[2]);
                StringToValue(pData->FirstAttribute("pad-r")->Value(), layer.convolution().pad()[3]);
            }
            else if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.convolution().pad()))
                return false;
            size_t inputNum;
            const XmlNode * pInput = pLayer->FirstNode("input");
            if (pInput)
            {
                const XmlNode * pPort = pInput->FirstNode("port");
                if (pPort)
                {
                    Shape input = ConvertShape(pPort);
                    assert(input.size() == 4);
                    inputNum = input[1];
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    Shape output = ConvertShape(pPort);
                    assert(output.size() == 4);
                    layer.convolution().outputNum() = (uint32_t)output[1];
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pWeights = pBlobs->FirstNode("weights");
                if (pWeights)
                {
                    size_t outputNum = layer.convolution().outputNum(), group = layer.convolution().group();
                    size_t kernelY = layer.convolution().kernel()[0], kernelX = layer.convolution().kernel()[1];
                    layer.weight().resize(1);
                    if(layer.type() == Synet::LayerTypeConvolution)
                    { 
                        Shape shape = Shape({ outputNum, inputNum / group, kernelY,  kernelX});
                        if (trans)
                        {
                            shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
                            layer.weight()[0].format() = TensorFormatNhwc;
                        }
                        layer.weight()[0].dim() = shape;
                        ConvertWeight(pWeights, srcBin, trans ? 2 : 0, Shape(), layer.weight()[0], dstBin);                    
                    }
                    else
                    {
                        Shape shape = Shape({ inputNum, outputNum / group, kernelY,  kernelX });
                        if (trans)
                        {
                            shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                            layer.weight()[0].format() = TensorFormatNhwc;
                        }
                        layer.weight()[0].dim() = shape;
                        ConvertWeight(pWeights, srcBin, trans ? 1 : 0, Shape(), layer.weight()[0], dstBin);
                    }
                }
                else
                    return false;
                const XmlNode * pBiases = pBlobs->FirstNode("biases");
                if (pBiases)
                {
                    layer.weight().resize(2);
                    layer.weight()[1].dim() = Shape({ (size_t)layer.convolution().outputNum() });
                    ConvertWeight(pBiases, srcBin, 0, Shape(), layer.weight()[1], dstBin);
                    layer.convolution().biasTerm() = true;
                }
                else
                    layer.convolution().biasTerm() = false;
            }
            else
                return false;

            return true;
        }

        bool ConvertDetectionOutputLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeDetectionOutput;
            const XmlNode * pData = pLayer->FirstNode("data");
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
            if(pData->FirstAttribute("clip"))
                StringToValue(pData->FirstAttribute("clip")->Value(), layer.detectionOutput().clip());

            return true;
        }

        bool ConvertEltwiseLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeEltwise;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String operation = pData->FirstAttribute("operation")->Value();
            if (operation == "sum")
            {
                layer.eltwise().operation() = EltwiseOperationTypeSum;
                if (pData->FirstAttribute("coeff") && pData->FirstAttribute("coeff")->Value() != String())
                    ConvertVector(pData->FirstAttribute("coeff"), layer.eltwise().coefficients());
            }
            else if (operation == "max")
                layer.eltwise().operation() = EltwiseOperationTypeMax;
            else if (operation == "min")
                layer.eltwise().operation() = EltwiseOperationTypeMin;
            else if (operation == "mul")
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
            else
                assert(0);
            return true;
        }

        bool ConvertCtcGreedyDecoderLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeCtcGreedyDecoder;
            return true;
        }

        bool ConvertFlattenLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeFlatten;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.flatten().axis());
            StringToValue(pData->FirstAttribute("end_axis")->Value(), layer.flatten().endAxis());
            return true;
        }

        bool ConvertFullyConnectedLayer(const XmlNode * pLayer, const XmlNode * pPrevLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (pData->FirstAttribute("quantization_level") && pData->FirstAttribute("quantization_level")->Value() == String("I8"))
                layer.convolution().quantizationLevel() = TensorType8i;
            StringToValue(pData->FirstAttribute("out-size")->Value(), layer.innerProduct().outputNum());
            Shape inputShape = ConvertInputShape(pLayer);
            size_t inputSize = 1;
            for (size_t i = 0; i < inputShape.size(); ++i)
                inputSize *= inputShape[i];
            if (pPrevLayer && String(pPrevLayer->FirstAttribute("type")->Value()) == "Reshape")
            {
                inputShape = ConvertInputShape(pPrevLayer);
                size_t inputSize = 1;
                for (size_t i = 0; i < inputShape.size(); ++i)
                    inputSize *= inputShape[i];
            }
            Shape outputShape = ConvertOutputShape(pLayer);
            assert(outputShape.size() == 2 && layer.innerProduct().outputNum() == outputShape[1]);
            layer.weight().resize(1);
            layer.weight()[0].dim() = Shape({ (size_t)layer.innerProduct().outputNum(), inputSize });
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pWeights = pBlobs->FirstNode("weights");
                if (pWeights)
                    ConvertWeight(pWeights, srcBin, trans && inputShape.size() == 4 ? 3 : 0, inputShape, layer.weight()[0], dstBin);
                else
                    return false;
                const XmlNode * pBiases = pBlobs->FirstNode("biases");
                if (pBiases)
                {
                    layer.weight().resize(2);
                    layer.weight()[1].dim() = Shape({ (size_t)layer.innerProduct().outputNum() });
                    ConvertWeight(pBiases, srcBin, 0, Shape(), layer.weight()[1], dstBin);
                    layer.innerProduct().biasTerm() = true;
                }
                else
                    layer.innerProduct().biasTerm() = false;
            }
            else
                return false;

            return true;
        }

        bool ConvertInputLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeInput;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.input().shape().resize(1);
                    Shape shape = ConvertShape(pPort);
                    if (trans)
                    {
                        if (shape.size() == 4)
                            shape = Shape({shape[0], shape[2], shape[3], shape[1]});
                        layer.input().shape()[0].format() = TensorFormatNhwc;
                    }
                    layer.input().shape()[0].dim() = shape;
                }
            }
            return true;
        }

        bool ConvertInterpLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = LayerTypeInterp2;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr * pFactor = pData->FirstAttribute("factor");
            if (pFactor == NULL)
                return false;
            StringToValue(pFactor->Value(), layer.interp2().factor());
            const XmlAttr * pPadBeg = pData->FirstAttribute("pad_beg");
            const XmlAttr * pPadEnd = pData->FirstAttribute("pad_end");
            if (pPadBeg && pPadEnd)
            {
                size_t padBeg, padEnd;
                StringToValue(pPadBeg->Value(), padBeg);
                StringToValue(pPadEnd->Value(), padEnd);
                layer.interp2().pad() = Shape({padBeg, padBeg, padEnd, padEnd});
            }
            if (pData->FirstAttribute("height"))
                StringToValue(pData->FirstAttribute("height")->Value(), layer.interp2().height());
            if (pData->FirstAttribute("width"))
                StringToValue(pData->FirstAttribute("width")->Value(), layer.interp2().width());
            if (pData->FirstAttribute("align_corners"))
                StringToValue(pData->FirstAttribute("align_corners")->Value(), layer.interp2().alignCorners());
            return true;
        }

        bool ConvertLogLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeLog;
            return true;
        }

        bool ConvertPermuteLayer(const XmlNode * pLayer, const XmlNode * pPrevLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypePermute;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr * pOrder = pData->FirstAttribute("order");
            if (pOrder == NULL)
                return false;
            Shape order = ConvertShape(pOrder->Value());
            if (trans && order.size() == 5)
            {
                Shape nhwcc = Shape({ 0, 3, 4, 1, 2 });
                Shape ncchw = Shape({ 0, 3, 4, 1, 2 });
                order = Shape({ ncchw[order[nhwcc[0]]], ncchw[order[nhwcc[1]]], ncchw[order[nhwcc[2]]], ncchw[order[nhwcc[3]]], ncchw[order[nhwcc[4]]] });
            }
            else if (trans && order.size() == 4)
            {
                bool reorderChannels = false;
                if (pPrevLayer && String(pPrevLayer->FirstAttribute("type")->Value()) == "Reshape")
                {
                    Shape i = ConvertInputShape(pPrevLayer);
                    Shape o = ConvertOutputShape(pPrevLayer);
                    if(order == Shape({ 0, 2, 1, 3 }) && i.size() == 4 && o.size() == 4 && o[1] * o[2] == i[1] && o[3] == i[2] * i[3])
                        reorderChannels = true;
                }
                if (reorderChannels)
                    order = Shape({ 0, 1, 3, 2 });
                else if (order == Shape({ 0, 2, 3, 1 }))
                {
                    order = Shape({ 0, 1, 2, 3 });
                    layer.permute().format() = TensorFormatNchw;
                }
                else if (order == Shape({ 3, 0, 1, 2 }))
                {
                    order = Shape({ 2, 3, 1, 0 });
                    layer.permute().format() = TensorFormatNchw;
                }
                else
                {
                    Shape nhwc = Shape({ 0, 2, 3, 1 });
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    order = Shape({ nchw[order[nhwc[0]]], nchw[order[nhwc[1]]], nchw[order[nhwc[2]]], nchw[order[nhwc[3]]] });
                }
            }
            else if(trans && order.size() == 3)
            {
                bool reorderChannels = false;
                if (pPrevLayer && String(pPrevLayer->FirstAttribute("type")->Value()) == "Reshape")
                {
                    Shape i = ConvertInputShape(pPrevLayer);
                    Shape o = ConvertOutputShape(pPrevLayer);
                    if (order == Shape({ 1, 0, 2 }) && i.size() == 4 && o.size() == 3 && o[0] * o[1] == i[1] && o[2] == i[2] * i[3])
                        reorderChannels = true;
                }
                if (reorderChannels)
                    order = Shape({ 0, 2, 1 });
            }
            layer.permute().order() = order;
            return true;
        }

        bool ConvertPoolingLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypePooling;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String method = pData->FirstAttribute("pool-method")->Value();
            if (method == "max")
                layer.pooling().method() = PoolingMethodTypeMax;
            else if (method == "avg")
                layer.pooling().method() = PoolingMethodTypeAverage;
            else
                assert(0);
            if (pData->FirstAttribute("kernel-y") && pData->FirstAttribute("kernel-x"))
            {
                layer.pooling().kernel().resize(2);
                StringToValue(pData->FirstAttribute("kernel-y")->Value(), layer.pooling().kernel()[0]);
                StringToValue(pData->FirstAttribute("kernel-x")->Value(), layer.pooling().kernel()[1]);
            }
            else if (!ConvertVector(pData->FirstAttribute("kernel"), layer.pooling().kernel()))
                return false;
            if (pData->FirstAttribute("stride-y") && pData->FirstAttribute("stride-x"))
            {
                layer.pooling().stride().resize(2);
                StringToValue(pData->FirstAttribute("stride-y")->Value(), layer.pooling().stride()[0]);
                StringToValue(pData->FirstAttribute("stride-x")->Value(), layer.pooling().stride()[1]);
            }
            else if (!ConvertVector(pData->FirstAttribute("strides"), layer.pooling().stride()))
                return false;
            if (pData->FirstAttribute("pad-y") && pData->FirstAttribute("pad-x") && pData->FirstAttribute("pad-b") && pData->FirstAttribute("pad-r"))
            {
                layer.pooling().pad().resize(4);
                StringToValue(pData->FirstAttribute("pad-y")->Value(), layer.pooling().pad()[0]);
                StringToValue(pData->FirstAttribute("pad-x")->Value(), layer.pooling().pad()[1]);
                StringToValue(pData->FirstAttribute("pad-b")->Value(), layer.pooling().pad()[2]);
                StringToValue(pData->FirstAttribute("pad-r")->Value(), layer.pooling().pad()[3]);
            }
            else if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.pooling().pad()))
                return false;
            const XmlAttr * pRoundingType = pData->FirstAttribute("rounding_type");
            if (pRoundingType && String(pRoundingType->Value()) == "floor")
                layer.pooling().roundingType() = RoundingTypeFloor;
            const XmlAttr * pAutoPad = pData->FirstAttribute("auto_pad");
            if (pAutoPad && String(pAutoPad->Value()) == "valid")
                layer.pooling().roundingType() = RoundingTypeFloor;
            if (pData->FirstAttribute("exclude-pad"))
                StringToValue(pData->FirstAttribute("exclude-pad")->Value(), layer.pooling().excludePad());
            return true;
        }

        bool ConvertPowerLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePower;
            StringToValue(pData->FirstAttribute("power")->Value(), layer.power().power());
            StringToValue(pData->FirstAttribute("scale")->Value(), layer.power().scale());
            StringToValue(pData->FirstAttribute("shift")->Value(), layer.power().shift());
            return true;
        }

        bool ConvertPreluLayer(const XmlNode * pLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypePrelu;
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pWeights = pBlobs->FirstNode("weights");
                if (pWeights)
                {
                    layer.weight().resize(1);
                    const XmlNode * pData = pLayer->FirstNode("data");
                    bool channelShared = false;
                    if (pData && pData->FirstAttribute("channel_shared"))
                        StringToValue(pData->FirstAttribute("channel_shared")->Value(), channelShared);
                    layer.weight()[0].dim() = Shape({ channelShared ? size_t{1} : ConvertInputShape(pLayer)[1] });
                    ConvertWeight(pWeights, srcBin, 0, Shape(), layer.weight()[0], dstBin);
                }
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        bool ConvertPriorBoxLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
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

        bool ConvertPriorBoxClusteredLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
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

        bool ConvertPriorBoxV2Layer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.type() = Synet::LayerTypePriorBox;
            layer.priorBox().version() = 2;
            StringToValue(pData->FirstAttribute("clip")->Value(), layer.priorBox().clip());
            StringToValue(pData->FirstAttribute("flip")->Value(), layer.priorBox().flip());
            StringToValue(pData->FirstAttribute("offset")->Value(), layer.priorBox().offset());
            if(pData->FirstAttribute("scale_all_sizes"))
                StringToValue(pData->FirstAttribute("scale_all_sizes")->Value(), layer.priorBox().scaleAllSizes());
            ConvertVector(pData->FirstAttribute("aspect_ratio"), layer.priorBox().aspectRatio());
            ConvertVector(pData->FirstAttribute("max_size"), layer.priorBox().maxSize());
            ConvertVector(pData->FirstAttribute("min_size"), layer.priorBox().minSize());
            ConvertVector(pData->FirstAttribute("variance"), layer.priorBox().variance());
            return true;
        }

        bool ConvertReluLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData)
            {
                if (pData->FirstAttribute("negative_slope"))
                    StringToValue(pData->FirstAttribute("negative_slope")->Value(), layer.relu().negativeSlope());
            }
            return true;
        }

        bool ConvertEluLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeElu;
            return true;
        }

        bool ConvertResampleLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeInterp;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    Shape output = ConvertShape(pPort);
                    if (output.size() == 4)
                    {
                        layer.interp().height() = (int)output[2];
                        layer.interp().width() = (int)output[3];
                    }
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData)
            {
                const XmlAttr * pType = pData->FirstAttribute("type");
                if (pType)
                {
                    if (String(pType->Value()) == "caffe.ResampleParameter.NEAREST")
                        layer.interp().interpolationType() = InterpolationTypeNearest;
                }
            }
            return true;
        }

        bool PermutedToNchw(const LayerParams & layers, bool checkInnerProduct = true)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                if (layers[i].type() == LayerTypePermute && layers[i].permute().format() == TensorFormatNchw)
                    return true;
                if (checkInnerProduct && layers[i].type() == LayerTypeInnerProduct)
                    return true;
            }
            return false;
        }

        String ClearLayerName(String name)
        {
            size_t delimiter = name.find_first_of(":");
            if (delimiter != std::string::npos)
                name = name.substr(0, delimiter);
            return name;
        }

        const LayerParam & PrevLayer(const LayerParam & layer)
        {
            return _layers[ClearLayerName(layer.src()[0])];
        }

        bool ConvertReshapeLayer(const XmlNode * pLayer, bool trans, LayerParam & layer, LayerParams & layers)
        {
            layer.type() = Synet::LayerTypeReshape;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    Shape input = ConvertInputShape(pLayer);
                    Shape output = ConvertShape(pPort);
                    if (trans && input.size() == 4 && output.size() == 5)// && !PermutedToNchw(layers, false))
                    {
                        output = Shape({ output[0], output[3] , output[4] , output[1], output[2] });
                    }
                    if (trans && input.size() == 5 && output.size() == 4)// && !PermutedToNchw(layers, false))
                    {
                        output = Shape({ output[0], output[2] , output[3], output[1] });
                    }
                    else if (trans && output.size() == 4 && !PermutedToNchw(layers, false))
                    {
                        if (input.size() > 3 && output[1]*output[2] == input[1] && output[3] == input[2]*input[3])
                            output = Shape({ output[0], output[3] , output[1] , output[2] });
                        else
                            output = Shape({ output[0], output[2] , output[3] , output[1] });
                    }
                    if (trans && output.size() == 3 && !PermutedToNchw(layers))
                    {
                        if (input.size() > 3 && output[0] * output[1] == input[1] && output[2] == input[2] * input[3])
                            output = Shape({ output[2] , output[0] , output[1] });
                        else
                            output = Shape({ output[1] , output[2] , output[0] });
                    }
                    if (input.size() > 1 && input[0] == 1 && output.size() >= 1 && output[0] == 1 && PrevLayer(layer).type() != LayerTypeUnpack)
                    {
                        layer.reshape().axis() = 1;
                        output.erase(output.begin(), output.begin() + 1);
                    }
                    if ((!trans || (trans && PermutedToNchw(layers))) && layer.reshape().axis() == 1)
                    {
                        if (output.size() == 1)
                            output[0] = -1;
                        if (output.size() == 2)
                        {
                            if(output[0] == 1)
                                output[1] = -1;
                            else
                                output[0] = -1;
                        }
                        if (output.size() == 3 && output[0] == 1)
                            output[1] = -1;
                    }
                    layer.reshape().shape() = output;
                }
                else
                    return false;
            }
            else
                return false;
            if (layer.src().size() == 2)
            {
                bool found = false;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    if (layer.src()[1] == layers[i].name())
                    {
                        if (layers[i].type() != LayerTypeConst)
                            return false;
                        layer.src().pop_back();
                        found = true;
                        break;
                    }
                }
                if (!found)
                    return false;
            }
            return true;
        }

        bool ConvertScaleShiftLayer(const XmlNode * pLayer, const Vector & srcBin, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypeScale;
            size_t channels;
            const XmlNode * pInput = pLayer->FirstNode("input");
            if (pInput)
            {
                const XmlNode * pPort = pInput->FirstNode("port");
                if (pPort)
                {
                    Shape input = ConvertShape(pPort);
                    assert(input.size() >= 2);
                    channels = input[1];
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    Shape output = ConvertShape(pPort);
                    assert(output.size() >= 2 && channels == output[1]);
                }
                else
                    return false;
            }
            else
                return false;
            layer.weight().resize(1);
            layer.weight()[0].dim() = Shape({ channels });
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pWeights = pBlobs->FirstNode("weights");
                if (pWeights)
                    ConvertWeight(pWeights, srcBin, 0, Shape(), layer.weight()[0], dstBin);
                else
                    return false;
                const XmlNode * pBiases = pBlobs->FirstNode("biases");
                if (pBiases)
                {
                    layer.weight().resize(2);
                    layer.weight()[1].dim() = Shape({ channels });
                    ConvertWeight(pBiases, srcBin, 0, Shape(), layer.weight()[1], dstBin);
                    layer.scale().biasTerm() = true;
                }
                else
                    layer.scale().biasTerm() = false;
            }
            else
                return false;

            return true;
        }

        bool ConvertSoftmaxLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.softmax().axis());
            if (trans)
            {
                Shape input;
                const XmlNode * pInput = pLayer->FirstNode("input");
                if (pInput)
                {
                    const XmlNode * pPort = pInput->FirstNode("port");
                    if (pPort)
                        input = ConvertShape(pPort);
                    else
                        return false;
                }
                else
                    return false;
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.softmax().axis() = (int32_t)nchw[layer.softmax().axis()];
                }
                //if (input.size() == 2)
                //{
                //    Shape cs = Shape({ 1, 0 });
                //    layer.softmax().axis() = (int32_t)cs[layer.softmax().axis()];
                //}
            }
            return true;
        }

        bool ConvertSplitLayer(const XmlNode * pLayer, bool trans, LayerParam & layer, const LayerParams & layers)
        {
            layer.type() = Synet::LayerTypeUnpack;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.unpack().axis());
            if (trans && !PermutedToNchw(layers))
            {
                Shape input;
                const XmlNode * pInput = pLayer->FirstNode("input");
                if (pInput)
                {
                    const XmlNode * pPort = pInput->FirstNode("port");
                    if (pPort)
                        input = ConvertShape(pPort);
                    else
                        return false;
                }
                else
                    return false;
                if (input.size() == 4)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                }
                if (input.size() == 3)
                {
                    if (PrevLayer(layer).type() == LayerTypePermute)
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
            return true;
        }

        bool ConvertSqueezeLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeSqueeze;
            return true;
        }

        bool GetShapeFromConst(const LayerParam & layer, const Vector & bin, Shape & shape)
        {
            if (layer.type() != LayerTypeConst)
                return false;
            if (layer.weight().size() != 1)
                return false;
            const WeightParam & weight = layer.weight()[0];
            if (weight.dim().size() != 1)
                return false;
            size_t size = weight.dim()[0];
            if (size != weight.size() / 4)
                return false;
            int32_t * data = (int32_t*)((uint8_t*)bin.data() + weight.offset());
            shape.resize(size);
            for (size_t i = 0; i < size; ++i)
                shape[i] = data[i];
            return true;
        }

        bool ConvertStridedSliceLayer(const XmlNode * pLayer, LayerParam & layer, LayerParams & layers, const Vector & bin)
        {
            layer.type() = Synet::LayerTypeStridedSlice;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            if (!ConvertVector(pData->FirstAttribute("begin_mask"), layer.stridedSlice().beginMask()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("ellipsis_mask"), layer.stridedSlice().ellipsisMask()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("end_mask"), layer.stridedSlice().endMask()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("new_axis_mask"), layer.stridedSlice().newAxisMask()))
                return false;
            if (!ConvertVector(pData->FirstAttribute("shrink_axis_mask"), layer.stridedSlice().shrinkAxisMask()))
                return false;
            for (size_t l = layer.src().size() - 1; l > 0; --l)
            {
                bool found = false;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    if (layer.src()[l] == layers[i].name())
                    {
                        switch (l)
                        {
                        case 1:
                            found = GetShapeFromConst(layers[i], bin, layer.stridedSlice().beginDims());
                            break;
                        case 2:
                            found = GetShapeFromConst(layers[i], bin, layer.stridedSlice().endDims());
                            break;
                        case 3:
                            found = GetShapeFromConst(layers[i], bin, layer.stridedSlice().strideDims());
                            break;
                        }
                        layer.src().pop_back();
                        break;
                    }
                }
                if (!found)
                    return false;
            }
            return true;
        }

        bool ConvertTileLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeTile;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.tile().axis());
            StringToValue(pData->FirstAttribute("tiles")->Value(), layer.tile().tiles());
            if (trans && ConvertInputShape(pLayer).size() == 4)
            {
                uint32_t order[4] = { 0, 3, 1, 2 };
                layer.tile().axis() = order[layer.tile().axis()];
            }
            return true;
        }

        bool ConvertUnsqueezeLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeExpandDims;
            Shape input = ConvertInputShape(pLayer);
            Shape output = ConvertOutputShape(pLayer);
            size_t axis = 0;
            for (; axis < input.size(); ++axis)
                if (input[axis] != 1 && output[axis] == 1)
                    break;
            layer.expandDims().axis() = (int32_t)axis;
            return true;
        }
    };
}
