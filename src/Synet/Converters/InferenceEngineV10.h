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
                if (type == "Add" && !ConvertAddLayer(pLayer, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Const" && !ConvertConstLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Convolution" && !ConvertConvolutionLayer(pLayer, srcBin, trans, dstXml.layers(), layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Multiply" && !ConvertMultiplyLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Parameter" && !ConvertParameterLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Result" && !ConvertResultLayer(pLayer, layer))
                    return ErrorMessage(pLayer);

#if 0
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

        bool ConvertAddLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (layer.src().size() != 2)
            {
                std::cout << "Wrong number of sources = " << layer.src().size() << " !" << std::endl;
                return false;
            }
            if (layer.src()[1] == layers.back().name() && layers.back().type() == LayerTypeConst)
            {
                layer.type() = Synet::LayerTypeBias;
                layer.weight() = layers.back().weight();
                layer.src().resize(1);
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
            }
            else
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeSum;
            }
            return true;
        }

        bool ConvertConstLayer(const XmlNode* pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeConst;
            const XmlNode* pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode* pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.weight().resize(1);
                    layer.weight()[0].dim() = ConvertShape(pPort);
                }
                else
                    return false;
            }
            else
                return false;
            const XmlNode* pData = pLayer->FirstNode("data");
            if (pData)
            {
                Shape shape;
                if(!ConvertVector(pData->FirstAttribute("shape"), shape) || shape != layer.weight()[0].dim())
                    return false;
                StringToValue(pData->FirstAttribute("offset")->Value(), layer.weight()[0].offset());
                StringToValue(pData->FirstAttribute("size")->Value(), layer.weight()[0].size());
                String type = pData->FirstAttribute("element_type")->Value();
                if (type == "f32")
                    layer.weight()[0].type() = TensorType32f;
                else if (type == "i64")
                    layer.weight()[0].type() = TensorType64i;
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

        bool ConvertConvolutionLayer(const XmlNode* pLayer, const Vector& srcBin, bool trans, const LayerParams& layers, LayerParam& layer, Vector& dstBin)
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
            if (!ConvertVectors(pData->FirstAttribute("pads_begin"), pData->FirstAttribute("pads_end"), layer.convolution().pad()))
                return false;

            //size_t inputNum;
            //const XmlNode* pInput = pLayer->FirstNode("input");
            //if (pInput)
            //{
            //    const XmlNode* pPort = pInput->FirstNode("port");
            //    if (pPort)
            //    {
            //        Shape input = ConvertShape(pPort);
            //        assert(input.size() == 4);
            //        inputNum = input[1];
            //    }
            //    else
            //        return false;
            //}
            //else
            //    return false;
            //const XmlNode* pOutput = pLayer->FirstNode("output");
            //if (pOutput)
            //{
            //    const XmlNode* pPort = pOutput->FirstNode("port");
            //    if (pPort)
            //    {
            //        Shape output = ConvertShape(pPort);
            //        assert(output.size() == 4);
            //        layer.convolution().outputNum() = (uint32_t)output[1];
            //    }
            //    else
            //        return false;
            //}
            //else
            //    return false;
            //const XmlNode* pBlobs = pLayer->FirstNode("blobs");
            //if (pBlobs)
            //{
            //    const XmlNode* pWeights = pBlobs->FirstNode("weights");
            //    if (pWeights)
            //    {
            //        size_t outputNum = layer.convolution().outputNum(), group = layer.convolution().group();
            //        size_t kernelY = layer.convolution().kernel()[0], kernelX = layer.convolution().kernel()[1];
            //        layer.weight().resize(1);
            //        if (layer.type() == Synet::LayerTypeConvolution)
            //        {
            //            Shape shape = Shape({ outputNum, inputNum / group, kernelY,  kernelX });
            //            if (trans)
            //            {
            //                shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
            //                layer.weight()[0].format() = TensorFormatNhwc;
            //            }
            //            layer.weight()[0].dim() = shape;
            //            ConvertWeight(pWeights, srcBin, trans ? 2 : 0, Shape(), layer.weight()[0], dstBin);
            //        }
            //        else
            //        {
            //            Shape shape = Shape({ inputNum, outputNum / group, kernelY,  kernelX });
            //            if (trans)
            //            {
            //                shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
            //                layer.weight()[0].format() = TensorFormatNhwc;
            //            }
            //            layer.weight()[0].dim() = shape;
            //            ConvertWeight(pWeights, srcBin, trans ? 1 : 0, Shape(), layer.weight()[0], dstBin);
            //        }
            //    }
            //    else
            //        return false;
            //    const XmlNode* pBiases = pBlobs->FirstNode("biases");
            //    if (pBiases)
            //    {
            //        layer.weight().resize(2);
            //        layer.weight()[1].dim() = Shape({ (size_t)layer.convolution().outputNum() });
            //        ConvertWeight(pBiases, srcBin, 0, Shape(), layer.weight()[1], dstBin);
            //        layer.convolution().biasTerm() = true;
            //    }
            //    else
            //        layer.convolution().biasTerm() = false;
            //}
            //else
            //    return false;

            return true;
        }

        bool ConvertMultiplyLayer(const XmlNode* pLayer, const std::vector<float>& srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (layer.src().size() != 2)
            {
                std::cout << "Wrong number of sources = " << layer.src().size() << " !" << std::endl;
                return false;
            }
            if (layer.src()[1] == layers.back().name() && layers.back().type() == LayerTypeConst)
            {
                if (TensorSize(layers.back().weight()[0].dim()) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().power() = 1.0f;
                    layer.power().scale() = GetWeight<float>(layers.back().weight()[0], srcBin)[0];
                    layer.power().shift() = 0.0f;
                }
                else
                {
                    layer.type() = Synet::LayerTypeScale;
                    layer.scale().biasTerm() = false;
                    layer.weight() = layers.back().weight();
                    if (!CompactShape(layer.weight()[0].dim()))
                        return false;
                }
                layer.src().resize(1);
            }
            else
                return false;
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

        bool ConvertReluLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertResultLayer(const XmlNode* pLayer, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeStub;
            if (layer.dst().empty())
                layer.dst().push_back(layer.name());
            return true;
        }

        //---------------------------------------------------------------------

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
                std::cout << "Can't compact shape {";
                for (size_t i = 0; i < shape.size(); ++i)
                    std::cout << " " << shape[i];
                std::cout << " } !" << std::endl;
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

        template<class T> static const T * GetWeight(const WeightParam & param, const Vector & bin)
        {
            return (const T*)((const uint8_t*)bin.data() + param.offset());
        }
    };
}
