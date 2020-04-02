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
                if ((type == "Convolution" || type == "GroupConvolution") && !ConvertConvolutionLayer(pLayer, trans, dstXml.layers(), layer, dstBin))
                    return ErrorMessage(pLayer);
                if (type == "Multiply" && !ConvertMultiplyLayer(pLayer, srcBin, dstXml.layers(), layer))
                    return ErrorMessage(pLayer);
                if (type == "Parameter" && !ConvertParameterLayer(pLayer, trans, layer))
                    return ErrorMessage(pLayer);
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Result" && !ConvertResultLayer(pLayer, layer))
                    return ErrorMessage(pLayer);
                if (type == "Transpose" && !ConvertTransposeLayer(pLayer, srcBin, dstXml.layers(), trans, layer))
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

        struct Pin
        {
            String name;
            int index;
            Pin(const String& n = String(), int i = 0) : name(n), index(i) {}
        };

        bool ConvertAddLayer(const XmlNode* pLayer, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeConst)
            {
                layer.type() = Synet::LayerTypeBias;
                layer.weight() = second->weight();
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

        bool ConvertConvolutionLayer(const XmlNode* pLayer, bool trans, const LayerParams& layers, LayerParam& layer, Vector& dstBin)
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
                layer.convolution().outputNum() = shape[0];
            }
            else
            {
                if (!CheckDims(shape, 5, "convolution weight"))
                    return false;
                layer.convolution().kernel() = Shape({ shape[3], shape[4] });
                layer.convolution().group() = shape[0];
                layer.convolution().outputNum() = shape[0] * shape[1];
                layer.weight()[0].dim() = Shape({ shape[0] * shape[1] , shape[2], shape[3], shape[4] });
            }
            layer.src().resize(1);
            if (trans)
                return ReorderWeight(Shape(), layer, dstBin);
            return true;
        }

        bool ConvertMultiplyLayer(const XmlNode* pLayer, const std::vector<float>& srcBin, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeConst)
            {
                if (TensorSize(layers.back().weight()[0].dim()) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().power() = 1.0f;
                    layer.power().scale() = GetWeight<float>(second->weight()[0], srcBin)[0];
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

        bool ConvertTransposeLayer(const XmlNode* pLayer, const std::vector<float>& srcBin, const LayerParams& layers, bool trans, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam * second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst || second->weight()[0].dim().size() != 1)
                return false;
            Shape input = ConvertInputShape(pLayer);
            if (!CheckDims(input, second->weight()[0].dim()[0], "order size"))
                return false;
            Shape& order = layer.permute().order();
            const int64_t * weight = GetWeight<int64_t>(second->weight()[0], srcBin);
            layer.type() = LayerTypePermute;
            order.resize(input.size());
            for (size_t i = 0; i < order.size(); ++i)
                order[i] = (size_t)weight[i];
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

        template<class T> static const T * GetWeight(const WeightParam & param, const Vector & bin)
        {
            return (const T*)((const uint8_t*)bin.data() + param.offset());
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

        static bool ReorderWeight(const Shape & input, LayerParam & layer, Vector& bin)
        {
            if (layer.weight().size() < 1)
            {
                std::cout << "There is no weight to reorder!" << std::endl;
                return false;
            }
            WeightParam& weight = layer.weight()[0];
            float * pDst = bin.data() + weight.offset() / sizeof(float);
            Vector buf(pDst, pDst + weight.size() / sizeof(float));
            const float * pSrc = buf.data();
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
            default:
                std::cout << "Unknsupported layer type " << ValueToString(layer.type())  << " to convert weight !" << std::endl;
                return false;
            }
            return true;
        }
    };
}
