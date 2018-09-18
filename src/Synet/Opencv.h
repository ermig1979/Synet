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

#if defined(SYNET_OPENCV_ENABLE)

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace Synet
{
    class OpencvToSynet
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

            XmlDoc xml;
            XmlFile file;
            if (!LoadModel(srcModelPath, file, xml))
            {
                std::cout << "Can't load opencv::dnn model '" << srcModelPath << "' !" << std::endl;
                return false;
            }

            Vector bin;
            if (!LoadWeight(srcWeightPath, bin))
            {
                std::cout << "Can't load opencv::dnn weight '" << srcWeightPath << "' !" << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Tensors weight;
            if (!ConvertNetwork(xml, bin, holder(), weight))
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
        typedef Xml::File<char> XmlFile;
        typedef Xml::XmlDocument<char> XmlDoc;
        typedef Xml::XmlNode<char> XmlNode;
        typedef Xml::XmlAttribute<char> XmlAttr;

        struct Edge
        {
            size_t fromLayer, fromPort, toLayer, toPort;
        };
        typedef std::vector<Edge> Edges;

        bool LoadModel(const String & path, XmlFile & file, XmlDoc & xml)
        {
            if (file.Open(path.c_str()))
            {
                try
                {
                    xml.Parse<0>(file.Data());
                }
                catch (std::exception e)
                {
                    return false;
                }
            }
            return true;
        }

        bool LoadWeight(const String & path, Vector & bin)
        {
            std::ifstream ifs(path.c_str(), std::ofstream::binary);
            if (!ifs.is_open())
                return false;
            size_t beg = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            size_t end = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            size_t size = (end - beg) / sizeof(float);
            bin.resize(size);
            ifs.read((char*)bin.data(), size * sizeof(float));
            ifs.close();
            return true;
        }

        bool ConvertNetwork(const XmlDoc & xml, const Vector & bin, Synet::NetworkParam & network, Tensors & weight)
        {
            const XmlNode * pNet = xml.FirstNode("net");
            if (pNet == NULL)
                return false;

            const XmlAttr * pName = pNet->FirstAttribute("name");
            if (pName)
                network.name() = pName->Value();

            Edges edges;
            const XmlNode * pEdges = pNet->FirstNode("edges");
            if (pEdges == NULL)
                return false;
            const XmlNode * pEdge = pEdges->FirstNode("edge");
            while (pEdge)
            {
                Edge edge;
                StringToValue(pEdge->FirstAttribute("from-layer")->Value(), edge.fromLayer);
                StringToValue(pEdge->FirstAttribute("from-port")->Value(), edge.fromPort);
                StringToValue(pEdge->FirstAttribute("to-layer")->Value(), edge.toLayer);
                StringToValue(pEdge->FirstAttribute("to-port")->Value(), edge.toPort);
                edges.push_back(edge);
                pEdge = pEdge->NextSibling("edge");
            }

            const XmlNode * pLayers = pNet->FirstNode("layers");
            if (pLayers == NULL)
                return false;
            const XmlNode * pLayer = pLayers->FirstNode("layer");
            while (pLayer)
            {
                size_t layerId;
                StringToValue(pLayer->FirstAttribute("id")->Value(), layerId);
                assert(layerId == network.layers().size());

                LayerParam layer;
                layer.name() = pLayer->FirstAttribute("name")->Value();

                const XmlNode * pInput = pLayer->FirstNode("input");
                if (pInput)
                {
                    const XmlNode * pPort = pInput->FirstNode("port");
                    while (pPort)
                    {
                        size_t portId;
                        StringToValue(pPort->FirstAttribute("id")->Value(), portId);
                        bool find = false;
                        for (size_t i = 0; i < edges.size(); ++i)
                        {
                            if (edges[i].toLayer == layerId && edges[i].toPort == portId)
                            {
                                layer.src().push_back(network.layers()[edges[i].fromLayer].name());
                                find = true;
                                break;
                            }
                        }
                        if (!find)
                            return false;
                        pPort = pPort->NextSibling("port");
                    }
                }
                layer.dst().push_back(layer.name());

                String type = pLayer->FirstAttribute("type")->Value();
                if (type == "Activation" && !ConvertActivationLayer(pLayer, layer))
                    return false;
                if (type == "Clamp" && !ConvertClampLayer(pLayer, layer))
                    return false;
                if (type == "Concat" && !ConvertConcatLayer(pLayer, layer))
                    return false;
                if (type == "Convolution" && !ConvertConvolutionLayer(pLayer, bin, layer, weight))
                    return false;
                if (type == "Eltwise" && !ConvertEltwiseLayer(pLayer, layer))
                    return false;
                if (type == "Input" && !ConvertInputLayer(pLayer, layer))
                    return false;
                if (type == "Permute" && !ConvertPermuteLayer(pLayer, layer))
                    return false;
                if (type == "Reshape" && !ConvertReshapeLayer(pLayer, layer))
                    return false;

                if (layer.type() == LayerTypeUnknown)
                    NotImplemented(pLayer, layer);

                network.layers().push_back(layer);
                pLayer = pLayer->NextSibling("layer");
            }

            return true;
        }

        static Shape ConvertShape(const XmlNode * pPort)
        {
            Shape shape;
            const XmlNode * pDim = pPort->FirstNode("dim");
            while (pDim)
            {
                size_t dim;
                StringToValue(pDim->Value(), dim);
                shape.push_back(dim);
                pDim = pDim->NextSibling("dim");
            }
            return shape;
        }

        static Shape ConvertShape(const String & str)
        {
            Strings strs = Separate(str, ",");
            Shape shape(strs.size());
            for (size_t i = 0; i < strs.size(); ++i)
                StringToValue(strs[i], shape[i]);
            return shape;
        }

        bool ConvertActivationLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            String type = pData->FirstAttribute("type")->Value();
            if (type == "sigmoid")
                layer.type() = Synet::LayerTypeSigmoid;
            else
                assert(0);
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

        bool ConvertConcatLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeConcat;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.concat().axis());
            return true;
        }

        bool ConvertConvolutionLayer(const XmlNode * pLayer, const Vector & bin, LayerParam & layer, Tensors & weight)
        {
            layer.type() = Synet::LayerTypeConvolution;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            layer.convolution().kernel().resize(2);
            StringToValue(pData->FirstAttribute("kernel-y")->Value(), layer.convolution().kernel()[0]);
            StringToValue(pData->FirstAttribute("kernel-x")->Value(), layer.convolution().kernel()[1]);
            layer.convolution().stride().resize(2);
            StringToValue(pData->FirstAttribute("stride-y")->Value(), layer.convolution().stride()[0]);
            StringToValue(pData->FirstAttribute("stride-x")->Value(), layer.convolution().stride()[1]);
            layer.convolution().dilation().resize(2);
            StringToValue(pData->FirstAttribute("dilation-y")->Value(), layer.convolution().dilation()[0]);
            StringToValue(pData->FirstAttribute("dilation-x")->Value(), layer.convolution().dilation()[1]);
            StringToValue(pData->FirstAttribute("group")->Value(), layer.convolution().group());
            String padType = pData->FirstAttribute("auto_pad")->Value();
            if (padType == "same_upper" || padType == "same_lower")
            {
                layer.convolution().pad().resize(2);
                layer.convolution().pad()[0] = layer.convolution().kernel()[0] / 2;
                layer.convolution().pad()[1] = layer.convolution().kernel()[1] / 2;
            }
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
                    layer.convolution().outputNum() = output[1];
                }
                else
                    return false;
            }
            else
                return false;
            layer.weight().resize(2);
            layer.weight()[0].dim() = Shape({ (size_t)layer.convolution().outputNum(), inputNum / layer.convolution().group(),  (size_t)layer.convolution().kernel()[0],  (size_t)layer.convolution().kernel()[1] });
            layer.weight()[1].dim() = Shape({ (size_t)layer.convolution().outputNum() });
            const XmlNode * pBlobs = pLayer->FirstNode("blobs");
            if (pBlobs)
            {
                const XmlNode * pWeights = pBlobs->FirstNode("weights");
                if (pWeights)
                {
                    size_t offset, size;
                    StringToValue(pWeights->FirstAttribute("offset")->Value(), offset);
                    StringToValue(pWeights->FirstAttribute("size")->Value(), size);
                    weight.push_back(Tensor());
                    weight.back().Reshape(layer.weight()[0].dim());
                    assert(size == weight.back().Size() * sizeof(float));
                    memcpy(weight.back().CpuData(), bin.data() + offset / sizeof(float), size);
                }
                else
                    return false;
                const XmlNode * pBiases = pBlobs->FirstNode("biases");
                if (pBiases)
                {
                    size_t offset, size;
                    StringToValue(pBiases->FirstAttribute("offset")->Value(), offset);
                    StringToValue(pBiases->FirstAttribute("size")->Value(), size);
                    weight.push_back(Tensor());
                    weight.back().Reshape(layer.weight()[1].dim());
                    assert(size == weight.back().Size() * sizeof(float));
                    memcpy(weight.back().CpuData(), bin.data() + offset / sizeof(float), size);
                }
                else
                    return false;
            }
            else
                return false;

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
                assert(pData->FirstAttribute("coeff") == NULL);
            }
            else
                assert(0);
            return true;
        }

        bool ConvertInputLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeInput;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.input().shape().resize(1);
                    layer.input().shape()[0].dim() = ConvertShape(pPort);
                }
            }
            return true;
        }

        bool ConvertPermuteLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypePermute;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            const XmlAttr * pOrder = pData->FirstAttribute("order");
            if (pOrder == NULL)
                return false;
            layer.permute().order() = ConvertShape(pOrder->Value());
            return true;
        }

        bool ConvertReshapeLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeReshape;
            const XmlNode * pOutput = pLayer->FirstNode("output");
            if (pOutput)
            {
                const XmlNode * pPort = pOutput->FirstNode("port");
                if (pPort)
                {
                    layer.reshape().shape() = ConvertShape(pPort);
                }
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        String NotImplementedMarker()
        {
            return "~~~NOT_IMPLEMENTED~~~";
        }

        void NotImplemented(const XmlNode * src, LayerParam & dst)
        {
            dst.debug().clear();
            dst.debug().push_back(NotImplementedMarker());
            dst.debug().push_back(src->FirstAttribute("type")->Value());
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

    bool ConvertOpencvToSynet(const String & srcData, const String & srcWeights, const String & dstXml, const String & dstBin)
    {
        OpencvToSynet opencvToSynet;
        return opencvToSynet.Convert(srcData, srcWeights, dstXml, dstBin);
    }
}

#endif