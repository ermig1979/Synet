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
#include "Synet/Converters/Optimizer.h"

#if defined(SYNET_OPENCV_ENABLE)

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace Synet
{
    class OpencvToSynet
    {
    public:
        bool Convert(const String & srcModelPath, const String & srcWeightPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
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

            Vector srcBin;
            if (!LoadWeight(srcWeightPath, srcBin))
            {
                std::cout << "Can't load opencv::dnn weight '" << srcWeightPath << "' !" << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Vector dstBin = srcBin;
            if (!ConvertNetwork(xml, srcBin, trans, holder(), dstBin))
                return false;

            Optimizer optimizer;
            if (!optimizer.Run(holder()))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(dstBin, dstWeightPath))
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

        typedef std::pair<ptrdiff_t, ptrdiff_t> Ids;
        typedef std::vector<Ids> Index;
        typedef std::map<String, Index> IndexMap;

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

        bool ConvertNetwork(const XmlDoc & xml, const Vector & srcBin, bool trans, Synet::NetworkParam & network, Vector & dstBin)
        {
            network.version() = 1;

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
            const XmlNode * pLayer = pLayers->FirstNode("layer"), * pPrevLayer = NULL;
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
                                const LayerParam & fromLayer = network.layers()[edges[i].fromLayer];
                                if(fromLayer.dst().size() == 1)
                                    layer.src().push_back(fromLayer.name());
                                else
                                    layer.src().push_back(fromLayer.name() + ":" + ValueToString(edges[i].fromPort - 1));
                                find = true;
                                break;
                            }
                        }
                        if (!find)
                            return false;
                        pPort = pPort->NextSibling("port");
                    }
                }
                const XmlNode * pOutput = pLayer->FirstNode("output");
                if (pOutput)
                {
                    Shape portIds;
                    const XmlNode * pPort = pOutput->FirstNode("port");
                    while (pPort)
                    {
                        portIds.push_back(-1);
                        StringToValue(pPort->FirstAttribute("id")->Value(), portIds.back());
                        pPort = pPort->NextSibling("port");
                    }
                    if (portIds.empty())
                        return false;
                    if (portIds.size() == 1)
                        layer.dst().push_back(layer.name());
                    else
                    {
                        for(size_t i = 0; i < portIds.size(); ++i)
                            layer.dst().push_back(layer.name() + ":" + ValueToString(i));
                    }
                }

                String type = pLayer->FirstAttribute("type")->Value();
                if (type == "Activation" && !ConvertActivationLayer(pLayer, layer))
                    return false;
                if (type == "Clamp" && !ConvertClampLayer(pLayer, layer))
                    return false;
                if (type == "Const" && !ConvertConstLayer(pLayer, srcBin, trans, layer, dstBin))
                    return false;
                if (type == "Concat" && !ConvertConcatLayer(pLayer, trans, layer))
                    return false;
                if (type == "Convolution" && !ConvertConvolutionLayer(pLayer, srcBin, trans, layer, dstBin))
                    return false;
                if (type == "DetectionOutput" && !ConvertDetectionOutputLayer(pLayer, layer))
                    return false;
                if (type == "Eltwise" && !ConvertEltwiseLayer(pLayer, layer))
                    return false;
                if (type == "FullyConnected" && !ConvertFullyConnectedLayer(pLayer, pPrevLayer, srcBin, trans, layer, dstBin))
                    return false;
                if (type == "Input" && !ConvertInputLayer(pLayer, trans, layer))
                    return false;
                if (type == "Permute" && !ConvertPermuteLayer(pLayer, pPrevLayer, trans, layer))
                    return false;
                if (type == "Pooling" && !ConvertPoolingLayer(pLayer, layer))
                    return false;
                if (type == "Power" && !ConvertPowerLayer(pLayer, layer))
                    return false;
                if (type == "PReLU" && !ConvertPreluLayer(pLayer, srcBin, trans, layer, dstBin))
                    return false;
                if (type == "ReLU" && !ConvertReluLayer(pLayer, layer))
                    return false;
                if (type == "Resample" && !ConvertResampleLayer(pLayer, layer))
                    return false;
                if (type == "Reshape" && !ConvertReshapeLayer(pLayer, trans, layer))
                    return false;
                if (type == "ScaleShift" && !ConvertScaleShiftLayer(pLayer, srcBin, trans, layer, dstBin))
                    return false;
                if (type == "SoftMax" && !ConvertSoftmaxLayer(pLayer, trans, layer))
                    return false;
                if (type == "Split" && !ConvertSplitLayer(pLayer, trans, layer))
                    return false;
                if (type == "Tile" && !ConvertTileLayer(pLayer, trans, layer))
                    return false;

                if (layer.type() == LayerTypeUnknown)
                    NotImplemented(pLayer, layer);

                network.layers().push_back(layer);
                pPrevLayer = pLayer;
                pLayer = pLayer->NextSibling("layer");
            }

            return true;
        }

        template<class T> static bool ConvertVector(const XmlAttr * pSrc, std::vector<T> & dst, const String & delimeter = ",")
        {
            if (pSrc == NULL)
                return false;
            const String & str = pSrc->Value();
            size_t current = 0;
            Strings subs;
            while (current != std::string::npos)
            {
                size_t next = str.find(delimeter, current);
                subs.push_back(str.substr(current, next - current));
                current = next;
                if (current != std::string::npos)
                    current += delimeter.size();
            }
            dst.resize(subs.size());
            for (size_t i = 0; i < subs.size(); ++i)
                StringToValue(subs[i], dst[i]);
            return true;
        }

        template<class T> static bool ConvertVectors(const XmlAttr * pBeg, const XmlAttr * pEnd, std::vector<T> & dst, const String & delimeter = ",")
        {
            std::vector<T> beg, end;
            if (!(ConvertVector(pBeg, beg, delimeter) && ConvertVector(pEnd, end, delimeter)))
                return false;
            dst.resize(beg.size() + end.size());
            for (size_t i = 0; i < beg.size(); ++i)
                dst[i] = beg[i];
            for (size_t i = 0; i < end.size(); ++i)
                dst[beg.size() + i] = end[i];
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

        static Shape ConvertInputShape(const XmlNode * pLayer)
        {
            const XmlNode * pInput = pLayer->FirstNode("input");
            assert(pInput);
            const XmlNode * pPort = pInput->FirstNode("port");
            assert(pPort);
            return ConvertShape(pPort);
        }

        static Shape ConvertOutputShape(const XmlNode * pLayer)
        {
            const XmlNode * pOutput = pLayer->FirstNode("output");
            assert(pOutput);
            const XmlNode * pPort = pOutput->FirstNode("port");
            assert(pPort);
            return ConvertShape(pPort);
        }

        static void ConvertWeight(const XmlNode * pNode, const Vector & srcBin, int mode, const Shape & input, WeightParam & param, Vector & dstBin)
        {
            const Shape & shape = param.dim();
            StringToValue(pNode->FirstAttribute("offset")->Value(), param.offset());
            StringToValue(pNode->FirstAttribute("size")->Value(), param.size());
            const float * pSrc = srcBin.data() + param.offset() / sizeof(float);
            float * pDst = dstBin.data() + param.offset() / sizeof(float);
            Tensor dst(pDst, param.size(), shape, param.format());
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

        bool ConvertConvolutionLayer(const XmlNode * pLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypeConvolution;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
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
                    Shape shape = Shape({ (size_t)layer.convolution().outputNum(), inputNum / layer.convolution().group(),  
                        (size_t)layer.convolution().kernel()[0],  (size_t)layer.convolution().kernel()[1] });
                    layer.weight().resize(1);
                    if (trans)
                    {
                        shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
                        layer.weight()[0].format() = TensorFormatNhwc;
                    }
                    layer.weight()[0].dim() = shape;
                    ConvertWeight(pWeights, srcBin, trans ? 2 : 0, Shape(), layer.weight()[0], dstBin);
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
                assert(pData->FirstAttribute("coeff") == NULL);
            }
            else if (operation == "max")
                layer.eltwise().operation() = EltwiseOperationTypeMax;
            else if (operation == "min")
                layer.eltwise().operation() = EltwiseOperationTypeMin;
            else
                assert(0);
            return true;
        }

        bool ConvertFullyConnectedLayer(const XmlNode * pLayer, const XmlNode * pPrevLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
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
            if (trans && order.size() == 4)
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
                else
                {
                    Shape nhwc = Shape({ 0, 2, 3, 1 });
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    order = Shape({ nchw[order[nhwc[0]]], nchw[order[nhwc[1]]], nchw[order[nhwc[2]]], nchw[order[nhwc[3]]] });
                }
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
                    layer.weight()[0].dim() = Shape({ ConvertInputShape(pLayer)[1] });
                    ConvertWeight(pWeights, srcBin, 0, Shape(), layer.weight()[0], dstBin);
                }
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        bool ConvertReluLayer(const XmlNode * pLayer, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeRelu;
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

        bool ConvertReshapeLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
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
                    if (trans && output.size() == 4)
                    {
                        if (input.size() > 3 && output[1]*output[2] == input[1] && output[3] == input[2]*input[3])
                            output = Shape({ output[0], output[3] , output[1] , output[2] });
                        else
                            output = Shape({ output[0], output[2] , output[3] , output[1] });
                    }
                    if (input.size() == 4)
                        output[0] = -1;
                    //if (trans && output.size() == 3)
                    //{
                    //    output = Shape({ output[0], output[2] , output[1] });
                    //}
                    layer.reshape().shape() = output;
                }
                else
                    return false;
            }
            else
                return false;
            return true;
        }

        bool ConvertScaleShiftLayer(const XmlNode * pLayer, const Vector & srcBin, bool trans, LayerParam & layer, Vector & dstBin)
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

        bool ConvertSplitLayer(const XmlNode * pLayer, bool trans, LayerParam & layer)
        {
            layer.type() = Synet::LayerTypeUnpack;
            const XmlNode * pData = pLayer->FirstNode("data");
            if (pData == NULL)
                return false;
            StringToValue(pData->FirstAttribute("axis")->Value(), layer.unpack().axis());
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
                    layer.unpack().axis() = (int32_t)nchw[layer.unpack().axis()];
                }
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

        bool SaveWeight(const Vector & bin, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (!ofs.is_open())
                return false;
            ofs.write((const char*)bin.data(), bin.size() * sizeof(float));
            bool result = (bool)ofs;
            ofs.close();
            return result;
        }
    };

    bool ConvertOpencvToSynet(const String & srcData, const String & srcWeights, bool trans, const String & dstXml, const String & dstBin)
    {
        OpencvToSynet opencvToSynet;
        return opencvToSynet.Convert(srcData, srcWeights, trans, dstXml, dstBin);
    }
}

#endif