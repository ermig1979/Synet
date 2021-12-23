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
#include "Synet/Tensor.h"
#include "Synet/Converters/SynetUtils.h"

//#define SYNET_IE_PARSE_STOP_ON_ERROR

namespace Synet
{
    class InferenceEngineConverter : public SynetUtils
    {
    protected:
        typedef Cpl::Xml::File<char> XmlFile;
        typedef Cpl::Xml::XmlBase<char> XmlBase;
        typedef Cpl::Xml::XmlDocument<char> XmlDoc;
        typedef Cpl::Xml::XmlNode<char> XmlNode;
        typedef Cpl::Xml::XmlAttribute<char> XmlAttr;

        struct Edge
        {
            size_t fromLayer, fromPort, toLayer, toPort;
        };
        typedef std::vector<Edge> Edges;

        typedef std::map<size_t, size_t> IndexMap;

        struct TensorInfo
        {
            String name, desc;
            Shape shape;
            size_t id;// , batch, channels, spatial, unknown;
        };
        typedef std::vector<TensorInfo> TensorInfos;
        typedef std::map<String, TensorInfo> TensorInfoMap;

        typedef std::map<String, LayerParam> LayerParamMap;

        static bool ParseEdges(const XmlNode& src, Edges& edges)
        {
            edges.clear();
            const XmlNode* pEdges = src.FirstNode("edges");
            if (pEdges == NULL)
            {
                std::cout << "Can't find 'edges' node!" << std::endl;
                return false;
            }
            const XmlNode* pEdge = pEdges->FirstNode("edge");
            while (pEdge)
            {
                Edge edge;
                Cpl::ToVal(pEdge->FirstAttribute("from-layer")->Value(), edge.fromLayer);
                Cpl::ToVal(pEdge->FirstAttribute("from-port")->Value(), edge.fromPort);
                Cpl::ToVal(pEdge->FirstAttribute("to-layer")->Value(), edge.toLayer);
                Cpl::ToVal(pEdge->FirstAttribute("to-port")->Value(), edge.toPort);
                edges.push_back(edge);
                pEdge = pEdge->NextSibling("edge");
            }
            return true;
        }

        static bool ParseInputOutput(const XmlNode &src, const Edges & edges, const LayerParams & layers, LayerParam & dst, IndexMap & index, TensorInfoMap & info)
        {
            size_t layerId;
            Cpl::ToVal(src.FirstAttribute("id")->Value(), layerId);
            index[layerId] = layers.size();
            //assert(layerId == layers.size());

            dst.name() = src.FirstAttribute("name")->Value();

            const XmlNode* pInput = src.FirstNode("input");
            if (pInput)
            {
                const XmlNode* pPort = pInput->FirstNode("port");
                while (pPort)
                {
                    size_t portId;
                    Cpl::ToVal(pPort->FirstAttribute("id")->Value(), portId);
                    bool find = false;
                    for (size_t i = 0; i < edges.size(); ++i)
                    {
                        if (edges[i].toLayer == layerId && edges[i].toPort == portId)
                        {
                            const LayerParam& fromLayer = layers[index[edges[i].fromLayer]];
                            if (fromLayer.dst().size() == 1)
                                dst.src().push_back(fromLayer.name());
                            else
                            {
                                size_t dstPortMin = INT_MAX;
                                for (size_t j = 0; j < edges.size(); ++j)
                                    if (edges[j].fromLayer == edges[i].fromLayer)
                                        dstPortMin = Synet::Min(dstPortMin, edges[j].fromPort);
                                dst.src().push_back(fromLayer.name() + ":" + Cpl::ToStr(edges[i].fromPort - dstPortMin));
                            }
                            find = true;
                            break;
                        }
                    }
                    if (!find)
                        return ErrorMessage(&src);
                    pPort = pPort->NextSibling("port");
                }
            }
            const XmlNode* pOutput = src.FirstNode("output");
            if (pOutput)
            {
                TensorInfos tensorInfos;
                const XmlNode* pPort = pOutput->FirstNode("port");
                while (pPort)
                {
                    tensorInfos.push_back(TensorInfo());
                    Cpl::ToVal(pPort->FirstAttribute("id")->Value(), tensorInfos.back().id);
                    tensorInfos.back().shape = ConvertShape(pPort);
                    pPort = pPort->NextSibling("port");
                }
                if (tensorInfos.empty())
                    return ErrorMessage(&src);
                if (tensorInfos.size() == 1)
                    dst.dst().push_back(dst.name());
                else
                {
                    for (size_t i = 0; i < tensorInfos.size(); ++i)
                        dst.dst().push_back(dst.name() + ":" + Cpl::ToStr(i));
                }
                for (size_t i = 0; i < dst.dst().size(); ++i)
                {
                    tensorInfos[i].name = dst.dst()[i];
                    info[tensorInfos[i].name] = tensorInfos[i];
                }
            }
            return true;
        }

        template<class T> static bool ConvertValue(const XmlBase * pSrc, T & dst)
        {
            if (pSrc == NULL)
                return false;
            Cpl::ToVal(pSrc->Value(), dst);
            return true;
        }

        template<class T> static bool ConvertVector(const XmlBase * pSrc, std::vector<T> & dst, const String & delimeter = ",")
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
                Cpl::ToVal(subs[i], dst[i]);
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
                Cpl::ToVal(pDim->Value(), dim);
                shape.push_back(dim);
                pDim = pDim->NextSibling("dim");
            }
            return shape;
        }

        static Shape ConvertShape(const String & str)
        {
            Strings strs = Cpl::Separate(str, ",");
            Shape shape(strs.size());
            for (size_t i = 0; i < strs.size(); ++i)
                Cpl::ToVal(strs[i], shape[i]);
            return shape;
        }

        static Shape ConvertInputShape(const XmlNode * pLayer, String port = String())
        {
            const XmlNode * pInput = pLayer->FirstNode("input");
            assert(pInput);
            const XmlNode * pPort = pInput->FirstNode("port");
            if (!port.empty())
            {
                while (pPort && port != pPort->FirstAttribute("id")->Value())
                    pPort = pPort->NextSibling("port");
            }
            assert(pPort);
            return ConvertShape(pPort);
        }

        static Shape ConvertOutputShape(const XmlNode * pLayer, String port = String())
        {
            const XmlNode * pOutput = pLayer->FirstNode("output");
            assert(pOutput);
            const XmlNode * pPort = pOutput->FirstNode("port");
            if (!port.empty())
            {
                while (pPort && port != pPort->FirstAttribute("id")->Value())
                    pPort = pPort->NextSibling("port");
            }
            assert(pPort);
            return ConvertShape(pPort);
        }

        static void NotImplemented(const XmlNode * src, LayerParam & dst)
        {
            //dst.type() = LayerTypeStub;
            dst.debug().clear();
            dst.debug().push_back(NotImplementedMarker());
            dst.debug().push_back(src->FirstAttribute("type")->Value());
        }

        static bool ErrorMessage(const XmlNode * pLayer)
        {
            std::cout << "Can't convert layer :";
            std::cout << " id = " << pLayer->FirstAttribute("id")->Value();
            std::cout << " , name = " << pLayer->FirstAttribute("name")->Value();
            std::cout << " , type = " << pLayer->FirstAttribute("type")->Value();
            if(pLayer->FirstAttribute("version"))
                std::cout << " , version = " << pLayer->FirstAttribute("version")->Value();
            std::cout << " !" << std::endl;
            return false;
        }
    };
}
