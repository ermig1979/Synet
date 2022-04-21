/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
#include "Synet/Utils/FileUtils.h"
#include "Synet/Converters/InferenceEngineV10.h"
#include "Synet/Converters/Optimizer.h"

namespace Synet
{
    class InferenceEngineToSynet
    {
    public:
        bool Convert(const String & srcModel, const String & srcWeight, bool trans, const String & dstModel, const String & dstWeight, const OptimizerParam & optParam)
        {
            if (!Cpl::FileExists(srcModel))
            {
                std::cout << "File '" << srcModel << "' is not exist!" << std::endl;
                return false;
            }

            if (!Cpl::FileExists(srcWeight))
            {
                std::cout << "File '" << srcWeight << "' is not exist!" << std::endl;
                return false;
            }

            XmlDoc srcXml;
            XmlFile file;
            if (!LoadModel(srcModel, file, srcXml))
            {
                std::cout << "Can't load Inference Engine model '" << srcModel << "' !" << std::endl;
                return false;
            }

            Vector srcBin;
            if (!LoadBinaryData(srcWeight, srcBin))
            {
                std::cout << "Can't load Inference Engine weight '" << srcWeight << "' !" << std::endl;
                return false;
            }

            int version;
            Synet::NetworkParamHolder dstXml;
            Vector dstBin = srcBin;
            if (!ConvertNetwork(srcXml, srcBin, trans, dstXml(), dstBin, version))
                return false;

            OptimizerParamHolder param;
            Optimizer optimizer(optParam);
            if (!optimizer.Run(dstXml(), dstBin))
                return false;

            if (version >= 10)
                dstXml().dst().clear();

            if (!dstXml.Save(dstModel, false))
            {
                std::cout << "Can't save Synet model '" << dstModel << "' !" << std::endl;
                return false;
            }

            if (!SaveBinaryData(dstBin, dstWeight))
            {
                std::cout << "Can't save Synet weight '" << dstWeight << "' !" << std::endl;
                return false;
            }

            return true;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;

        typedef std::vector<float> Vector;
        typedef Cpl::Xml::File<char> XmlFile;
        typedef Cpl::Xml::XmlBase<char> XmlBase;
        typedef Cpl::Xml::XmlDocument<char> XmlDoc;
        typedef Cpl::Xml::XmlNode<char> XmlNode;
        typedef Cpl::Xml::XmlAttribute<char> XmlAttr;

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
                    std::cout << "There is an exception: " << e.what() << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool ConvertNetwork(const XmlDoc & srcXml, const Vector & srcBin, bool trans, Synet::NetworkParam & dstXml, Vector & dstBin, int & version)
        {
            const XmlNode * pNet = srcXml.FirstNode("net");
            if (pNet == NULL)
            {
                std::cout << "Can't find 'net' node!" << std::endl;
                return false;
            }

            dstXml.info().version() = 1;
            const XmlAttr* pName = pNet->FirstAttribute("name");
            if (pName)
                dstXml.info().name() = pName->Value();

            const XmlAttr * pVersion = pNet->FirstAttribute("version");
            if (pVersion == NULL)
            {
                std::cout << "Can't find 'version' attribute!" << std::endl;
                return false;
            }
            dstXml.info().from() = String("InferenceEngine-v") + pVersion->Value();
            dstXml.info().when() = Cpl::CurrentDateTimeString();
            dstXml.info().synet() = Synet::Version();

            Cpl::ToVal(pVersion->Value(), version);

            if (version >= 10 && version <= 10)
            {
                InferenceEngineConverterV10 converterV10;
                if (!converterV10.Convert(*pNet, srcBin, trans, dstXml, dstBin))
                {
                    std::cout << "Can't convert IE model v" << version << "!" << std::endl;
                    return false;
                }
            }            
            else
            {
                std::cout << "Unsupported version " << version << " of IE model!" << std::endl;
                return false;
            }

            return true;
        }
    };

    bool ConvertInferenceEngineToSynet(const String & srcData, const String & srcWeights, bool trans, const String & dstXml, const String & dstBin, const OptimizerParam& optParam = OptimizerParam())
    {
        InferenceEngineToSynet ieToSynet;
        return ieToSynet.Convert(srcData, srcWeights, trans, dstXml, dstBin, optParam);
    }
}
