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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/Params.h"
#include "Cvt/InferenceEngineV10.h"
#include "Cvt/Optimizer.h"

namespace Synet
{
    class InferenceEngineToSynet
    {
    public:
        bool Convert(String srcModel, String srcWeight, bool trans, const String & dstModel, const String & dstWeight, const OnnxParam &onnxParam, const OptimizerParam & optParam)
        {
            if (!Cpl::FileExists(srcModel))
            {
                String altModel = Cpl::ChangeExtension(srcModel, ".dsc");
                if (altModel != srcModel)
                {
                    if (!Cpl::FileExists(altModel))
                        SYNET_ERROR("Files '" << srcModel << "' and '" << altModel << "' are not exist!");
                    srcModel = altModel;
                }
                else
                    SYNET_ERROR("File '" << srcModel << "' is not exist!");
            }

            if (!Cpl::FileExists(srcWeight))
            {
                String altWeight = Cpl::ChangeExtension(srcWeight, ".dat");
                if (altWeight != srcWeight)
                {
                    if (!Cpl::FileExists(altWeight))
                        SYNET_ERROR("Files '" << srcWeight << "' and '" << altWeight << "' are not exist!");
                    srcWeight = altWeight;
                }
                else
                    SYNET_ERROR("File '" << srcWeight << "' is not exist!");
            }

            XmlDoc srcXml;
            XmlFile file;
            if (!LoadModel(srcModel, file, srcXml))
                SYNET_ERROR("Can't load Inference Engine model '" << srcModel << "' !");

            Bytes srcBin;
            if (!LoadBinaryData(srcWeight, srcBin))
                SYNET_ERROR("Can't load Inference Engine weight '" << srcWeight << "' !");

            int version;
            Synet::NetworkParamHolder dstXml;
            Bytes dstBin = srcBin;
            if (!ConvertNetwork(srcXml, srcBin, trans, onnxParam, dstXml(), dstBin, version))
            {
                String errModel = Cpl::FileNameByPath(dstModel) == dstModel ?
                    "error.xml" : Cpl::MakePath(Cpl::DirectoryByPath(dstModel), "error.xml");
                if (!dstXml.Save(errModel, false))
                    SYNET_ERROR("Can't save Synet model with conversion error '" << errModel << "' !");
                return false;
            }

            if (optParam.saveUnoptimized())
            {
                String uoModel = Cpl::FileNameByPath(dstModel) == dstModel ?
                    "unopt.xml" : Cpl::MakePath(Cpl::DirectoryByPath(dstModel), "unopt.xml");
                if (!dstXml.Save(uoModel, false))
                    SYNET_ERROR("Can't save unoptimized Synet model '" << uoModel << "' !");

                String uoWeight = Cpl::FileNameByPath(dstWeight) == dstWeight ?
                    "unopt.bin" : Cpl::MakePath(Cpl::DirectoryByPath(dstWeight), "unopt.bin");
                if (!SaveBinaryData(dstBin, uoWeight))
                    SYNET_ERROR("Can't save unoptimized Synet weight '" << uoWeight << "' !");
            }

            OptimizerParamHolder param;
            Optimizer optimizer(optParam);
            if (!optimizer.Run(dstXml(), dstBin))
                return false;

            if (version >= 10)
                dstXml().dst().clear();

            if (!dstXml.Save(dstModel, false))
                SYNET_ERROR("Can't save Synet model '" << dstModel << "' !");

            if (!SaveBinaryData(dstBin, dstWeight))
                SYNET_ERROR("Can't save Synet weight '" << dstWeight << "' !");

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
                    xml.Parse<0>(file.Data(), file.Size());
                }
                catch (std::exception e)
                {
                    std::cout << "There is an exception: " << e.what() << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool ConvertNetwork(const XmlDoc & srcXml, const Bytes& srcBin, bool trans, const OnnxParam &onnxParam, Synet::NetworkParam & dstXml, Bytes& dstBin, int & version)
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

            if (version >= 10 && version <= 11)
            {
                InferenceEngineConverterV10 converterV10;
                if (!converterV10.Convert(*pNet, srcBin, trans, onnxParam, dstXml, dstBin))
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

    bool ConvertInferenceEngineToSynet(const String & srcData, const String & srcWeights, bool trans, const String & dstXml, const String & dstBin, const OnnxParam &onnxParam = OnnxParam(), const OptimizerParam& optParam = OptimizerParam())
    {
        InferenceEngineToSynet ieToSynet;
        return ieToSynet.Convert(srcData, srcWeights, trans, dstXml, dstBin, onnxParam, optParam);
    }
}
