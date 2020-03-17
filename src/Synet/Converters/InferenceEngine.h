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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Converters/InferenceEngineV7.h"
#include "Synet/Converters/Optimizer.h"

namespace Synet
{
    class InferenceEngineToSynet
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

            XmlDoc srcXml;
            XmlFile file;
            if (!LoadModel(srcModelPath, file, srcXml))
            {
                std::cout << "Can't load Inference Engine model '" << srcModelPath << "' !" << std::endl;
                return false;
            }

            Vector srcBin;
            if (!LoadWeight(srcWeightPath, srcBin))
            {
                std::cout << "Can't load Inference Engine weight '" << srcWeightPath << "' !" << std::endl;
                return false;
            }

            Synet::NetworkParamHolder dstXml;
            Vector dstBin = srcBin;
            if (!ConvertNetwork(srcXml, srcBin, trans, dstXml(), dstBin))
                return false;

            Optimizer optimizer;
            if (!optimizer.Run(dstXml(), dstBin))
                return false;

            if (!dstXml.Save(dstModelPath, false))
            {
                std::cout << "Can't save Synet model '" << dstModelPath << "' !" << std::endl;
                return false;
            }

            if (!SaveWeight(dstBin, dstWeightPath))
            {
                std::cout << "Can't save Synet weight '" << dstWeightPath << "' !" << std::endl;
                return false;
            }

            return true;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;

        typedef std::vector<float> Vector;
        typedef Xml::File<char> XmlFile;
        typedef Xml::XmlBase<char> XmlBase;
        typedef Xml::XmlDocument<char> XmlDoc;
        typedef Xml::XmlNode<char> XmlNode;
        typedef Xml::XmlAttribute<char> XmlAttr;

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

        bool ConvertNetwork(const XmlDoc & srcXml, const Vector & srcBin, bool trans, Synet::NetworkParam & dstXml, Vector & dstBin)
        {
            const XmlNode * pNet = srcXml.FirstNode("net");
            if (pNet == NULL)
            {
                std::cout << "Can't find 'net' node!" << std::endl;
                return false;
            }

            const XmlAttr* pName = srcXml.FirstAttribute("name");
            if (pName)
                dstXml.name() = pName->Value();

            const XmlAttr * pVersion = pNet->FirstAttribute("version");
            if (pVersion == NULL)
            {
                std::cout << "Can't find 'version' attribute!" << std::endl;
                return false;
            }

            int version;
            StringToValue(pVersion->Value(), version);

            if (version <= 7)
            {
                InferenceEngineConverterV7 converterV7;
                if (!converterV7.Convert(*pNet, srcBin, trans, dstXml, dstBin))
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

    bool ConvertInferenceEngineToSynet(const String & srcData, const String & srcWeights, bool trans, const String & dstXml, const String & dstBin)
    {
        InferenceEngineToSynet ieToSynet;
        return ieToSynet.Convert(srcData, srcWeights, trans, dstXml, dstBin);
    }
}
