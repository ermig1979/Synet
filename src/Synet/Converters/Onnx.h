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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"

#if defined(SYNET_ONNX_ENABLE)

#include <onnx_import/onnx.hpp>

namespace Synet
{
    class OnnxToSynet
    {
    public:
        bool Convert(const String& srcParamPath, const String& srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcParamPath))
            {
                std::cout << "File '" << srcParamPath << "' is not exist!" << std::endl;
                return false;
            }

            if (!Synet::FileExist(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            std::shared_ptr<ngraph::Function> function = ngraph::onnx_import::import_onnx_model(srcGraphPath);
            if (!function)
            {
                std::cout << "Can't read '" << srcGraphPath << "' !" << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Vector weight;
            if (!ConvertNetwork(*function, trans, holder(), weight))
                return false;

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(weight, dstWeightPath))
                return false;

            return false;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<float> Vector;

        bool ConvertNetwork(const ngraph::Function& function, bool trans, Synet::NetworkParam& network, Vector& weight)
        {
            std::vector<std::shared_ptr<ngraph::Node>> nodes = function.get_ordered_ops();
            std::cout << std::endl << "nodes.size(): " << nodes.size() << std::endl;
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                const ngraph::Node& node = *nodes[i];
                const String& type = node.get_type_name();
                LayerParam layer;
                layer.name() = node.get_friendly_name();
                if(type == "Parameter" && !ConvertNodeParameter(node, trans, layer))
                    return ErrorMessage(node);
                network.layers().push_back(layer);
                std::cout << i << ": type = " << node.get_type_name();
                std::cout << " desc = " << node.description();
                std::cout << std::endl;
                //node.write_description(std::cout, 3) << std::endl;
                if (i > 50)
                    break;
            }

            return true;
        }

        bool ConvertNodeParameter(const ngraph::Node& node, bool trans, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeInput;
            if (node.get_output_size() < 1)
                return false;
            layer.input().shape().resize(node.get_output_size());
            for (size_t i = 0; i < node.get_output_size(); ++i)
            {
                Shape shape = node.get_output_shape(i);
                if (trans)
                {
                    if (shape.size() == 4)
                        shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                    layer.input().shape()[i].format() = TensorFormatNhwc;
                }
                layer.input().shape()[i].dim() = shape;
            }
            return true;
        }

        bool SaveWeight(const Vector& bin, const String& path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (!ofs.is_open())
                return false;
            ofs.write((const char*)bin.data(), bin.size() * sizeof(float));
            bool result = (bool)ofs;
            ofs.close();
            return result;
        }

        static String NotImplementedMarker()
        {
            return "~~~NOT_IMPLEMENTED~~~";
        }

        static void NotImplemented(const ngraph::Node & src, LayerParam& dst)
        {
            //dst.type() = LayerTypeStub;
            dst.debug().clear();
            dst.debug().push_back(NotImplementedMarker());
            dst.debug().push_back(src.get_type_name());
        }

        static bool ErrorMessage(const ngraph::Node& node)
        {
            std::cout << "Can't convert layer :";
            //std::cout << " id = " << pLayer->FirstAttribute("id")->Value();
            std::cout << " name = " << node.get_friendly_name();
            std::cout << " , type = " << node.get_type_name();
            //if (pLayer->FirstAttribute("version"))
            //    std::cout << " , version = " << pLayer->FirstAttribute("version")->Value();
            std::cout << " !" << std::endl;
            return false;
        }
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif