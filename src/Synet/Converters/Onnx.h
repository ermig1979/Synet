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
#include <ngraph/op/constant.hpp>

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
            network.name() = function.get_friendly_name();
            std::vector<std::shared_ptr<ngraph::Node>> nodes = function.get_ordered_ops();
            std::cout << std::endl << "nodes.size(): " << nodes.size() << std::endl;
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                const ngraph::Node& node = *nodes[i];
                LayerParam layer;
                if(!ConvertNodeAny(node, layer))
                    return ErrorMessage(node);

                const String& type = node.get_type_name();
                if (type == "Constant" && !ConvertNodeConstant(node, trans, layer, weight))
                    return ErrorMessage(node);
                if(type == "Parameter" && !ConvertNodeParameter(node, trans, layer))
                    return ErrorMessage(node);

#if 0
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(node);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(node, layer);
                    std::cout << "Not implemented layer : name = " << layer.name() << " ; type = " << type << std::endl;
                }
#endif
                network.layers().push_back(layer);
                //node.write_description(std::cout, 1) << std::endl;
            }

            //if (!RemoveUnusedConst(network.layers()))
            //    return false;

            return true;
        }

        bool ConvertNodeAny(const ngraph::Node& node, LayerParam& layer)
        {
            layer.name() = node.get_friendly_name();
            for (size_t i = 0; i < node.get_input_size(); ++i)
                layer.src().push_back(node.get_input_node_ptr(i)->get_friendly_name());
            if(node.get_output_size() == 1)
                layer.dst().push_back(layer.name());
            else
            {
                for (size_t i = 0; i < node.get_output_size(); ++i)
                    layer.dst().push_back(layer.name() + ":" + ValueToString(i));
            }
            return true;
        }

        bool ConvertNodeConstant(const ngraph::Node& node, bool trans, LayerParam& layer, Vector& weight)
        {
            if (node.get_output_size() != 1)
                return false;
            const ngraph::descriptor::Tensor& tensor = node.get_output_tensor(0);
            ngraph::element::Type_t type = tensor.get_element_type();
            switch (type)
            {
            case ngraph::element::Type_t::f32:
            {
                layer.type() = Synet::LayerTypeConst;
                size_t offset = weight.size();
                layer.weight().resize(1);
                layer.weight()[0].dim() = tensor.get_shape();
                layer.weight()[0].offset() = offset * sizeof(float);
                layer.weight()[0].type() = TensorType32f;
                layer.weight()[0].size() = tensor.size();
                weight.resize(offset + DivHi(tensor.size(), sizeof(float)));
                memcpy(weight.data() + offset, ((ngraph::op::v0::Constant*)&node)->get_data_ptr(), tensor.size());
                break;
            }
            case ngraph::element::Type_t::i64:
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeConst;
                layer.meta().alpha().type() = TensorType64i;
                layer.meta().alpha().shape() = tensor.get_shape();
                layer.meta().alpha().i64().resize(tensor.size() / sizeof(int64_t));
                const int64_t* src = ((ngraph::op::v0::Constant*)&node)->get_data_ptr<int64_t>();
                for (size_t i = 0; i < layer.meta().alpha().i64().size(); ++i)
                    layer.meta().alpha().i64()[i] = src[i];
                break;
            }
            default:
                std::cout << "Unsupported ConstLayer type: " << tensor.get_element_type().get_type_name() << " !" << std::endl;
                return false;
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

        static void NotImplemented(const ngraph::Node & node, LayerParam& layer)
        {
            layer.debug().clear();
            layer.debug().push_back(NotImplementedMarker());
            layer.debug().push_back(node.get_type_name());
        }

        static bool ErrorMessage(const ngraph::Node& node)
        {
            std::cout << "Can't convert layer :";
            std::cout << " name = " << node.get_friendly_name();
            std::cout << " , type = " << node.get_type_name();
            std::cout << " !" << std::endl;
            return false;
        }

        static bool RemoveUnusedConst(LayerParams& layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const LayerParam& layer = layers[i];
                if (layer.type() == LayerTypeConst || (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst))
                {
                    const String& name = layer.name();
                    bool unused = true;
                    for (size_t j = i + 1; j < layers.size() && unused; ++j)
                        for (size_t k = 0; k < layers[j].src().size() && unused; ++k)
                            if (layers[j].src()[k] == name)
                                unused = false;
                    if (unused)
                        layers.erase(layers.begin() + i), i--;
                }
            }
            return true;
        }
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif