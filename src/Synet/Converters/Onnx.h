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
#include "Synet/Converters/SynetUtils.h"
#include "Synet/Utils/FileUtils.h"

#if defined(SYNET_ONNX_ENABLE)

#include <onnx_import/onnx.hpp>
#include <ngraph/ops.hpp>

namespace Synet
{
    class OnnxToSynet : public SynetUtils
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
            {
                std::cout << "Can't save Synet model '" << dstModelPath << "' !" << std::endl;
                return false;
            }

            if (!SaveBinaryData(weight, dstWeightPath))
            {
                std::cout << "Can't save Synet weight '" << dstWeightPath << "' !" << std::endl;
                return false;
            }

            return false;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<float> Vector;

        bool ConvertNetwork(const ngraph::Function& function, bool trans, Synet::NetworkParam& network, Vector& reordered)
        {
            network.name() = function.get_friendly_name();
            std::vector<std::shared_ptr<ngraph::Node>> nodes = function.get_ordered_ops();
            Vector original;
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                const ngraph::Node& node = *nodes[i];
                LayerParam layer;
                if(!ConvertNodeAny(node, layer))
                    return ErrorMessage(node);

                const String& type = node.get_type_name();
                if (type == "Add" && !ConvertNodeAdd(node, network.layers(), original, layer))
                    return ErrorMessage(node);
                if (type == "Constant" && !ConvertNodeConstant(node, trans, layer, original, reordered))
                    return ErrorMessage(node);
                if (type == "Convolution" && !ConvertNodeConvolution(node, trans, network.layers(), layer, original, reordered))
                    return ErrorMessage(node);
                if (type == "Multiply" && !ConvertNodeMultiply(node, network.layers(), original, layer))
                    return ErrorMessage(node);
                if (type == "Parameter" && !ConvertNodeParameter(node, trans, layer))
                    return ErrorMessage(node);
                if (type == "PRelu" && !ConvertNodePrelu(node, network.layers(), layer))
                    return ErrorMessage(node);
                if (type == "Relu" && !ConvertNodeRelu(node, layer))
                    return ErrorMessage(node);
                if (type == "Sigmoid" && !ConvertNodeSigmoid(node, layer))
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

            if (!RemoveUnusedConst(network.layers()))
                return false;

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

        bool ConvertNodeAdd(const ngraph::Node& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            Shape src0 = node.get_input_shape(0);
            Shape src1 = node.get_input_shape(1);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeConst && TensorSize(src0) >= TensorSize(src1))
            {
                if (TensorSize(src1) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    const float* pShift = original.data() + second->weight()[0].offset() / sizeof(float);
                    layer.power().shift() = pShift[0];
                }
                else
                {
                    layer.type() = Synet::LayerTypeBias;
                    layer.weight() = second->weight();
                    if (!CompactShape(layer.weight()[0].dim()))
                        return false;
                }
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeSum;
                if (TensorSize(src0) < TensorSize(src1))
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertNodeConstant(const ngraph::Node& node, bool trans, LayerParam& layer, Vector& original, Vector& reordered)
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
                size_t offset = original.size();
                layer.weight().resize(1);
                layer.weight()[0].dim() = tensor.get_shape();
                layer.weight()[0].offset() = offset * sizeof(float);
                layer.weight()[0].type() = TensorType32f;
                layer.weight()[0].size() = tensor.size();
                original.resize(offset + DivHi(tensor.size(), sizeof(float)));
                memcpy(original.data() + offset, ((ngraph::op::v0::Constant*)&node)->get_data_ptr(), tensor.size());
                reordered.resize(offset + DivHi(tensor.size(), sizeof(float)));
                memcpy(reordered.data() + offset, ((ngraph::op::v0::Constant*)&node)->get_data_ptr(), tensor.size());
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

        bool ConvertNodeConvolution(const ngraph::Node& node, bool trans, const LayerParams & layers, LayerParam& layer, Vector& original, Vector& reordered)
        {
            layer.type() = Synet::LayerTypeConvolution;
            layer.convolution().biasTerm() = false;
            const ngraph::op::v1::Convolution* conv = (ngraph::op::v1::Convolution*)&node;
            layer.convolution().stride() = conv->get_strides();
            layer.convolution().dilation() = conv->get_dilations();
            if (conv->get_auto_pad() == ngraph::op::PadType::SAME_UPPER)
                layer.convolution().autoPad() = true;
            layer.convolution().pad() = Shp(
                conv->get_pads_begin()[0], conv->get_pads_begin()[1], 
                conv->get_pads_end()[0], conv->get_pads_end()[1]);
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            const Shape& shape = second->weight()[0].dim();
            layer.weight() = second->weight();
            if (String(node.get_type_name()) == "Convolution")
            {
                if (!CheckDims(shape, 4, "convolution weight"))
                    return false;
                layer.convolution().kernel() = Shape({ shape[2], shape[3] });
                layer.convolution().outputNum() = (uint32_t)shape[0];
            }
            else
            {
                if (!CheckDims(shape, 5, "convolution weight"))
                    return false;
                layer.convolution().kernel() = Shape({ shape[3], shape[4] });
                layer.convolution().group() = (uint32_t)shape[0];
                layer.convolution().outputNum() = (uint32_t)(shape[0] * shape[1]);
                layer.weight()[0].dim() = Shape({ shape[0] * shape[1] , shape[2], shape[3], shape[4] });
            }
            layer.src().resize(1);
            if (trans)
                return ReorderWeight(original, Shape(), layer, reordered);
            return true;
        }

        bool ConvertNodeMultiply(const ngraph::Node& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (first == NULL || second == NULL)
                return false;
            if (first->type() == LayerTypeConst || second->type() == LayerTypeConst)
            {
                if (first->type() == LayerTypeConst)
                {
                    std::swap(layer.src()[0], layer.src()[1]);
                    std::swap(first, second);
                }
                if (TensorSize(second->weight()[0].dim()) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().power() = 1.0f;
                    layer.power().scale() = GetWeight<float>(original, second->weight()[0])[0];
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
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
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

        bool ConvertNodePrelu(const ngraph::Node& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            layer.type() = Synet::LayerTypePrelu;
            layer.weight() = second->weight();
            layer.src().resize(1);
            if (!CompactShape(layer.weight()[0].dim()))
                return false;
            return true;
        }

        bool ConvertNodeRelu(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertNodeSigmoid(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        //---------------------------------------------------------------------

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
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif