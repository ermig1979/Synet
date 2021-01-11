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
            //if (!Synet::FileExist(srcParamPath))
            //{
            //    std::cout << "File '" << srcParamPath << "' is not exist!" << std::endl;
            //    return false;
            //}

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

            return true;
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
                //node.write_description(std::cout, 1) << std::endl;

                LayerParam layer;
                if(!ConvertNodeAny(node, layer))
                    return ErrorMessage(node);

                const String& type = node.get_type_name();
                if (type == "Add" && !ConvertNodeAdd(node, network.layers(), original, layer))
                    return ErrorMessage(node);
                if (type == "AvgPool" && !ConvertNodeAvgPool(node, layer))
                    return ErrorMessage(node);
                if (type == "BatchNormInference" && !ConvertNodeBatchNormInference(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(node);
                if (type == "Clamp" && !ConvertNodeClamp(node, layer))
                    return ErrorMessage(node);
                if (type == "Concat" && !ConvertNodeConcat(node, trans, network.layers(), layer))
                    return ErrorMessage(node);
                if (type == "Constant" && !ConvertNodeConstant(node, trans, layer, original, reordered))
                    return ErrorMessage(node);
                if (type == "Convolution" && !ConvertNodeConvolution(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(node);
                if (type == "Gather" && !ConvertNodeGather(node, layer))
                    return ErrorMessage(node);
                if (type == "GroupConvolution" && !ConvertNodeGroupConvolution(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(node);
                if (type == "MatMul" && !ConvertNodeMatMul(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(node);
                if (type == "MaxPool" && !ConvertNodeMaxPool(node, layer))
                    return ErrorMessage(node);
                if (type == "Multiply" && !ConvertNodeMultiply(node, network.layers(), original, layer))
                    return ErrorMessage(node);
                if (type == "Parameter" && !ConvertNodeParameter(node, trans, layer))
                    return ErrorMessage(node);
                if (type == "PRelu" && !ConvertNodePrelu(node, network.layers(), layer))
                    return ErrorMessage(node);
                if (type == "Relu" && !ConvertNodeRelu(node, layer))
                    return ErrorMessage(node);
                if (type == "Reshape" && !ConvertNodeReshape(node, trans, network.layers(), original, layer))
                    return ErrorMessage(node);
                if (type == "Result" && !ConvertNodeResult(node, layer))
                    return ErrorMessage(node);
                if (type == "ShapeOf" && !ConvertNodeShapeOf(node, layer))
                    return ErrorMessage(node);
                if (type == "Sigmoid" && !ConvertNodeSigmoid(node, layer))
                    return ErrorMessage(node);
                if (type == "Unsqueeze" && !ConvertNodeUnsqueeze(node, network.layers(), original, layer))
                    return ErrorMessage(node);

#if 1
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(node);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(node, layer);
                    std::cout << "Not implemented layer : ";
                    node.write_description(std::cout, 1) << std::endl;
                }
#endif
                network.layers().push_back(layer);
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
            Shape src0 = GetInputShape(node, 0);
            Shape src1 = GetInputShape(node, 1);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL)
                return false;
            if (second->type() == LayerTypeReshape && second->src().size() == 1)
            {
                const LayerParam* prev = GetLayer(layers, second->src()[0]);
                if (prev == NULL)
                    return false;
                if (prev->type() == LayerTypeConst && TensorSize(src0) >= TensorSize(src1))
                    second = prev;
            }
            if (second->type() == LayerTypeConst && (TensorSize(src0) >= TensorSize(src1) || src0.size() >= src1.size()))
            {
                if (TensorSize(src1) == 1)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().shift() = GetWeight<float>(original, second->weight()[0])[0];
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
                if (TensorSize(src0) < TensorSize(src1) && src0.size() <= src1.size())
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertNodeAvgPool(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            const ngraph::op::v1::AvgPool* ap = (ngraph::op::v1::AvgPool*)&node;
            layer.pooling().kernel() = ap->get_kernel();
            layer.pooling().stride() = ap->get_strides();
            layer.pooling().pad() = Shp(
                ap->get_pads_begin()[0], ap->get_pads_begin()[1],
                ap->get_pads_end()[0], ap->get_pads_end()[1]);
            const Shape & out = GetOutputShape(node, 0);
            if (out[2] == 1 && out[3] == 1 && layer.pooling().stride() == Shp(1, 1))
                layer.pooling().globalPooling() = true;
            layer.src().resize(1);
            return true;
        }

        bool ConvertNodeBatchNormInference(const ngraph::Node& node, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            if (!CheckSourceNumber(layer, 5))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL || src0->type() != LayerTypeConst)
                return false;
            const float* gamma = GetWeight<float>(original, src0->weight()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            const float* beta = GetWeight<float>(original, src1->weight()[0]);
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                return false;
            const float* mean = GetWeight<float>(original, src3->weight()[0]);
            const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
            if (src4 == NULL || src4->type() != LayerTypeConst)
                return false;
            const float* var = GetWeight<float>(original, src4->weight()[0]);
            const float eps = (float)((ngraph::op::v0::BatchNormInference*)&node)->get_eps_value();

            layer.type() = Synet::LayerTypeScale;
            layer.scale().biasTerm() = true;
            layer.src() = Strings( { layer.src()[2] } );
            layer.weight().resize(2);
            layer.weight()[0] = src3->weight()[0];
            layer.weight()[1] = src4->weight()[0];
            float* scale = GetWeight<float>(reordered, layer.weight()[0]);
            float* shift = GetWeight<float>(reordered, layer.weight()[1]);
            size_t channels = layer.weight()[0].dim()[0];
            for (size_t c = 0; c < channels; c++)
            {
                scale[c] = gamma[c] / sqrt(var[c] + eps);
                shift[c] = -scale[c] * mean[c] + beta[c];
            }
            return true;
        }

        bool ConvertNodeClamp(const ngraph::Node& node, LayerParam& layer)
        {
            const ngraph::op::v0::Clamp* clamp = (ngraph::op::v0::Clamp*)&node;
            layer.type() = Synet::LayerTypeRestrictRange;
            layer.restrictRange().lower() = (float)clamp->get_min();
            layer.restrictRange().upper() = (float)clamp->get_max();
            return true;
        }

        bool ConvertNodeConcat(const ngraph::Node& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypePack;
            }
            else
            {
                layer.type() = Synet::LayerTypeConcat;
                layer.concat().axis()= ((ngraph::op::v0::Concat*)&node)->get_axis();
                if (trans && !PermutedToNchw(layers, false, true))
                {
                    Shape input = node.get_input_shape(0);
                    if (input.size() == 4)
                    {
                        Shape nchw = Shape({ 0, 3, 1, 2 });
                        layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                    }
                    else if (input.size() == 3)
                    {
                        Shape ncs = Shape({ 0, 2, 1 });
                        layer.concat().axis() = (uint32_t)ncs[layer.concat().axis()];
                    }
                    else
                        return false;
                }
                layer.concat().fixed() = true;
                for (size_t i = 0; i < layer.src().size() && layer.concat().fixed(); ++i)
                {
                    const LayerParam* src = GetLayer(layers, layer.src()[i]);
                    if (src == NULL)
                        return false;
                    if (src->type() != LayerTypePriorBox &&
                        src->type() != LayerTypePriorBoxClustered)
                        layer.concat().fixed() = false;
                }
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
                if (layer.weight()[0].dim().empty())
                    layer.weight()[0].dim() = Shp(1);
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
            case ngraph::element::Type_t::u64:
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

        bool ConvertNodeConvolution(const ngraph::Node& node, bool trans, const LayerParams & layers, const Vector& original, LayerParam& layer, Vector& reordered)
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
            if (!CheckDims(shape, 4, "convolution weight"))
                return false;
            layer.convolution().kernel() = Shape({ shape[2], shape[3] });
            layer.convolution().outputNum() = (uint32_t)shape[0];
            layer.src().resize(1);
            if (trans)
                return ReorderWeight(original, Shape(), layer, reordered);
            return true;
        }

        bool ConvertNodeGather(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeGather;
            return true;
        }

        bool ConvertNodeGroupConvolution(const ngraph::Node& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            layer.type() = Synet::LayerTypeConvolution;
            layer.convolution().biasTerm() = false;
            const ngraph::op::v1::GroupConvolution* gc = (ngraph::op::v1::GroupConvolution*)&node;
            layer.convolution().stride() = gc->get_strides();
            layer.convolution().dilation() = gc->get_dilations();
            if (gc->get_auto_pad() == ngraph::op::PadType::SAME_UPPER)
                layer.convolution().autoPad() = true;
            layer.convolution().pad() = Shp(
                gc->get_pads_begin()[0], gc->get_pads_begin()[1],
                gc->get_pads_end()[0], gc->get_pads_end()[1]);
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeReshape)
                return false;
            const LayerParam* previous = GetLayer(layers, second->src()[0]);
            if (previous == NULL || previous->type() != LayerTypeConst)
                return false;
            const Shape& shape = previous->weight()[0].dim();
            layer.weight() = previous->weight();
            if (!CheckDims(shape, 4, "group convolution weight"))
                return false;
            layer.convolution().kernel() = Shape({ shape[2], shape[3] });
            layer.convolution().group() = (uint32_t)shape[0];
            layer.convolution().outputNum() = (uint32_t)shape[0];
            layer.src().resize(1);
            if (trans)
                return ReorderWeight(original, Shape(), layer, reordered);
            return true;
        }

        bool ConvertNodeMatMul(const ngraph::Node& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            const ngraph::op::v0::MatMul* mm = (ngraph::op::v0::MatMul*)&node;
            bool transposeA = mm->get_transpose_a();
            bool transposeB = mm->get_transpose_b();
            layer.innerProduct().biasTerm() = false;
            layer.innerProduct().transposeA() = transposeA;
            layer.innerProduct().transposeB() = !transposeB;
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeConst)
                return false;
            const Shape& weight = second->weight()[0].dim();
            if (!CheckDims(weight, 2, "inner product weight"))
                return false;
            layer.weight() = second->weight();
            layer.innerProduct().outputNum() = (uint32_t)(transposeB ? weight[0] : weight[1]);
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, true, false))
            {
                const LayerParam* first = GetLayer(layers, layer.src()[0]);
                if (first == NULL)
                    return false;
                if (first->type() == LayerTypePooling && first->pooling().globalPooling())
                    return true;
                if (first->type() != LayerTypeReshape)
                    return false;
                Shape origin = GetInputShape(*node.get_input_node_ptr(0), 0);
                return ReorderWeight(original, origin, layer, reordered);
            }
            return true;
        }

        bool ConvertNodeMaxPool(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeMax;
            const ngraph::op::v1::MaxPool* mp = (ngraph::op::v1::MaxPool*)&node;
            layer.pooling().kernel() = mp->get_kernel();
            layer.pooling().stride() = mp->get_strides();
            layer.pooling().pad() = Shp(
                mp->get_pads_begin()[0], mp->get_pads_begin()[1],
                mp->get_pads_end()[0], mp->get_pads_end()[1]);
            if (mp->get_rounding_type() == ngraph::op::RoundingType::FLOOR)
                layer.pooling().roundingType() = RoundingTypeFloor;
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
                if (first->type() == LayerTypeConst && 
                    (second->type() != LayerTypeConst || 
                    TensorSize(first->weight()[0].dim()) == 1))
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
                if (first->type() == LayerTypeConst && layer.type() == Synet::LayerTypePower && !layer.power.Changed())
                {
                    layer.type() = LayerTypeConst;
                    layer.weight() = first->weight();
                    layer.src().resize(0);
                }
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
                Shape shape = GetOutputShape(node, i);
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

        bool ConvertNodeReshape(const ngraph::Node& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
                return false;
            if (second->meta().type() == MetaTypeConst)
            {
                Shape input = node.get_input_shape(0);
                Shape output = node.get_output_shape(0);
                if (second->meta().alpha().shape().size() != 1)
                    return false;
                if (!CheckDims(output, second->meta().alpha().shape()[0], "output shape"))
                    return false;
                Shape& shape = layer.reshape().shape();
                const int64_t* alpha = second->meta().alpha().i64().data();
                layer.type() = LayerTypeReshape;
                shape.resize(output.size());
                for (size_t i = 0; i < shape.size(); ++i)
                    shape[i] = (size_t)alpha[i];
                layer.src().resize(1);
                if (trans && !PermutedToNchw(layers, true, false))
                {
                    if (shape.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                    }
                }
                if (input.size() > 1 && output.size() > 1 && input[0] == 1 && output[0] == 1)
                {
                    layer.reshape().axis() = 1;
                    shape.erase(shape.begin(), shape.begin() + 1);
                }
            }
            else if (first->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReshape;
            }
            else
            {
                layer.type() = LayerTypeReshape;
            }
            return true;
        }

        bool ConvertNodeResult(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeStub;
            if (layer.dst().empty())
                layer.dst().push_back(layer.name());
            return true;
        }

        bool ConvertNodeShapeOf(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeShape;
            layer.meta().version() = 1;
            return true;
        }

        bool ConvertNodeSigmoid(const ngraph::Node& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        bool ConvertNodeUnsqueeze(const ngraph::Node& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            if (first->type() == LayerTypePriorBoxClustered || first->type() == LayerTypePriorBox)
            {
                layer.type() = Synet::LayerTypeStub;
                layer.src().resize(1);
            }
            else
            {
                const LayerParam* second = GetLayer(layers, layer.src()[1]);
                if (second == NULL || second->type() != LayerTypeMeta || second->meta().type() != MetaTypeConst)
                    return false;
                if (first->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeExpandDims;
                }
                else
                {
                    const int64_t* alpha = second->meta().alpha().i64().data();
                    layer.type() = Synet::LayerTypeExpandDims;
                    layer.expandDims().axis() = (int32_t)alpha[0];
                    layer.src().resize(1);
                }
            }
            return true;
        }

        //---------------------------------------------------------------------

        static Shape GetOutputShape(const ngraph::Node& node, size_t index)
        {
            const ngraph::PartialShape & ps = node.get_output_partial_shape(index);
            if (ps.is_static())
                return ps.get_shape();
            Shape shape = ps.get_min_shape();
            if (shape.size() && shape[0] == 0)
                shape[0] = 1;
            return shape;
        }

        static Shape GetInputShape(const ngraph::Node& node, size_t index)
        {
            const ngraph::PartialShape& ps = node.get_input_partial_shape(index);
            if (ps.is_static())
                return ps.get_shape();
            Shape shape = ps.get_min_shape();
            if (shape.size() && shape[0] == 0)
                shape[0] = 1;
            return shape;
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
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif