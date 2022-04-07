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

#include "Synet/Converters/OnnxCommon.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(const String& srcParamPath, const String& srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath, 
            const OnnxParam& onnxParam, const OptimizerParam& optParam)
        {
            if (!Cpl::FileExists(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            onnx::ModelProto model;
            if (!LoadModel(srcGraphPath, model))
                return false;

            Synet::NetworkParamHolder holder;
            Vector weight;
            if (!ConvertModel(model, trans, onnxParam, holder(), weight))
                return false;

            Optimizer optimizer(optParam);
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

        typedef std::map<String, String> Renames;
        typedef std::set<String> Consts;

        bool LoadModel(const String& path, onnx::ModelProto& model)
        {
            std::ifstream ifs(path.c_str(), std::ios::ate | std::ios_base::binary);
            if (!ifs.is_open())
            {
                std::cout << "Can't open file '" << path << "' !" << std::endl;
                return false;
            }
            size_t size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            ifs.read(buffer.data(), size);
            ifs.close();

            if (!model.ParseFromArray(buffer.data(), size))
            {
                std::cout << "Can't parse file '" << path << "' !" << std::endl;
                return false;
            }

            return true;
        }

        bool ConvertModel(const onnx::ModelProto & model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Vector& reordered)
        {
            const onnx::GraphProto& graph = model.graph();

            //PrintGraph(graph, std::cout, false, true);

            network.name() = graph.name();

            Vector original;
            Consts consts;
            for (size_t i = 0; i < graph.initializer_size(); ++i)
            {
                const onnx::TensorProto& tensor = graph.initializer(i);
                if (!ConvertInitializer(tensor, network, original))
                {
                    std::cout << "Can't convert initializer '" << tensor.name() << "' !" << std::endl;
                    return false;
                }
                consts.insert(tensor.name());
            }
            reordered = original;

            for (size_t i = 0; i < graph.input_size(); ++i)
            {
                const onnx::ValueInfoProto& input = graph.input(i);
                if (consts.find(input.name()) != consts.end())
                    continue;
                if (!ConvertInput(input, trans, network))
                {
                    std::cout << "Can't convert input '" << input.name() << "' !" << std::endl;
                    return false;
                }
            }

            Renames renames;
            for (size_t i = 0; i < graph.node_size(); ++i)
            {
                const onnx::NodeProto& node = graph.node(i);
                LayerParam layer;
                SetSrcAndDst(node, renames, layer);

                if (node.op_type() == "Add" && !ConvertAddNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "BatchNormalization" && !ConvertBatchNormalizationNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Cast" && !ConvertCastNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Clip" && !ConvertClipNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Concat" && !ConvertConcatNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Constant" && !ConvertConstantNode(node, layer, original, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Conv" && !ConvertConvNode(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Div" && !ConvertDivNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Exp" && !ConvertExpNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Flatten" && !ConvertFlattenNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Gather" && !ConvertGatherNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Gemm" && !ConvertGemmNode(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "GlobalAveragePool" && !ConvertGlobalAveragePoolNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "LeakyRelu" && !ConvertLeakyReluNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "MatMul" && !ConvertMatMulNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "MaxPool" && !ConvertMaxPoolNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Mul" && !ConvertMulNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "PRelu" && !ConvertPreluNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceMax" && !ConvertReduceMaxNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceMean" && !ConvertReduceMeanNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceSum" && !ConvertReduceSumNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Relu" && !ConvertReluNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Reshape" && !ConvertReshapeNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Resize" && !ConvertResizeNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Shape" && !ConvertShapeNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sigmoid" && !ConvertSigmoidNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Slice" && !ConvertSliceNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Softmax" && !ConvertSoftmaxNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Split" && !ConvertSplitNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Squeeze" && !ConvertSqueezeNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sub" && !ConvertSubNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Transpose" && !ConvertTransposeNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Unsqueeze" && !ConvertUnsqueezeNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);

#if defined(SYNET_ONNX_PARSE_STOP_ON_ERROR)
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(i, node);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(node, layer);
                    std::cout << "Not implemented node[" << i << "]: " << NodeString(node) << std::endl;
                }
#endif
                network.layers().push_back(layer);

                if (trans && !ManualInsertToNchwPermute(onnxParam, network.layers(), renames))
                    return false;
            }

            if (!RemoveUnusedConst(network.layers()))
                return false;

            return true;
        }

        bool ConvertInitializer(const onnx::TensorProto& tensor, Synet::NetworkParam& network, Vector& weight)
        {
            LayerParam layer;
            layer.name() = tensor.name();
            layer.dst().push_back(tensor.name());
            if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
            {
                layer.type() = LayerTypeConst;
                layer.weight().resize(1);
                layer.weight()[0].type() = TensorType32f;
                uint64_t size = 1, offset = weight.size();
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= (size_t)tensor.dims(i);
                    layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
                }
                layer.weight()[0].offset() = offset * sizeof(float);
                layer.weight()[0].size() = size * sizeof(float);
                if (size)
                {
                    if (tensor.has_raw_data())
                    {
                        weight.resize(offset + size);
                        memcpy(weight.data() + offset, tensor.raw_data().c_str(), layer.weight()[0].size());
                    }
                    else if (tensor.float_data_size())
                    {
                        if (size != tensor.float_data_size())
                        {
                            std::cout << "Wrong tensor float_data_size " << tensor.float_data_size() << " != " << size << " !" << std::endl;
                            return false;
                        }
                        weight.resize(offset + size);
                        for (size_t i = 0; i < size; ++i)
                            weight[offset + i] = tensor.float_data(i);
                    }
                    else
                    {
                        std::cout << "Can't parse '" << layer.name() << "' FP32 tensor!" << std::endl;
                        return false;
                    }
                }
            }
            else if(tensor.data_type() == onnx::TensorProto_DataType_INT64)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeConst;
                layer.meta().alpha().type() = TensorType64i;
                uint64_t size = 1;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= (size_t)tensor.dims(i);
                    layer.meta().alpha().shape().push_back(size_t(tensor.dims(i)));
                }
                layer.meta().alpha().i64().resize(size);
                if (size)
                {
                    if (tensor.has_raw_data())
                    {
                        for (size_t i = 0; i < size; ++i)
                            layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
                    }
                    else
                    {
                        std::cout << "Can't parse '" << layer.name() << "' INT64 tensor!" << std::endl;
                        return false;
                    }
                }
            }
            else
            {
                std::cout << " Unknown tensor type " << tensor.data_type() << " !" << std::endl;
                return false;
            }
            network.layers().push_back(layer);
            return true;
        }

        bool ConvertInput(const onnx::ValueInfoProto & input, bool trans, Synet::NetworkParam& network)
        {
            LayerParam layer;
            layer.type() = LayerTypeInput;
            layer.name() = input.name();
            layer.dst().push_back(input.name());
            layer.input().shape().resize(1);
            Shape shape = Convert(input.type().tensor_type().shape());
            if (trans)
            {
                if (shape.size() == 4)
                {
                    shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                    layer.input().shape()[0].format() = TensorFormatNhwc;
                }
            }
            if (shape.size() > 1 && shape[0] == -1)
                shape[0] = 1;
            layer.input().shape()[0].dim() = shape;
            network.layers().push_back(layer);
            return true;
        }

        void SetSrcAndDst(const onnx::NodeProto& node, const Renames &renames, LayerParam& layer)
        {
            layer.name() = node.name();
            for (size_t j = 0; j < node.input_size(); ++j)
            {
                String input = node.input(j);
                Renames::const_iterator rename = renames.find(input);
                if (rename != renames.end())
                    input = rename->second;
                layer.src().push_back(input);
            }
            for (size_t j = 0; j < node.output_size(); ++j)
                layer.dst().push_back(node.output(j));
            layer.name() = layer.dst()[0];
        }

        bool ManualInsertToNchwPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames)
        {
            LayerParam& layer = layers.back();
            for (size_t h = 0; h < onnxParam.toNchwHints().size(); ++h)
            {
                if (layer.name() == onnxParam.toNchwHints()[h])
                {
                    for (size_t d = 0; d < layer.dst().size(); ++d)
                    {
                        const String& dst = layer.dst()[d];
                        LayerParam permute;
                        permute.type() = LayerTypePermute;
                        permute.src().push_back(dst);
                        permute.name() = dst + "_permute_to_nchw";
                        permute.dst().push_back(permute.name());
                        permute.permute().order() = Shape({ 0, 3, 1, 2 });
                        permute.permute().format() = TensorFormatNchw;
                        layers.push_back(permute);
                        if (renames.find(dst) != renames.end())
                        {
                            std::cout << "Multiple manual NhwcToNchw permute at " << layer.name() << " !" << std::endl;
                            return false;
                        }
                        renames[dst] = permute.name();
                    }
                }
            }
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool ConvertAddNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = Synet::LayerTypeEltwise;
            layer.eltwise().operation() = EltwiseOperationTypeSum;
            return true;
        }

        bool ConvertBatchNormalizationNode(const onnx::NodeProto & node, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            if (!CheckSourceNumber(layer, 5))
                return false;

            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            const float* gamma = GetWeight<float>(original, src1->weight()[0]);

            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            const float* beta = GetWeight<float>(original, src2->weight()[0]);

            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                return false;
            const float* mean = GetWeight<float>(original, src3->weight()[0]);

            const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
            if (src4 == NULL || src4->type() != LayerTypeConst)
                return false;
            const float* var = GetWeight<float>(original, src4->weight()[0]);

            float epsilon, momentum;
            if (!ConvertAtrributeFloat(node, "epsilon", epsilon))
                return false;
            if (!ConvertAtrributeFloat(node, "momentum", momentum))
                return false;

            layer.type() = Synet::LayerTypeScale;
            layer.src().resize(1);
            layer.scale().biasTerm() = true;
            layer.weight().resize(2);
            layer.weight()[0] = src3->weight()[0];
            layer.weight()[1] = src4->weight()[0];
            float* scale = GetWeight<float>(reordered, layer.weight()[0]);
            float* shift = GetWeight<float>(reordered, layer.weight()[1]);
            size_t channels = layer.weight()[0].dim()[0];
            for (size_t c = 0; c < channels; c++)
            {
                scale[c] = gamma[c] / sqrt(var[c] + epsilon);
                shift[c] = -scale[c] * mean[c] + beta[c];
            }
            return true;
        }

        bool ConvertCastNode(const onnx::NodeProto& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            int to;
            if (!ConvertAtrributeInt(node, "to", to))
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeCast;
                if (to == onnx::TensorProto_DataType_FLOAT)
                    layer.meta().alpha().type() = TensorType32f;
                else if (to == onnx::TensorProto_DataType_INT32)
                    layer.meta().alpha().type() = TensorType32i;
                else if (to == onnx::TensorProto_DataType_INT64)
                    layer.meta().alpha().type() = TensorType64i;
                else
                    return false;
            }
            else
            {
                layer.type() = Synet::LayerTypeCast;
                if (to == onnx::TensorProto_DataType_FLOAT)
                    layer.cast().type() = TensorType32f;
                else if (to == onnx::TensorProto_DataType_INT32)
                    layer.cast().type() = TensorType32i;
            }
            return true;
        }

        bool ConvertClipNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            if (!ConvertAtrributeFloat(node, "max", layer.restrictRange().upper()))
                return false;
            if (!ConvertAtrributeFloat(node, "min", layer.restrictRange().lower()))
                return false;
            return true;
        }

        bool ConvertConcatNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0->type() == Synet::LayerTypeMeta || (src1 && src1->type() == Synet::LayerTypeMeta))
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypePack;
            }
            else
            {
                layer.type() = Synet::LayerTypeConcat;
                if (!ConvertAtrributeInt(node, "axis", layer.concat().axis()))
                    return false;
                if (trans && !PermutedToNchw(layers, false, true, true))
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                }
            }
            return true;
        }

        bool ConvertConstantNode(const onnx::NodeProto& node, LayerParam& layer, Vector& original, Vector& reordered)
        {
            String name = "value";
            const onnx::AttributeProto * value = GetAtrribute(node, name);
            if (value == NULL)
            {
                std::cout << "Can't find attribute " << name << " !" << std::endl;
                return false;
            }
            if (value->type() != onnx::AttributeProto_AttributeType_TENSOR)
            {
                std::cout << "Attribute has wrong type " << value->type() << " !" << std::endl;
                return false;
            }
            const onnx::TensorProto& tensor = value->t();
            if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeConst;
                layer.meta().alpha().type() = TensorType64i;
                uint64_t size = 1;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= tensor.dims(i);
                    layer.meta().alpha().shape().push_back(size_t(tensor.dims(i)));
                }
                if (tensor.dims_size() == 0)
                    layer.meta().alpha().shape().push_back(1);
                layer.meta().alpha().i64().resize(size);
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < size; ++i)
                        layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
                }
            }
            else if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
            {
                layer.type() = LayerTypeConst;
                layer.weight().resize(1);
                layer.weight()[0].type() = TensorType32f;
                uint64_t size = 1, offset = original.size();
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= tensor.dims(i);
                    layer.weight()[0].dim().push_back(size_t(tensor.dims(i)));
                }
                if (layer.weight()[0].dim().empty())
                    layer.weight()[0].dim().push_back(1);
                layer.weight()[0].offset() = offset * sizeof(float);
                layer.weight()[0].size() = size * sizeof(float);
                if (tensor.has_raw_data() && size)
                {
                    original.resize(offset + size);
                    reordered.resize(offset + size);
                    memcpy(original.data() + offset, tensor.raw_data().c_str(), layer.weight()[0].size());
                    memcpy(reordered.data() + offset, tensor.raw_data().c_str(), layer.weight()[0].size());
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertConvNode(const onnx::NodeProto & node, bool trans, const LayerParams& layers, const Vector& srcBin, LayerParam& layer, Vector& dstBin)
        {
            layer.type() = Synet::LayerTypeConvolution;
            if (layer.src().size() < 2 || layer.src().size() > 3)
                return false;
            if (!ConvertAtrributeInts(node, "dilations", layer.convolution().dilation()))
                return false;
            if(!ConvertAtrributeInt(node, "group", layer.convolution().group()))
                return false;
            if (!ConvertAtrributeInts(node, "kernel_shape", layer.convolution().kernel()))
                return false;
            if (!ConvertAtrributeInts(node, "pads", layer.convolution().pad()))
                return false;
            if (!ConvertAtrributeInts(node, "strides", layer.convolution().stride()))
                return false;
            layer.weight().resize(layer.src().size() - 1);
            const LayerParam* weight = GetLayer(layers, layer.src()[1]);
            if (weight == NULL || weight->type() != LayerTypeConst)
                return false;
            const Shape& shape = weight->weight()[0].dim();
            layer.weight()[0] = weight->weight()[0];
            layer.convolution().outputNum() = (uint32_t)shape[0];
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.convolution().biasTerm())
            {
                const LayerParam* bias = GetLayer(layers, layer.src()[2]);
                if (bias == NULL || bias->type() != LayerTypeConst)
                    return false;
                layer.weight()[1] = bias->weight()[0];
            }
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, false))
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }

        bool ConvertDivNode(const onnx::NodeProto& node, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = 1.0f / pScale[0];
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
            {
                return false;
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.scale().biasTerm() = false;
                layer.weight() = src1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                const float* pSrc = GetWeight<float>(original, layer.weight()[0]);
                float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                size_t size = TensorSize(layer.weight()[0].dim());
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = 1.0 / pSrc[i];
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeDiv;
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeDiv;
            }
            return true;
        }

        bool ConvertExpNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeExp;
            return true;
        }

        bool ConvertFlattenNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeFlatten;
            if (!ConvertAtrributeInt(node, "axis", layer.flatten().axis()))
                return false;
            return true;
        }

        bool ConvertGatherNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeGather;
            return true;
        }

        bool ConvertGemmNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            int transB;
            if (!ConvertAtrributeInt(node, "transB", transB))
                return false;
            layer.innerProduct().transposeB() = !transB;
            if (layer.src().size() < 2 || layer.src().size() > 3)
                return false;
            layer.weight().resize(layer.src().size() - 1);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            const Shape& weight = src1->weight()[0].dim();
            if (!CheckDims(weight, 2, "inner product weight"))
                return false;
            layer.weight()[0] = src1->weight()[0];
            layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[0] : weight[1]);
            if (layer.src().size() > 2)
            {
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                if (src2 == NULL || src2->type() != LayerTypeConst)
                    return false;
                layer.weight()[1] = src2->weight()[0];
            }
            else
                layer.innerProduct().biasTerm() = false;
            layer.src().resize(1);
            return true;
            if (trans && !PermutedToNchw(layers, true, false, true))
            {
                std::cout << "Can 't convert Gemm node for NCHW format!" << std::endl;
                return false;
            }
            return true;
        }

        bool ConvertGlobalAveragePoolNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            layer.pooling().globalPooling() = true;
            return true;
        }

        bool ConvertLeakyReluNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            if (!ConvertAtrributeFloat(node, "alpha", layer.relu().negativeSlope()))
                return false;
            return true;
        }

        bool ConvertMatMulNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = Synet::LayerTypeInnerProduct;
            int transB = false;
            layer.weight().resize(layer.src().size() - 1);
            layer.innerProduct().biasTerm() = false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst)
            {
                layer.weight()[0] = src1->weight()[0];
            }
            else if (src1->type() == LayerTypePermute)
            {
                if (!CheckSourceNumber(*src1, 1))
                    return false;
                const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
                if (src10 == NULL || src10->type() != LayerTypeConst)
                    return false;
                transB = true;
                layer.weight() = src10->weight();
                layers.erase(layers.begin() + (src1 - layers.data()));
            }
            else
                return false;
            Shape weight = layer.weight()[0].dim();
            if (!CheckDims(weight, 2, "inner product weight"))
                return false;
            layer.innerProduct().transposeB() = !transB;
            layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[0] : weight[1]);
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, true, false, true))
            {
                std::cout << "Can 't convert MatMul node for NCHW format!" << std::endl;
                return false;
            }
            return true;
        }

        bool ConvertMaxPoolNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeMax;
            if (!ConvertAtrributeInts(node, "kernel_shape", layer.pooling().kernel()))
                return false;
            if (!ConvertAtrributeInts(node, "pads", layer.pooling().pad()))
                return false;
            if (!ConvertAtrributeInts(node, "strides", layer.pooling().stride()))
                return false;

            if(GetAtrribute(node, "ceil_mode") == NULL)
                layer.pooling().roundingType() = RoundingTypeFloor;
            else
            {
                int ceilMode;
                if (!ConvertAtrributeInt(node, "ceil_mode", ceilMode))
                    return false;
                layer.pooling().roundingType() = ceilMode ? RoundingTypeCeil : RoundingTypeFloor;
            }
            return true;
        }

        bool ConvertMulNode(const onnx::NodeProto& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = pScale[0];
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
            }
            return true;
        }

        bool ConvertPreluNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            layer.type() = Synet::LayerTypePrelu;
            if (src1->type() == LayerTypeConst)
            {
                layer.weight() = src1->weight();
            }
            else if (src1->type() == LayerTypeExpandDims)
            {
                if (!CheckSourceNumber(*src1, 1))
                    return false;
                const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
                if (src10 == NULL || src10->type() != LayerTypeConst)
                    return false;
                layer.weight() = src10->weight();
                layers.erase(layers.begin() + (src1 - layers.data()));
            }
            else
                return false;
            layer.src().resize(1);
            if (!CompactShape(layer.weight()[0].dim()))
                return false;
            return true;
        }

        bool ConvertReduceMaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeMax;
            if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && !PermutedToNchw(layers, false, true, true))
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for(size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
            return true;
        }

        bool ConvertReduceMeanNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeMax;
            Ints axes;
            if (!ConvertAtrributeInts(node, "axes", axes))
                return false;
            if (axes.size() != 2 || axes[0] != 2 || axes[1] != 3)
                return false;
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            layer.pooling().globalPooling() = true;
            return true;
        }

        bool ConvertReduceSumNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeSum;
            if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && !PermutedToNchw(layers, false, true, true))
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for (size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
            return true;
        }

        bool ConvertReluNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertReshapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* first = GetLayer(layers, layer.src()[0]);
            const LayerParam* second = GetLayer(layers, layer.src()[1]);
            if (second == NULL || second->type() != LayerTypeMeta)
                return false;
            if (second->meta().type() == MetaTypeConst)
            {
                if (second->meta().alpha().shape().size() != 1)
                    return false;
                Shape& shape = layer.reshape().shape();
                const int64_t* alpha = second->meta().alpha().i64().data();
                layer.type() = LayerTypeReshape;
                shape.resize(second->meta().alpha().shape()[0]);
                for (size_t i = 0; i < shape.size(); ++i)
                    shape[i] = (size_t)alpha[i];
                layer.src().resize(1);
                if (trans && !PermutedToNchw(layers, layer.src(), true, false, true))
                {
                    if (shape.size() == 5)
                    {
                        shape = Shape({ shape[0], shape[3], shape[4], shape[1], shape[2] });
                    }
                    if (shape.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                    }
                    if (shape.size() == 3)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[1] });
                    }
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

        bool ConvertResizeNode(const onnx::NodeProto& node, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            if (layer.src().size() == 3)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                    return false;
                layer.src().erase(layer.src().begin() + 1);
            }

            String mode;
            if (!ConvertAtrributeString(node, "mode", mode))
                return false;
            if (mode == "nearest")
                layer.interp().interpolationType() = InterpolationTypeNearest;
            else if (mode == "linear")
                layer.interp().interpolationType() = InterpolationTypeBilinear;
            else
                return false;

            if (GetAtrribute(node, "coordinate_transformation_mode"))
            {
                String coordTransf;
                if (!ConvertAtrributeString(node, "coordinate_transformation_mode", coordTransf))
                    return false;
                if (coordTransf == "pytorch_half_pixel")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypeHalfPixel;
                else if (coordTransf == "asymmetric")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypePytorch;
                else
                    return false;
            }

            layer.type() = Synet::LayerTypeInterp;
            return true;
        }

        bool ConvertShapeNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeShape;
            layer.meta().version() = 1;
            return true;
        }

        bool ConvertSigmoidNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        bool ConvertSliceNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (layer.src().size() == 1)
            {
                if (!ConvertAtrributeInts(node, "axes", layer.stridedSlice().axes()))
                    return false;
                if (!ConvertAtrributeInts(node, "starts", layer.stridedSlice().beginDims()))
                    return false;
                if (!ConvertAtrributeInts(node, "ends", layer.stridedSlice().endDims()))
                    return false;
                layer.type() = Synet::LayerTypeStridedSlice;
            }
            else if (layer.src().size() == 4)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
                if (src0 == NULL || src1 == NULL || src2 == NULL || src3 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = MetaTypeSlice;
                }
                else
                    return false;
            }
            else
            {
                if (!CheckSourceNumber(layer, 5))
                    return false;
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
                const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
                if (src0 == NULL || src1 == NULL || src2 == NULL || src3 == NULL || src4 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                    return false;
                if (src1->type() != LayerTypeMeta || src1->meta().type() != Synet::MetaTypeConst || src1->meta().alpha().i64().size() != 1)
                    return false;
                if (src2->type() != LayerTypeMeta || src2->meta().type() != Synet::MetaTypeConst || src2->meta().alpha().i64().size() != 1)
                    return false;
                if (src3->type() != LayerTypeMeta || src3->meta().type() != Synet::MetaTypeConst || src3->meta().alpha().i64().size() != 1)
                    return false;
                if (src4->type() != LayerTypeMeta || src4->meta().type() != Synet::MetaTypeConst || src4->meta().alpha().i64().size() != 1)
                    return false;
                layer.type() = Synet::LayerTypeStridedSlice;
                layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                layer.stridedSlice().beginDims().push_back((size_t)src1->meta().alpha().i64()[0]);
                layer.stridedSlice().endDims().push_back((size_t)src2->meta().alpha().i64()[0]);
                layer.stridedSlice().strideDims().push_back((size_t)src4->meta().alpha().i64()[0]);
                if (trans && !PermutedToNchw(layers, false, true, true))
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertSoftmaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Vector& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            if (!ConvertAtrributeInt(node, "axis", layer.softmax().axis()))
                return false;
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                return false;
            }
            return true;
        }

        bool ConvertSplitNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            if (!ConvertAtrributeInt(node, "axis", layer.unpack().axis()))
                return false;
            if (!ConvertAtrributeInts(node, "split", layer.unpack().parts()))
                return false;
            layer.type() = Synet::LayerTypeUnpack;
            if (trans && !PermutedToNchw(layers, true, false, true))
            {
                Shape nchw = Shape({ 0, 3, 1, 2 });
                layer.unpack().axis() = nchw[layer.unpack().axis()];
            }
            return true;
        }

        bool ConvertSqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSqueeze;
            }
            else
            {
                layer.type() = Synet::LayerTypeSqueeze;
                if (!ConvertAtrributeInts(node, "axes", layer.squeeze().axes()))
                    return false;
            }
            return true;
        }

        bool ConvertSubNode(const onnx::NodeProto& node, const LayerParams& layers, const Vector& original, LayerParam& layer, Vector& reordered)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pShift = GetWeight<float>(original, src1->weight()[0]);
                layer.power().shift() = -pShift[0];
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                layer.power().scale() = -1.0f;
                const float* pShift = GetWeight<float>(original, src0->weight()[0]);
                layer.power().shift() = pShift[0];
                layer.src()[0] = layer.src()[1];
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypeBias;
                layer.weight() = src1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                const float* pSrc = GetWeight<float>(original, layer.weight()[0]);
                float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                size_t size = TensorSize(layer.weight()[0].dim());
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = -pSrc[i];
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeSub;
            }
            return true;
        }

        bool ConvertTransposeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            layer.type() = Synet::LayerTypePermute;
            Shape order;
            if (!ConvertAtrributeInts(node, "perm", order))
                return false;
            if (trans && !PermutedToNchw(layers, true, false, true))
            {
                if (order == Shape({ 0, 2, 1, 3, 4 }))
                    order = Shape({ 0, 1, 2, 4, 3 });
                if (order == Shape({ 0, 2, 3, 1 }))
                {
                    order = Shape({ 0, 1, 2, 3 });
                    layer.permute().format() = TensorFormatNchw;
                }
                if (order == Shape({ 0, 2, 1 }))
                {
                    order = Shape({ 0, 1, 2 });
                    layer.permute().format() = TensorFormatNchw;
                }
            }
            layer.permute().order() = order;
            return true;
        }

        bool ConvertUnsqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeExpandDims;
                layer.meta().alpha().type() = TensorType64i;
                if (!ConvertAtrributeInts(node, "axes", layer.meta().alpha().i64()))
                    return false;
                layer.meta().alpha().shape().resize(1, layer.meta().alpha().i64().size());
            }
            else
            {
                layer.type() = Synet::LayerTypeExpandDims;
                if (!ConvertAtrributeInts(node, "axes", layer.expandDims().axes()))
                    return false;
            }
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool PrintGraph(const onnx::GraphProto& graph, std::ostream & os, bool printConst = false, bool filterInput = true)
        {
            os << std::endl;
            os << "graph name: " << graph.name() << std::endl;
            Consts consts;
            for (size_t i = 0; i < graph.initializer_size(); ++i)
                consts.insert(graph.initializer(i).name());
            for (size_t i = 0; i < graph.input_size(); ++i)
            {
                if (filterInput && consts.find(graph.input(i).name()) != consts.end())
                    continue;
                os << " input[" << i << "] " << ValueInfoString(graph.input(i)) << std::endl;
            }
            if (printConst)
            {
                for (size_t i = 0; i < graph.initializer_size(); ++i)
                    os << " const[" << i << "] " << TensorString(graph.initializer(i), 5) << std::endl;
            }
            for (size_t i = 0; i < graph.node_size(); ++i)
                os << " node[" << i << "] " << NodeString(graph.node(i)) << std::endl;
            for (size_t i = 0; i < graph.output_size(); ++i)
                os << " output[" << i << "] " << ValueInfoString(graph.output(i)) << std::endl;
            os << std::endl;
            return true;
        }

        String ValueInfoString(const onnx::ValueInfoProto& info)
        {
            std::stringstream ss;
            ss << info.name();
            if (info.type().has_tensor_type())
            {
                Shape shape = Convert(info.type().tensor_type().shape());
                ss << " {";
                for (size_t j = 0; j < shape.size(); ++j)
                    ss << " " << ptrdiff_t(shape[j]);
                ss << " }";
            }
            if (info.type().has_sequence_type())
            {
                ss <<  " ValueInfoString: I can't print sequence!";
            }
            if (info.type().has_map_type())
            {
                ss << " ValueInfoString: I can't print map!";
            }
            return ss.str();
        }

        String TensorString(const onnx::TensorProto& tensor, size_t printSizeMax = 3)
        {
            std::stringstream ss;
            ss << tensor.name() << " ";
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT: ss << "f32"; break;
            case onnx::TensorProto_DataType_INT64: ss << "i64"; break;
            default: ss << " unknown-" << tensor.data_type();
            }
            if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
                ss << " external";
            ss << " {";
            uint64_t size = 1;
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                ss << " " << tensor.dims(i);
                size *= tensor.dims(i);
            }
            size_t printSize = std::min<size_t>(printSizeMax, size);
            ss << " }";

            ss << "[";
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT: 
            {
                ss << std::fixed << std::setprecision(3);
                if (tensor.float_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.float_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((float*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_INT64:
            {
                if (tensor.int64_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int64_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((int64_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            }
            if (size > printSize)
                ss << " ...";
            ss << " ]";
            return ss.str();
        }

        String AttributeString(const onnx::AttributeProto& attribute)
        {
            std::stringstream ss;
            ss << attribute.name() << ":(";
            switch (attribute.type())
            {
            case onnx::AttributeProto_AttributeType_INT:
                ss << attribute.i();
                break;
            case onnx::AttributeProto_AttributeType_FLOAT:
                ss << attribute.f();
                break;
            case onnx::AttributeProto_AttributeType_STRING:
                ss << attribute.s();
                break;
            case onnx::AttributeProto_AttributeType_TENSOR:
                ss << TensorString(attribute.t());
                break;
            case onnx::AttributeProto_AttributeType_INTS:
                for(size_t i = 0; i < attribute.ints_size(); ++i)
                    ss << (i ? " " : "") << attribute.ints(i);
                break;
            default:
                ss << "unknown-" << attribute.type();
            }
            ss << ")";
            return ss.str();
        }

        String NodeString(const onnx::NodeProto& node)
        {
            std::stringstream ss;
            ss << "type: " << node.op_type() << ", name: " << node.name() << " (";
            for (size_t j = 0; j < node.input_size(); ++j)
                ss << " " << node.input(j);
            ss << " ) -> (";
            for (size_t j = 0; j < node.output_size(); ++j)
                ss << " " << node.output(j);
            ss << " ) {";
            for (size_t j = 0; j < node.attribute_size(); ++j)
                ss << " " << AttributeString(node.attribute(j));
            ss << " }";
            return ss.str();
        }

        Shape Convert(const onnx::TensorShapeProto& shapeProto)
        {
            Shape shape;
            for (size_t i = 0; i < shapeProto.dim_size(); ++i)
            {
                if (shapeProto.dim(i).has_dim_value())
                    shape.push_back((size_t)shapeProto.dim(i).dim_value());
                else
                    shape.push_back(size_t(-1));
            }
            return shape;
        }

        void NotImplemented(const onnx::NodeProto& node, LayerParam& dst)
        {
            //dst.type() = LayerTypeStub;
            dst.debug().clear();
            dst.debug().push_back(NotImplementedMarker());
            dst.debug().push_back(node.op_type());
        }

        bool ErrorMessage(size_t index, const onnx::NodeProto& node)
        {
            std::cout << "Can't convert node[" << index << "]: " << NodeString(node) << " !" << std::endl;
            return false;
        }

        const onnx::AttributeProto * GetAtrribute(const onnx::NodeProto& node, const String& name)
        {
            for (size_t i = 0; i < node.attribute_size(); ++i)
                if (node.attribute(i).name() == name)
                    return &node.attribute(i);
            return NULL;
        }

        template<class T> bool ConvertAtrributeInt(const onnx::NodeProto& node, const String& name, T & value, bool optional = false, const T & defVal = T())
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
                if (optional)
                {
                    value = defVal;
                    return true;
                }
                std::cout << "Can't find attribute " << name << " !" << std::endl;
                return false;
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_INT)
            {
                std::cout << "Attribute " << name << " has wrong type " << attribute->type() << " !" << std::endl;
                return false;
            }
            value = attribute->i();
            return true;
        }

        bool ConvertAtrributeFloat(const onnx::NodeProto& node, const String& name, float & value, bool optional = false, const float & defVal = float())
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
                if (optional)
                {
                    value = defVal;
                    return true;
                }
                std::cout << "Can't find attribute " << name << " !" << std::endl;
                return false;
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_FLOAT)
            {
                std::cout << "Attribute " << name << " has wrong type " << attribute->type() << " !" << std::endl;
                return false;
            }
            value = attribute->f();
            return true;
        }

        bool ConvertAtrributeString(const onnx::NodeProto& node, const String& name, String & value, bool optional = false, const String & defVal = String())
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
                if (optional)
                {
                    value = defVal;
                    return true;
                }
                std::cout << "Can't find attribute " << name << " !" << std::endl;
                return false;
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_STRING)
            {
                std::cout << "Attribute " << name << " has wrong type " << attribute->type() << " !" << std::endl;
                return false;
            }
            value = attribute->s();
            return true;
        }

        template<class T> bool ConvertAtrributeInts(const onnx::NodeProto& node, const String& name, std::vector<T>& values, 
            bool optional = false, const std::vector<T> & defVals = std::vector<T>())
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
                if (optional)
                {
                    values = defVals;
                    return true;
                }
                std::cout << "Can't find attribute " << name << " !" << std::endl;
                return false;
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_INTS)
            {
                std::cout << "Attribute " << name << " has wrong type " << attribute->type() << " !" << std::endl;
                return false;
            }
            values.resize(attribute->ints_size());
            for(size_t i = 0; i < attribute->ints_size(); ++i)
                values[i] = (T)attribute->ints(i);
            return true;
        }
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, 
        const OnnxParam& onnxParam, const OptimizerParam& optParam)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin, onnxParam, optParam);
    }
}

#endif