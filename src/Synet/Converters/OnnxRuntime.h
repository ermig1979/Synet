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

//#define SYNET_ONNX_PARSE_STOP_ON_ERROR

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"
#include "Synet/Converters/SynetUtils.h"
#include "Synet/Utils/FileUtils.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(String srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath, 
            const OnnxParam& onnxParam, const OptimizerParam& optParam)
        {
            if (!Cpl::FileExists(srcGraphPath))
            {
                String altGraphPath = Cpl::ChangeExtension(srcGraphPath, ".dat");
                if (altGraphPath != srcGraphPath)
                {
                    if(!Cpl::FileExists(altGraphPath))
                        SYNET_ERROR("Files '" << srcGraphPath << "' and '" << altGraphPath << "' are not exist!");
                    srcGraphPath = altGraphPath;
                }
                else
                    SYNET_ERROR("File '" << srcGraphPath << "' is not exist!");
            }

            onnx::ModelProto model;
            if (!LoadModel(srcGraphPath, model))
                return false;

            Synet::NetworkParamHolder holder;
            Bytes weight;
            if (!ConvertModel(model, trans, onnxParam, holder(), weight))
            {
                String errModelPath = Cpl::FileNameByPath(dstModelPath) == dstModelPath ?
                    "error.xml" : Cpl::MakePath(Cpl::DirectoryByPath(dstModelPath), "error.xml");
                if (!holder.Save(errModelPath, false))
                    SYNET_ERROR("Can't save Synet model with conversion error '" << errModelPath << "' !");
                SYNET_ERROR("There is Synet model conversion error! Partial converted model is saved to '" << errModelPath << "'.");
            }

            if (optParam.saveUnoptimized())
            {
                String uoModelPath = Cpl::FileNameByPath(dstModelPath) == dstModelPath ? 
                    "unopt.xml" : Cpl::MakePath(Cpl::DirectoryByPath(dstModelPath), "unopt.xml");
                if (!holder.Save(uoModelPath, false))
                    SYNET_ERROR("Can't save unoptimized Synet model '" << uoModelPath << "' !");

                String uoWeightPath = Cpl::FileNameByPath(dstWeightPath) == dstWeightPath ?
                    "unopt.bin" : Cpl::MakePath(Cpl::DirectoryByPath(dstWeightPath), "unopt.bin");
                if (!SaveBinaryData(weight, uoWeightPath))
                    SYNET_ERROR("Can't save unoptimized Synet weight '" << uoWeightPath << "' !");
            }

            Optimizer optimizer(optParam);
            if (!optimizer.Run(holder(), weight))
                SYNET_ERROR("Can't perform Synet model optimization!");

            if (!holder.Save(dstModelPath, false))
                SYNET_ERROR("Can't save Synet model '" << dstModelPath << "' !");

            if (!SaveBinaryData(weight, dstWeightPath))
                SYNET_ERROR("Can't save Synet weight '" << dstWeightPath << "' !");

            return true;
        }

    private:

        typedef std::map<String, String> Renames;
        typedef std::set<String> Consts;

        bool LoadModel(const String& path, onnx::ModelProto& model)
        {
            std::ifstream ifs(path.c_str(), std::ios::ate | std::ios_base::binary);
            if (!ifs.is_open())
                SYNET_ERROR("Can't open file '" << path << "' !");
            size_t size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            ifs.read(buffer.data(), size);
            ifs.close();
            if (!model.ParseFromArray(buffer.data(), size))
                SYNET_ERROR("Can't parse file '" << path << "' !");
            return true;
        }

        bool ConvertModel(const onnx::ModelProto & model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Bytes& reordered)
        {
            const onnx::GraphProto& graph = model.graph();

            //PrintGraph(graph, std::cout, true, true);

            network.info().version() = 1;
            network.info().name() = graph.name();
            network.info().from() = "OnnxRuntime";
            network.info().when() = Cpl::CurrentDateTimeString();
            network.info().synet() = Synet::Version();

            network.layers().reserve(graph.initializer_size() + graph.input_size() + graph.node_size() * 2);

            Bytes original;
            Consts consts;
            Renames renames;
            PermuteMap permuteMap;
            TensorFormatMap tensorFormatMap;
            for (size_t i = 0; i < graph.initializer_size(); ++i)
            {
                const onnx::TensorProto& tensor = graph.initializer(i);
                if (!ConvertInitializer(tensor, network, original, renames))
                    SYNET_ERROR("Can't convert initializer '" << tensor.name() << "' !");
                consts.insert(tensor.name());
            }
            reordered = original;

            for (size_t i = 0; i < graph.input_size(); ++i)
            {
                const onnx::ValueInfoProto& input = graph.input(i);
                if (consts.find(input.name()) != consts.end())
                    continue;
                if (!ConvertInput(input, trans, network, renames))
                    SYNET_ERROR("Can't convert input '" << input.name() << "' !");
            }

            //for (size_t i = 0; i < graph.output_size(); ++i)
            //    network.dst().push_back(graph.output(i).name());

            for (size_t i = 0; i < graph.node_size(); ++i)
            {
                const onnx::NodeProto& node = graph.node(i);
                LayerParam layer;
                SetSrcAndDst(node, renames, layer);

                //CPL_LOG_SS(Info, "Convert node[" << i << "]: " << NodeString(node));

                if (node.op_type() == "Abs" && !ConvertAbsNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Add" && !ConvertAddNode(node, network.layers(), original, onnxParam, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "And" && !ConvertAndNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ArgMax" && !ConvertArgMaxNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "AveragePool" && !ConvertAveragePoolNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "BatchNormalization" && !ConvertBatchNormalizationNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Cast" && !ConvertCastNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Ceil" && !ConvertCeilNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Clip" && !ConvertClipNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Concat" && !ConvertConcatNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Constant" && !ConvertConstantNode(node, layer, original, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ConstantOfShape" && !ConvertConstantOfShapeNode(node, network.layers(), layer, original, reordered))
                    return ErrorMessage(i, node);
                if ((node.op_type() == "Conv" || node.op_type() == "ConvTranspose") && !ConvertConvOrConvTransposeNode(node, trans, network.layers(), original, layer, reordered, &permuteMap))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Cos" && !ConvertCosNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "DequantizeLinear" && !ConvertDequantizeLinearNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Div" && !ConvertDivNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Erf" && !ConvertErfNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Equal" && !ConvertEqualNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Exp" && !ConvertExpNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Expand" && !ConvertExpandNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Flatten" && !ConvertFlattenNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Floor" && !ConvertFloorNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if ((node.op_type() == "Gather" || node.op_type() == "GatherElements") && !ConvertGatherNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Gemm" && !ConvertGemmNode(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "GlobalAveragePool" && !ConvertGlobalAveragePoolNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Greater" && !ConvertGreaterNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "GridSample" && !ConvertGridSampleNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "HardSigmoid" && !ConvertHardSigmoidNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Identity" && !ConvertIdentityNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "InstanceNormalization" && !ConvertInstanceNormalizationNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "LayerNormalization" && !ConvertLayerNormalizationNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "LeakyRelu" && !ConvertLeakyReluNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Less" && !ConvertLessNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Log" && !ConvertLogNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "LogSoftmax" && !ConvertLogSoftmaxNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "LSTM" && !ConvertLstmNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "MatMul" && !ConvertMatMulNode(node, trans, network.layers(), layer, &tensorFormatMap))
                    return ErrorMessage(i, node);
                if (node.op_type() == "MaxPool" && !ConvertMaxPoolNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Mod" && !ConvertModNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Mul" && !ConvertMulNode(node, network.layers(), original, onnxParam, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Neg" && !ConvertNegNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "NonMaxSuppression" && !ConvertNonMaxSuppressionNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "NonZero" && !ConvertNonZeroNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Not" && !ConvertNotNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Pad" && !ConvertPadNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Pow" && !ConvertPowNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "PRelu" && !ConvertPreluNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "QuantizeLinear" && !ConvertQuantizeLinearNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Range" && !ConvertRangeNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceL2" && !ConvertReduceL2Node(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceMax" && !ConvertReduceMaxNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceMean" && !ConvertReduceMeanNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ReduceSum" && !ConvertReduceSumNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Relu" && !ConvertReluNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Reshape" && !ConvertReshapeNode(node, trans, network.layers(), original, onnxParam, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Resize" && !ConvertResizeNode(node, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ScaledDotProductAttention" && !ConvertScaledDotProductAttentionNode(node, network.layers(), layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "ScatterND" && !ConvertScatterNdNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Shape" && !ConvertShapeNode(node, trans, network.layers(), onnxParam, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sigmoid" && !ConvertSigmoidNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sin" && !ConvertSinNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Slice" && !ConvertSliceNode(node, trans, network.layers(), layer, &permuteMap))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Softmax" && !ConvertSoftmaxNode(node, trans, network.layers(), original, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Split" && !ConvertSplitNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sqrt" && !ConvertSqrtNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Squeeze" && !ConvertSqueezeNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Sub" && !ConvertSubNode(node, network.layers(), original, layer, reordered))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Tanh" && !ConvertTanhNode(node, layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Tile" && !ConvertTileNode(node, trans, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "TopK" && !ConvertTopKNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Transpose" && !ConvertTransposeNode(node, trans, network.layers(), onnxParam, layer, &tensorFormatMap))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Unsqueeze" && !ConvertUnsqueezeNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);
                if (node.op_type() == "Where" && !ConvertWhereNode(node, network.layers(), layer))
                    return ErrorMessage(i, node);

#if defined(SYNET_ONNX_PARSE_STOP_ON_ERROR)
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(i, node);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(node, layer);
                    CPL_LOG_SS(Warning, "Not implemented node[" << i << "]: " << NodeString(node));
                }
#endif
                network.layers().push_back(layer);

                if (trans && !ManualInsertToNchwPermute(onnxParam, network.layers(), renames))
                    return false;
                if (trans && !ManualInsertToNhwcPermute(onnxParam, network.layers(), renames))
                    return false;
            }

            if (!RemoveUnusedConst(network.layers()))
                return false;

            return true;
        }

        String ValidName(const String & src, Renames& renames)
        {
            String dst = src;
            for (size_t i = 0; i < dst.size(); ++i)
            {
                if (dst[i] == ':' || dst[i] == ' ')
                    dst[i] = '_';
            }
            if (dst != src)
                renames[src] = dst;
            return dst;
        }

        bool ConvertInitializer(const onnx::TensorProto& tensor, Synet::NetworkParam& network, Bytes& weight, Renames& renames)
        {
            LayerParam layer;
            layer.name() = ValidName(tensor.name(), renames);
            layer.dst().push_back(layer.name());
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
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * sizeof(float);
                if (size)
                {
                    if (size == 1 && layer.weight()[0].dim().empty())
                    {
                        layer.weight()[0].dim().push_back(1);
                        layer.weight()[0].scalar() = true;
                    }
                    if (tensor.has_raw_data())
                        Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                    else if (tensor.float_data_size())
                    {
                        if (size != tensor.float_data_size())
                            SYNET_ERROR("Wrong tensor float_data_size " << tensor.float_data_size() << " != " << size << " !");
                        for (size_t i = 0; i < size; ++i)
                            PushBack<float>(weight, tensor.float_data(i));
                    }
                    else
                        SYNET_ERROR("Can't parse '" << layer.name() << "' FP32 tensor!");
                }
            }
            else if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
            {
                layer.type() = LayerTypeConst;
                layer.weight().resize(1);
                layer.weight()[0].type() = TensorType32i;
                uint64_t size = 1, offset = weight.size();
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= (size_t)tensor.dims(i);
                    layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
                }
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * sizeof(int32_t);
                if (size)
                {
                    if (size == 1 && layer.weight()[0].dim().empty())
                    {
                        layer.weight()[0].dim().push_back(1);
                        layer.weight()[0].scalar() = true;
                    }
                    if (tensor.has_raw_data())
                        Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                    else if (tensor.int32_data_size())
                    {
                        if (size != tensor.int32_data_size())
                            SYNET_ERROR("Wrong tensor int32_data_size " << tensor.int32_data_size() << " != " << size << " !");
                        for (size_t i = 0; i < size; ++i)
                            PushBack<int32_t>(weight, tensor.int32_data(i));
                    }
                    else
                        SYNET_ERROR("Can't parse '" << layer.name() << "' INT32 tensor!");
                }
            }
            else if(tensor.data_type() == onnx::TensorProto_DataType_INT64)
            {
                ptrdiff_t size = 1;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                    size *= (size_t)tensor.dims(i);
                if (size < 16)
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
                        if (layer.meta().alpha().shape().empty())
                        {
                            layer.meta().alpha().shape().push_back(1);
                            layer.meta().alpha().scalar() = true;
                        }
                        if (tensor.has_raw_data())
                        {
                            for (size_t i = 0; i < size; ++i)
                                layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
                        }
                        else if (tensor.int64_data_size())
                        {
                            for (size_t i = 0; i < size; ++i)
                                layer.meta().alpha().i64()[i] = tensor.int64_data(i);
                        }
                        else
                            SYNET_ERROR("Can't parse '" << layer.name() << "' INT64 tensor!");
                    }
                }
                else
                {
                    size_t offset = weight.size();
                    layer.type() = LayerTypeConst;
                    layer.weight().resize(1);
                    layer.weight()[0].type() = TensorType64i;
                    for (size_t i = 0; i < tensor.dims_size(); ++i)
                        layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
                    layer.weight()[0].offset() = offset;
                    layer.weight()[0].size() = size * sizeof(int64_t);
                    if (size)
                    {
                        if (tensor.has_raw_data())
                            Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                        else if (tensor.int64_data_size())
                        {
                            if (size != tensor.int64_data_size())
                                SYNET_ERROR("Wrong tensor int64_data_size " << tensor.int64_data_size() << " != " << size << " !");
                            for (size_t i = 0; i < size; ++i)
                                PushBack<int64_t>(weight, tensor.int64_data(i));
                        }
                        else
                            SYNET_ERROR("Can't parse '" << layer.name() << "' INT64 tensor!");
                    }
                }
            }
            else
                SYNET_ERROR(" Unknown tensor type " << tensor.data_type() << " !");
            network.layers().push_back(layer);
            return true;
        }

        bool ConvertInput(const onnx::ValueInfoProto & input, bool trans, Synet::NetworkParam& network, Renames& renames)
        {
            LayerParam layer;
            layer.type() = LayerTypeInput;
            layer.name() = ValidName(input.name(), renames);
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
            switch (input.type().tensor_type().elem_type())
            {
            case onnx::TensorProto_DataType_FLOAT: layer.input().shape()[0].type() = Synet::TensorType32f; break;
            case onnx::TensorProto_DataType_INT32: layer.input().shape()[0].type() = Synet::TensorType32i; break;
            default:
                SYNET_ERROR(" Unknown input tensor type " << input.type().tensor_type().elem_type() << " !");
            }
            network.layers().push_back(layer);
            return true;
        }

        void SetSrcAndDst(const onnx::NodeProto& node, Renames &renames, LayerParam& layer)
        {
            for (size_t j = 0; j < node.input_size(); ++j)
            {
                String input = node.input(j);
                if (input.empty())
                    continue;
                Renames::const_iterator rename = renames.find(input);
                if (rename != renames.end())
                {
                    input = rename->second;
                    rename = renames.find(input);
                    if (rename != renames.end())
                        input = rename->second;
                }
                layer.src().push_back(input);
            }
            for (size_t j = 0; j < node.output_size(); ++j)
                layer.dst().push_back(ValidName(node.output(j), renames));
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
                        permute.permute().order() = Shp(0, 3, 1, 2);
                        permute.permute().format() = TensorFormatNchw;
                        if (renames.find(dst) != renames.end())
                            SYNET_ERROR("Multiple manual NhwcToNchw permute at " << layer.name() << " !");
                        renames[dst] = permute.name();
                        //CPL_LOG_SS(Info, "Insert manual NhwcToNchw permute at " << layer.name() << " !");
                        layers.push_back(permute);
                    }
                }
            }
            return true;
        }

        bool ManualInsertToNhwcPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames)
        {
            LayerParam& layer = layers.back();
            for (size_t h = 0; h < onnxParam.toNhwcHints().size(); ++h)
            {
                if (layer.name() == onnxParam.toNhwcHints()[h])
                {
                    for (size_t d = 0; d < layer.dst().size(); ++d)
                    {
                        const String& dst = layer.dst()[d];
                        LayerParam permute;
                        permute.type() = LayerTypePermute;
                        permute.src().push_back(dst);
                        permute.name() = dst + "_permute_to_nhwc";
                        permute.dst().push_back(permute.name());
                        permute.permute().order() = Shp(0, 2, 3, 1);
                        permute.permute().format() = TensorFormatNhwc;
                        if (renames.find(dst) != renames.end())
                            SYNET_ERROR("Multiple manual NchwToNhwc permute at " << layer.name() << " !");
                        renames[dst] = permute.name();
                        //CPL_LOG_SS(Info, "Insert manual NchwToNhwc permute at " << layer.name() << " !");
                        layers.push_back(permute);
                    }
                }
            }
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool ConvertAbsNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeAbs;
            return true;
        }

        bool ConvertAddNode(const onnx::NodeProto& node, LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
                GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
            {
                layer.type() = Synet::LayerTypeQuantizedAdd;
                if (!MoveDequantizeLinearToLayer(layers, layer))
                    return false;
                return true;
            }
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeAdd;
            }
            else if(src1->type() == LayerTypeConst && src1->weight()[0].dim() == Shp(1))
            {
                const float* shift = GetWeight<float>(original, src1->weight()[0]);
                layer.type() = Synet::LayerTypePower;
                layer.power().power() = 1.0f;
                layer.power().scale() = 1.0f;
                layer.power().shift() = shift[0];
                layer.src().resize(1);
            }
            else
            {
                if (onnxParam.addToEltwise())
                {
                    layer.type() = Synet::LayerTypeEltwise;
                    layer.eltwise().operation() = EltwiseOperationTypeSum;
                }
                else
                    layer.type() = Synet::LayerTypeAdd;
                if (src0->type() == LayerTypeConst && src1->type() != LayerTypeConst)
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertAndNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            layer.type() = Synet::LayerTypeBinaryOperation;
            layer.binaryOperation().type() = BinaryOperationTypeAnd;
            return true;
        }

        bool ConvertArgMaxNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            layer.type() = Synet::LayerTypeArgMax;
            if (!ConvertAtrributeInt(node, "axis", layer.argMax().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.argMax().keepDims()))
                return false;
            return true;
        }

        bool ConvertAveragePoolNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeAverage;
            if (!ConvertAtrributeInts(node, "kernel_shape", layer.pooling().kernel()))
                return false;
            if (GetAtrribute(node, "pads"))
            {
                if (!ConvertAtrributeInts(node, "pads", layer.pooling().pad()))
                    return false;
            }
            if (!ConvertAtrributeInts(node, "strides", layer.pooling().stride()))
                return false;
            if (GetAtrribute(node, "ceil_mode") == NULL)
                layer.pooling().roundingType() = RoundingTypeFloor;
            else
            {
                int ceilMode;
                if (!ConvertAtrributeInt(node, "ceil_mode", ceilMode))
                    return false;
                layer.pooling().roundingType() = ceilMode ? RoundingTypeCeil : RoundingTypeFloor;
            }
            if (GetAtrribute(node, "count_include_pad"))
            {
                int64_t countIncludePad;
                if (!ConvertAtrributeInt(node, "count_include_pad", countIncludePad))
                    return false;
                layer.pooling().excludePad() = (countIncludePad == 0);
            }
            return true;
        }

        bool ConvertBatchNormalizationNode(const onnx::NodeProto & node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 5))
                return false;

            bool shared1, shared2;
            const LayerParam* src1 = GetWeightLayer(layers, layer.src()[1], &shared1);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                SYNET_ERROR("BatchNormalization src[1] must be Const type!");
            const float* gamma = GetWeight<float>(original, src1->weight()[0]);

            const LayerParam* src2 = GetWeightLayer(layers, layer.src()[2], &shared2);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                SYNET_ERROR("BatchNormalization src[2] must be Const type!");
            const float* beta = GetWeight<float>(original, src2->weight()[0]);

            const LayerParam* src3 = GetWeightLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                SYNET_ERROR("BatchNormalization src[3] must be Const type!");
            const float* mean = GetWeight<float>(original, src3->weight()[0]);

            const LayerParam* src4 = GetWeightLayer(layers, layer.src()[4]);
            if (src4 == NULL || src4->type() != LayerTypeConst)
                SYNET_ERROR("BatchNormalization src[4] must be Const type!");
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
            layer.weight()[0] = src1->weight()[0];
            if(shared1)
            {
                size_t size = TensorSize(layer.weight()[0].dim()), offset = reordered.size();
                original.resize(offset + size * 4);
                reordered.resize(offset + size * 4);
                layer.weight()[0].offset() = offset;
            }
            layer.weight()[1] = src2->weight()[0];
            if (shared2)
            {
                size_t size = TensorSize(layer.weight()[1].dim()), offset = reordered.size();
                original.resize(offset + size * 4);
                reordered.resize(offset + size * 4);
                layer.weight()[1].offset() = offset;
            }
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

        bool ConvertCastNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
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
                    SYNET_ERROR("Unsupported cast type!");
            }
            else
            {
                layer.type() = Synet::LayerTypeCast;
                if (to == onnx::TensorProto_DataType_FLOAT)
                    layer.cast().type() = TensorType32f;
                else if (to == onnx::TensorProto_DataType_INT32)
                    layer.cast().type() = TensorType32i;
                else if (to == onnx::TensorProto_DataType_INT64)
                    layer.cast().type() = TensorType64i;
                else if (to == onnx::TensorProto_DataType_UINT8)
                    layer.cast().type() = TensorType8u;
                else
                    SYNET_ERROR("Unsupported cast type!");
                if (src0->type() == LayerTypeConst && src0->weight().size() && src0->weight()[0].type() == layer.cast().type())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.cast().type() = TensorTypeUnknown;
                }
                if (src0->type() == LayerTypeQuantizeLinear && src0->quantize().type() == layer.cast().type())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.cast().type() = TensorTypeUnknown;
                }
            }
            return true;
        }

        bool ConvertCeilNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeCeil;
            }
            else
            {
                SYNET_ERROR("Unsupported src type!");
            }
            return true;
        }

        bool ConvertClipNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            if (layer.src().size() > 1)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeConst || src1->weight().size() != 1)
                    return false;
                const float* min = GetWeight<float>(original, src1->weight()[0]);
                layer.restrictRange().lower() = min[0];
                if (layer.src().size() > 2)
                {
                    const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                    if (src2 == NULL || src2->type() != LayerTypeConst || src2->weight().size() != 1)
                        return false;
                    const float* max = GetWeight<float>(original, src2->weight()[0]);
                    layer.restrictRange().upper() = max[0];
                }
                layer.src().resize(1);
            }
            else
            {
                if (!ConvertAtrributeFloat(node, "min", layer.restrictRange().lower(), true, -FLT_MAX))
                    return false;
                if (!ConvertAtrributeFloat(node, "max", layer.restrictRange().upper(), true, +FLT_MAX))
                    return false;
            }
            return true;
        }

        bool ConvertConcatNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            const LayerParam* src1 = layer.src().size() < 2 ? 0 : GetLayer(layers, layer.src()[1]);
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
                if (trans && CurrentTensorFormat(layers, layer.src(), true, true, true) == TensorFormatNhwc)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    if(layer.concat().axis() >= 0 && layer.concat().axis() < 4)
                        layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                }
            }
            return true;
        }

        void ConvertConstantTensor(const onnx::TensorProto& tensor, TensorType typeName, size_t typeSize, LayerParam& layer, Bytes& original, Bytes& reordered)
        {
            layer.type() = LayerTypeConst;
            layer.weight().resize(1);
            layer.weight()[0].type() = typeName;
            uint64_t size = typeSize, offset = original.size();
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                size *= tensor.dims(i);
                layer.weight()[0].dim().push_back(size_t(tensor.dims(i)));
            }
            if (layer.weight()[0].dim().empty())
                layer.weight()[0].dim().push_back(1);
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size;
            if (tensor.has_raw_data() && size)
            {
                Append(original, layer.weight()[0], tensor.raw_data().c_str());
                Append(reordered, layer.weight()[0], tensor.raw_data().c_str());
            }
        }

        bool ConvertConstantNode(const onnx::NodeProto& node, LayerParam& layer, Bytes& original, Bytes& reordered)
        {
            String name = "value_ints";
            const onnx::AttributeProto* value_ints = GetAtrribute(node, name);
            if (value_ints)
            {
                size_t size = value_ints->ints_size();
                if(size == 0)
                    SYNET_ERROR("Attribute " << name << " is empty!");
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeConst;
                layer.meta().alpha().type() = TensorType64i;
                layer.meta().alpha().shape() = Shp(size);
                layer.meta().alpha().i64().resize(size);
                for (size_t i = 0; i < size; ++i)
                    layer.meta().alpha().i64()[i] = value_ints->ints(i);
                return true;
            }
            name = "value";
            const onnx::AttributeProto * value = GetAtrribute(node, name);
            if (value == NULL)
                SYNET_ERROR("Can't find attribute " << name << " !");
            if (value->type() != onnx::AttributeProto_AttributeType_TENSOR)
                SYNET_ERROR("Attribute has wrong type " << value->type() << " !");
            const onnx::TensorProto& tensor = value->t();
            if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
            {
                ptrdiff_t size = 1;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                    size *= tensor.dims(i);
                if (size < 16)
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
                    if (layer.meta().alpha().shape().empty())
                        layer.meta().alpha().shape().push_back(1);
                    layer.meta().alpha().i64().resize(size);
                    if (tensor.has_raw_data())
                    {
                        for (size_t i = 0; i < size; ++i)
                            layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
                    }
                }
                else
                    ConvertConstantTensor(tensor, TensorType64i, sizeof(int64_t), layer, original, reordered);
            }
            else if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
                ConvertConstantTensor(tensor, TensorType32f, sizeof(float), layer, original, reordered);
            else if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                ConvertConstantTensor(tensor, TensorType32i, sizeof(int32_t), layer, original, reordered);
            else if (tensor.data_type() == onnx::TensorProto_DataType_DOUBLE)
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
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * sizeof(float);
                if (tensor.has_raw_data() && size)
                {
                    for (size_t i = 0; i < size; ++i)
                    {
                        float value = float(((double*)tensor.raw_data().c_str())[i]);
                        PushBack<float>(original, value);
                        PushBack<float>(reordered, value);
                    }
                }
            }
            else if (tensor.data_type() == onnx::TensorProto_DataType_BOOL)
                ConvertConstantTensor(tensor, TensorTypeBool, sizeof(bool), layer, original, reordered);
            else if (tensor.data_type() == onnx::TensorProto_DataType_UINT8)
                ConvertConstantTensor(tensor, TensorType8u, sizeof(uint8_t), layer, original, reordered);
            else if (tensor.data_type() == onnx::TensorProto_DataType_INT8)
                ConvertConstantTensor(tensor, TensorType8i, sizeof(int8_t), layer, original, reordered);
            else
                SYNET_ERROR("Unsupported format of Constant node!");
            return true;
        }

        bool ConvertConstantOfShapeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& original, Bytes& reordered)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)// || src0->type() != Synet::LayerTypeMeta)
                return false;
            Shape shape;
            if (IsMetaConst64i(*src0))
                shape = Shp(src0->meta().alpha().i64());
            const onnx::AttributeProto * attribute = GetAtrribute(node, "value");
            if (attribute && attribute->type() == onnx::AttributeProto_AttributeType_TENSOR)
            {
                const onnx::TensorProto& tensor = attribute->t();
                if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                {
                    int64_t value;
                    if (tensor.int64_data_size())
                        value = tensor.int64_data(0);
                    else if (tensor.has_raw_data())
                        value = ((int64_t*)tensor.raw_data().c_str())[0];
                    else
                        return false;
                    if (src0->type() != Synet::LayerTypeMeta)
                        return false;
                    if (src0->meta().type() == Synet::MetaTypeConst)
                    {
                        if (src0->meta().alpha().type() != Synet::TensorType64i || src0->meta().alpha().shape().size() != 1 || src0->meta().alpha().shape()[0] != 1)
                            return false;
                        layer.type() = Synet::LayerTypeMeta;
                        layer.meta().type() = Synet::MetaTypeConst;
                        layer.meta().alpha().type() = Synet::TensorType64i;
                        layer.meta().alpha().shape().push_back(src0->meta().alpha().i64()[0]);
                        layer.meta().alpha().i64().resize(src0->meta().alpha().i64()[0], value);
                        layer.src().resize(0); 
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeMeta;
                        layer.meta().type() = Synet::MetaTypeConstantOfShape;
                        layer.meta().alpha().type() = Synet::TensorType64i;
                        layer.meta().alpha().shape() = Shp(1);
                        layer.meta().alpha().i64().resize(1, value);
                    }
                }
                else if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
                {
                    float value;
                    if (tensor.float_data_size())
                        value = tensor.float_data(0);
                    else if (tensor.has_raw_data())
                        value = ((float*)tensor.raw_data().c_str())[0];
                    else
                        return false;
                    layer.type() = Synet::LayerTypeConstantOfShape;
                    layer.constantOfShape().value().type() = TensorType32f;
                    layer.constantOfShape().value().shape() = Shp(1);
                    layer.constantOfShape().value().f32().resize(1, value);
                }
                else if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                {
                    int32_t value;
                    if (tensor.int32_data_size())
                        value = tensor.int32_data(0);
                    else if (tensor.has_raw_data())
                        value = ((int32_t*)tensor.raw_data().c_str())[0];
                    else
                        return false;                    
                    if (shape.empty())
                    {
                        layer.type() = Synet::LayerTypeConstantOfShape;
                        layer.constantOfShape().value().type() = TensorType32i;
                        layer.constantOfShape().value().shape() = Shp(1);
                        layer.constantOfShape().value().i32().resize(1, value);
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeConst;
                        layer.weight().resize(1);
                        layer.weight()[0].type() = Synet::TensorType32i;
                        layer.weight()[0].dim() = shape;
                        layer.weight()[0].scalar() = true;
                        layer.weight()[0].offset() = original.size();
                        layer.weight()[0].size() = sizeof(int32_t);
                        PushBack<int32_t>(original, value);
                        PushBack<int32_t>(reordered, value);
                        layer.src().clear();
                    }
                }
                else
                    return false;
            }
            else
            {
                CPL_LOG_SS(Error, "Unsupported type of attribute 'value'");
                return false;
            }
            return true;
        }

        bool ConvertConvOrConvTransposeNode(const onnx::NodeProto & node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, PermuteMap* permuteMap)
        {
            if (node.op_type() == "Conv")
                layer.type() = Synet::LayerTypeConvolution;
            else if (node.op_type() == "ConvTranspose")
                layer.type() = Synet::LayerTypeDeconvolution;
            else
                return false;
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
            if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear && 
                GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
            {
                const LayerParam* dequantized = GetLayer(layers, layer.src()[1]);
                if (dequantized->weight().empty())
                    return false;
                const Shape& shape = dequantized->weight()[0].dim();
                layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
                layer.convolution().biasTerm() = layer.src().size() > 2;
                if (layer.type() == Synet::LayerTypeConvolution)
                {
                    layer.type() = Synet::LayerTypeQuantizedConvolution;
                    if (!MoveDequantizeLinearToLayer(layers, layer))
                        return false;
                }
                else
                    return false;
                if (trans && !PermutedToNchw(layers, layer.src(), true, false, false, permuteMap))
                    return ReorderWeight(srcBin, Shape(), layer, dstBin);
                return true;
            }
            const LayerParam* weight = GetWeightLayer(layers, layer.src()[1]);
            if (weight == NULL || weight->type() != LayerTypeConst)
                return false;
            const Shape& shape = weight->weight()[0].dim();
            layer.weight().resize(layer.src().size() - 1);
            layer.weight()[0] = weight->weight()[0];
            layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.convolution().biasTerm())
            {
                const LayerParam* bias = GetWeightLayer(layers, layer.src()[2]);
                if (bias == NULL || bias->type() != LayerTypeConst)
                    return false;
                layer.weight()[1] = bias->weight()[0];
            }
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, false, permuteMap))
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }

        bool ConvertCosNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeCos;
            return true;
        }

        bool ConvertDequantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetWeightLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetWeightLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            layer.type() = Synet::LayerTypeDequantizeLinear;
            if (!ConvertAtrributeInt(node, "axis", layer.quantize().axis(), true, 0))
                return false;
            if (src0->type() == LayerTypeConst)
            {
                layer.weight().push_back(src0->weight()[0]);
                layer.weight().push_back(src1->weight()[0]);
                layer.weight().push_back(src2->weight()[0]);
                layer.src().resize(0);
            }
            else
            {
                layer.weight().push_back(src1->weight()[0]);
                layer.weight().push_back(src2->weight()[0]);
                layer.src().resize(1);
            }
            if (TensorSize(layer.weight().back().dim()) == 1 || layer.weight().back().scalar())
            {
                switch (layer.weight().back().type())
                {
                case TensorType8u:
                    layer.quantize().zero() = GetWeight<uint8_t>(original, layer.weight().back())[0];
                    break;
                case TensorType32i:
                    layer.quantize().zero() = GetWeight<int32_t>(original, layer.weight().back())[0];
                    break;
                default:
                    return false;
                }
                layer.quantize().type() = layer.weight().back().type();
                layer.weight().resize(layer.weight().size() - 1);
                if (TensorSize(layer.weight().back().dim()) == 1 || layer.weight().back().scalar())
                {
                    switch (layer.weight().back().type())
                    {
                    case TensorType32f:
                        layer.quantize().scale() = GetWeight<float>(original, layer.weight().back())[0];
                        break;
                    default:
                        return false;
                    }
                    layer.weight().resize(layer.weight().size() - 1);
                }
            }
            return true;
        }

        bool ConvertDivNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
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
                layer.power().scale() = float(1.0 / double(pScale[0]));
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
            {
                const float* pSrc0 = GetWeight<float>(original, src0->weight()[0]);
                if (pSrc0[0] != 1.0f)
                    return false;
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = UnaryOperationTypeRcp;
                layer.src().erase(layer.src().begin());
                return true;
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
            {
                const float* pSrc = GetWeight<float>(original, src1->weight()[0]);
                size_t size = TensorSize(src1->weight()[0].dim());
                bool uniform = true;
                for (size_t i = 1; i < size && uniform; ++i)
                    uniform = (pSrc[i] == pSrc[0]);
                if (uniform)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().scale() = 1.0f / pSrc[0];
                }
                else
                {
                    layer.weight() = src1->weight();
                    const Shape& dim = src1->weight()[0].dim();
                    if ((dim.size() == 4 && dim[1] != 1) || (dim.size() == 3 && dim[0] != 1) || dim.size() == 1)
                    {
                        layer.type() = Synet::LayerTypeScale;
                        layer.scale().biasTerm() = false;
                        if (dim.size() == 1)
                            layer.scale().axis() = -1;
                        if (!CompactShape(layer.weight()[0].dim()))
                            return false;                    
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeMul;
                    }
                    float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                    for (size_t i = 0; i < size; ++i)
                        pDst[i] = 1.0f / pSrc[i];
                }
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

        bool ConvertErfNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeErf;
            return true;
        }

        bool ConvertEqualNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeEqual;
            }
            else
            {
                layer.type() = Synet::LayerTypeCompare;
                layer.compare().compareType() = CompareTypeEqual;
            }
            return true;
        }

        bool ConvertExpNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeExp;
            return true;
        }

        bool ConvertExpandNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeMeta)
            {
                const MetaParam & meta = src1->meta();
                if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i && AllEqualTo(meta.alpha().i64(), int64_t(1)))
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.src().resize(1);
                }
                else if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i &&
                    src0->type() == LayerTypeConst && Shp(meta.alpha().i64()) == src0->weight()[0].dim())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.src().resize(1);
                }
                else
                {
                    layer.type() = Synet::LayerTypeTile;
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertFlattenNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeFlatten;
            if (!ConvertAtrributeInt(node, "axis", layer.flatten().axis()))
                return false;
            return true;
        }

        bool ConvertFloorNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeFloor;
            }
            else
            {
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = Synet::UnaryOperationTypeFloor;
            }
            return true;
        }

        bool ConvertGatherNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeGather;
            }
            else
            {
                layer.type() = LayerTypeGather;
                if (!ConvertAtrributeInt(node, "axis", layer.gather().axis()))
                    return false;
                layer.gather().version() = node.op_type() == "GatherElements" ? 1 : 0;
            }
            return true;
        }

        bool ConvertGemmNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            layer.type() = Synet::LayerTypeInnerProduct;
            int transB;
            if (!ConvertAtrributeInt(node, "transB", transB))
                return false;
            layer.innerProduct().transposeB() = !transB;
            if (layer.src().size() < 2 || layer.src().size() > 3)
                return false;
            if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
                GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
            {
                layer.type() = Synet::LayerTypeQuantizedInnerProduct;
                const LayerParam* dequantized = GetLayer(layers, layer.src()[1]);
                if (dequantized->weight().empty())
                    return false;
                const Shape& shape = dequantized->weight()[0].dim();
                if (!CheckDims(shape, 2, "quantized inner product weight"))
                    return false;
                layer.innerProduct().outputNum() = (uint32_t)(transB ? shape[0] : shape[1]);
                layer.convolution().biasTerm() = layer.src().size() > 2;
                if (!MoveDequantizeLinearToLayer(layers, layer))
                    return false;
                return true;
            }
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

        bool ConvertGlobalAveragePoolNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear)
            {
                layer.type() = Synet::LayerTypeQuantizedPooling;
                layer.pooling().method() = PoolingMethodTypeAverage;
                layer.pooling().globalPooling() = true;
                if (!MoveDequantizeLinearToLayer(layers, layer))
                    return false;
            }
            else
            {
                layer.type() = Synet::LayerTypePooling;
                layer.pooling().method() = PoolingMethodTypeAverage;
                layer.pooling().globalPooling() = true;
            }
            return true;
        }

        bool ConvertGreaterNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeCompare;
            layer.compare().compareType() = CompareTypeGreaterThan;
            return true;
        }

        bool ConvertGridSampleNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;

            layer.type() = LayerTypeGridSample;
            if (!ConvertAtrributeInt(node, "align_corners", layer.gridSample().alignCorners()))
                return false;

            String interpMode;
            if (!ConvertAtrributeString(node, "mode", interpMode))
                return false;
            if (interpMode == "bilinear")
                layer.gridSample().interpMode() = GridSampleInterpModeBilinear;
            else if (interpMode == "nearest")
                layer.gridSample().interpMode() = GridSampleInterpModeNearest;
            else if (interpMode == "bicubic")
                layer.gridSample().interpMode() = GridSampleInterpModeBicubic;
            else
                return false;

            String paddingMode;
            if (!ConvertAtrributeString(node, "padding_mode", paddingMode))
                return false;
            if (paddingMode == "zeros")
                layer.gridSample().paddingMode() = GridSamplePaddingModeZeros;
            else if (paddingMode == "border")
                layer.gridSample().paddingMode() = GridSamplePaddingModeBorder;
            else if (paddingMode == "reflection")
                layer.gridSample().paddingMode() = GridSamplePaddingModeReflection;
            else
                return false;

            return true;
        }

        bool ConvertHardSigmoidNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeHardSigmoid;
            if (!ConvertAtrributeFloat(node, "alpha", layer.hardSigmoid().scale()))
                return false;
            return true;
        }

        bool ConvertIdentityNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeStub;
            }
            else
            {
                layer.type() = LayerTypeStub;
            }
            return true;
        }

        bool ConvertInstanceNormalizationNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            layer.type() = Synet::LayerTypeNormalize;
            layer.normalize().version() = 3;
            if (!ConvertAtrributeFloat(node, "epsilon", layer.normalize().eps()))
                return false;
            layer.weight().resize(2);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            layer.weight()[0] = src1->weight()[0];
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = src2->weight()[0];
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                layer.normalize().axis() = -1;
            }
            else
                layer.normalize().axis() = 1;
            return true;
        }

        bool ConvertLayerNormalizationNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            layer.type() = Synet::LayerTypeNormalize;
            layer.normalize().version() = 3;
            if (!ConvertAtrributeFloat(node, "epsilon", layer.normalize().eps()))
                return false;
            layer.weight().resize(2);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            layer.weight()[0] = src1->weight()[0];
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = src2->weight()[0];
            layer.src().resize(1);
            if (GetAtrribute(node, "axis"))
            {
                if (!ConvertAtrributeInt(node, "axis", layer.normalize().axis()))
                    return false;
                if(layer.normalize().axis() == -1)
                    layer.normalize().version() = 2;
            }
            else
            {
                if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
                {
                    layer.normalize().axis() = -1;
                }
                else
                    layer.normalize().axis() = 1;
            }
            return true;
        }

        bool ConvertLeakyReluNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            if (!ConvertAtrributeFloat(node, "alpha", layer.relu().negativeSlope()))
                return false;
            return true;
        }

        bool ConvertLessNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeCompare;
            layer.compare().compareType() = CompareTypeLessThan;
            return true;
        }

        bool ConvertLogNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeLog;
            return true;
        }

        bool ConvertLogSoftmaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            if (!ConvertAtrributeInt(node, "axis", layer.softmax().axis()))
                return false;
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                CPL_LOG_SS(Error, "This layer can work only in NCHW format!");
                return false;
            }
            layer.softmax().log() = true;
            return true;
        }

        bool ConvertLstmNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeLstm;
            if (!ConvertAtrributeInt(node, "hidden_size", layer.lstm().hiddenSize()))
                return false;
            String direction;
            if (!ConvertAtrributeString(node, "direction", direction))
                return false;
            if (direction == "forward")
                layer.lstm().direction() = LstmDirectionTypeForward;
            else if (direction == "reverse")
                layer.lstm().direction() = LstmDirectionTypeReverse;
            else if (direction == "bidirectional")
                layer.lstm().direction() = LstmDirectionTypeBidirectional;
            else
                return false;
            if (!CheckSourceNumber(layer, 6))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                return false;
            layer.weight().resize(3);
            layer.weight()[0] = src1->weight()[0];
            layer.weight()[1] = src2->weight()[0];           
            layer.weight()[2] = src3->weight()[0];
            layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 4);
            layer.dst().resize(1);
            return true;
        }

        bool ConvertMatMulNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, TensorFormatMap *tensorFormatMap)
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
                if (src10 == NULL) 
                    return false;
                if (src10->type() == LayerTypeConst)
                {
                    transB = true;
                    layer.weight() = src10->weight();
                    layers.erase(layers.begin() + (src1 - layers.data()));
                }
            }
            Shape weight = layer.weight()[0].dim();
            layer.innerProduct().transposeB() = !transB;
            if (weight.empty())
            {
                layer.weight().clear();
                layer.innerProduct().outputNum() = 0;
                layer.innerProduct().axis() = -1;
            }
            else
            {
                //if (!CheckSignificantDims(weight, 2, "MatMul weight"))
                //    return false;
                if(weight.size() > 2)
                    layer.innerProduct().axis() = weight.size() - 1;
                layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[weight.size() - 2] : weight[weight.size() - 1]);
                layer.src().resize(1);
                if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, tensorFormatMap) == TensorFormatNhwc)
                    SYNET_ERROR("Can 't convert MatMul node for NHWC format!");
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

        bool ConvertModNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeMod;
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeMod;
            }
            return true;
        }

        bool ConvertMulNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeConst)
            {
                std::swap(src0, src1);
                std::swap(layer.src()[0], layer.src()[1]);
            }
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = pScale[0];
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 3 && src1->weight()[0].dim()[0] != 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.weight() = src1->weight();
                //if (!CompactShape(layer.weight()[0].dim()))
                //    return false;
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 4 && src1->weight()[0].dim()[1] != 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.weight() = src1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeMul;
            }
            else
            {
                if (onnxParam.mulToEltwise())
                {
                    layer.type() = Synet::LayerTypeEltwise;
                    layer.eltwise().operation() = EltwiseOperationTypeProduct;
                }
                else
                    layer.type() = Synet::LayerTypeMul;
                if (src0->type() == LayerTypeConst && src1->type() != LayerTypeConst)
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertNegNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = Synet::UnaryOperationTypeNeg;
            return true;
        }

        bool ConvertNonMaxSuppressionNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& bin, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 5))
                return false;

            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeMeta || src2->meta().type() != MetaTypeConst)
                return false;
            layer.nonMaxSuppression().maxOutputBoxesPerClass() = src2->meta().alpha().i64()[0];

            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                return false;
            layer.nonMaxSuppression().iouThreshold() = bin[src3->weight()[0].offset() / sizeof(float)];

            const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
            if (src4 == NULL || src4->type() != LayerTypeConst)
                return false;
            layer.nonMaxSuppression().scoreThreshold() = bin[src4->weight()[0].offset() / sizeof(float)];

            layer.type() = Synet::LayerTypeNonMaxSuppression;
            layer.src().resize(2);

            return true;
        }

        bool ConvertNonZeroNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            layer.type() = Synet::LayerTypeNonZero;
            return true;
        }

        bool ConvertNotNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = Synet::UnaryOperationTypeNot;
            return true;
        }

        bool ConvertPadNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 3))
                return false;

            layer.type() = Synet::LayerTypePad;
            String mode;
            if (!ConvertAtrributeString(node, "mode", mode, true, "constant"))
                return false;
            if (mode == "constant")
                layer.pad().mode() = PadModeConstant;
            else if (mode == "reflect")
                layer.pad().mode() = PadModeReflect;
            else if (mode == "edge")
                layer.pad().mode() = PadModeEdge;
            else if (mode == "wrap")
                layer.pad().mode() = PadModeWrap;
            else
                SYNET_ERROR("Unknown type of pad mode: " << mode << " !");
            
            if (layer.src().size() > 1)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeMeta)
                    return false;
                if (layer.src().size() > 2)
                {
                    const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                    if (src2 == NULL || src2->type() != LayerTypeConst || src2->weight()[0].type() != TensorType32f)
                        return false;
                    if(GetWeight<float>(original, src2->weight()[0])[0] != 0)
                        SYNET_ERROR("Synet support only pad value == 0!");
                    layer.src().resize(2);
                }
            }  
            else
            {
                if (!ConvertAtrributeInts(node, "pads", layer.pad().pads()))
                    return false;
            }

            return true;
        }

        bool ConvertPowNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
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
                const float* pPower = GetWeight<float>(original, src1->weight()[0]);
                layer.power().power() = pPower[0];
                layer.src().resize(1);
            }
            else
            {
                std::cout << "PowerNode error: src1 { type: " << Cpl::ToStr(src1->type()) << " size: " << TensorSize(src1->weight()[0].dim()) << " }" << std::endl;
                return false;
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

        bool ConvertQuantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            layer.type() = Synet::LayerTypeQuantizeLinear;
            if (!ConvertAtrributeInt(node, "axis", layer.quantize().axis(), true, 0))
                return false;
            if (src1->type() == LayerTypeConst && src2->type() == LayerTypeConst)
            {
                if (TensorSize(src1->weight()[0].dim()) == 1 && TensorSize(src2->weight()[0].dim()) == 1)
                {
                    layer.quantize().scale() = GetWeight<float>(original, src1->weight()[0])[0];
                    layer.quantize().type() = src2->weight()[0].type();
                    switch (layer.quantize().type())
                    {
                    case TensorType8u:
                        layer.quantize().zero() = GetWeight<uint8_t>(original, src2->weight()[0])[0];
                        break;
                    default:
                        SYNET_ERROR("QuantizeLinear: unsupported src[2] type!");
                    }
                }
                else
                {
                    layer.weight().push_back(src1->weight()[0]);
                    layer.weight().push_back(src2->weight()[0]);
                }
                layer.src().resize(1);
            }
            else
                SYNET_ERROR("QuantizeLinear: src[1] or src[2] is not const!");
            return true;
        }

        bool ConvertRangeNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() != LayerTypeMeta && src0->type() != LayerTypeConst)
                return false;
            if (src1->type() != LayerTypeMeta && src1->type() != LayerTypeConst)
                return false;
            if (src2->type() != LayerTypeMeta && src2->type() != LayerTypeConst)
                return false;
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeRange;
            return true;
        }

        bool ConvertReduceL2Node(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeL2;
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

        bool ConvertReduceMeanNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            Ints axes;
            if (!ConvertAtrributeInts(node, "axes", axes))
                return false;
            if (axes == Ints({ 2, 3 }))
            {
                layer.type() = Synet::LayerTypePooling;
                layer.pooling().method() = PoolingMethodTypeAverage;
                layer.pooling().globalPooling() = true;
            }
            else
            {
                layer.type() = Synet::LayerTypeReduction;
                layer.reduction().type() = ReductionTypeMean;
                for (size_t i = 0; i < axes.size(); ++i)
                    layer.reduction().axis().push_back(axes[i]);
                ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims(), true, true);
                if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true) == TensorFormatNhwc)
                {
                    Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                    for (size_t i = 0; i < axis.size(); ++i)
                        layer.reduction().axis()[i] = nchw[axis[i]];
                }
            }
            return true;
        }

        bool ConvertReduceSumNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeSum;
            if (layer.src().size() == 2)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeMeta || src1->meta().type() != Synet::MetaTypeConst)
                    return false;
                const TensorParam& alpha = src1->meta().alpha();
                if (alpha.type() != TensorType64i)
                    return false;
                layer.reduction().axis().resize(alpha.i64().size());
                for (size_t i = 0; i < alpha.i64().size(); ++i)
                    layer.reduction().axis()[i] = (int)alpha.i64()[i];
                layer.src().resize(1);
            }
            else
            {
                if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                    return false;
            }
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true) == TensorFormatNhwc)
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

        bool ConvertReshapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL || src1->type() != LayerTypeMeta)
                return false;
            if (src1->meta().type() == MetaTypeStub)
                src1 = GetLayer(layers, src1->src()[0]);
            if (src0->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReshape;
            }            
            else if (src1->meta().type() == MetaTypeConst)
            {
                const TensorParam & alpha = src1->meta().alpha();
                if (alpha.shape().size() != 1)
                    return false;
                Shape& shape = layer.reshape().shape();
                layer.type() = LayerTypeReshape;
                shape = Shp(alpha.i64().data(), alpha.shape()[0]);
                layer.src().resize(1);
                if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true) == TensorFormatNhwc)
                {
                    if (shape.size() == 5)
                    {
                        shape = Shp( shape[0], shape[3], shape[4], shape[1], shape[2]);
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
            else
            {
                layer.type() = LayerTypeReshape;
            }
            if (onnxParam.setReshapeAxis1() && layer.type() == LayerTypeReshape)
                layer.reshape().axis() = 1;
            return true;
        }

        bool ConvertResizeNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (layer.src().size() == 4)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                    return false;
                const LayerParam * src2 = GetLayer(layers, layer.src()[2]);
                if (src2->type() != Synet::LayerTypeConst || src2->weight()[0].dim()[0] != 0)
                    return false;
                layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 3);
                const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
                if (src1b->type() == Synet::LayerTypeConst)
                {
                    layer.weight() = src1b->weight();
                    layer.src().resize(1);
                }
            }           
            else if (layer.src().size() == 3)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                    return false;
                layer.src().erase(layer.src().begin() + 1);
                const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
                if (src1b->type() == Synet::LayerTypeConst)
                {
                    layer.weight() = src1b->weight();
                    layer.src().resize(1);
                }
            }
            if (layer.src().size() == 2)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() == Synet::LayerTypeMeta && src1->meta().type() == Synet::MetaTypeConst)
                {
                    const TensorParam & alpha = src1->meta().alpha();
                    if (alpha.shape().size() == 1 && alpha.shape()[0] == 4)
                    {
                        layer.interp().height() = (int32_t)alpha.i64()[2];
                        layer.interp().width() = (int32_t)alpha.i64()[3];
                        layer.src().resize(1);
                    }
                    else
                        return false;
                }
                else if (src1->type() == Synet::LayerTypeConst)
                {
                    layer.weight() = src1->weight();
                    layer.src().resize(1);
                }
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
                else if (coordTransf == "half_pixel")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypeHalfPixel;
                else if (coordTransf == "align_corners")
                    layer.interp().coordinateTransformType() = CoordinateTransformTypeCaffe;
                else
                    return false;
            }

            layer.type() = Synet::LayerTypeInterp;
            return true;
        }

        bool ConvertScaledDotProductAttentionNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 4))
                return false;
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst || src3->weight()[0].type() != TensorType32f || src3->weight()[0].dim() != Shp(1))
                return false;
            layer.type() = Synet::LayerTypeScaledDotProductAttention;
            layer.src().resize(3);
            return true;
        }

        bool ConvertScatterNdNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes & original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            layer.type() = Synet::LayerTypeScatterNd;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst)
            {
                const TensorParam & alpha = src1->meta().alpha();
                size_t size = TensorSize(alpha.shape()), offset = reordered.size();
                layer.type() = Synet::LayerTypeScatterNd;
                layer.weight().resize(1);
                layer.weight()[0].dim() = alpha.shape();
                layer.weight()[0].type() = TensorType32i;
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * 4;
                layer.src().erase(layer.src().begin() + 1);
                original.resize(offset + size * 4);
                reordered.resize(offset + size * 4);
                if (alpha.type() == TensorType64i)
                {
                    const int64_t* src = alpha.i64().data();
                    int32_t * dst = GetWeight<int32_t>(reordered, layer.weight()[0]);
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = (int32_t)src[i];
                }
                else
                {
                    std::cout << "src[1] type must be meta const int64!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool ConvertShapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const OnnxParam& onnxParam, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeShape;
            layer.meta().version() = 1;
            if (trans)
            {
                for (size_t i = 0; i < onnxParam.shapeV2s().size(); ++i)
                    if (layer.name() == onnxParam.shapeV2s()[i])
                        layer.meta().version() = 2;
            }
            return true;
        }

        bool ConvertSigmoidNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        bool ConvertSinNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeSin;
            return true;
        }

        bool ConvertSliceNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, PermuteMap* permuteMap)
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
            else if (layer.src().size() == 3)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                if (src0 == NULL || src1 == NULL || src2 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = MetaTypeSlice;
                }
                else
                    return false;
            }
            else if(layer.src().size() >= 4 && layer.src().size() <= 5)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
                if (src0 == NULL || src1 == NULL || src2 == NULL || src3 == NULL)
                    return false;
                const LayerParam* src4 = NULL;
                if (layer.src().size() > 4)
                {
                    src4 = GetLayer(layers, layer.src()[4]);
                    if (src4 == NULL)
                        return false;
                }
                if (src0->type() == LayerTypeMeta)
                {
                    if (!CheckSourceNumber(layer, 4, 5))
                        return false;
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = MetaTypeSlice;
                }
                else
                {
                    layer.type() = Synet::LayerTypeStridedSlice;
                    if (layer.src().size() == 4)
                    {
                        if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta)
                            return false;
                        if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                            src3->meta().type() == Synet::MetaTypeConst)
                        {
                            if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 ||
                                src3->meta().alpha().i64().size() != 1)
                                return false;
                            layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                            layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                            layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                            //layer.stridedSlice().strideDims().push_back((size_t)src4->meta().alpha().i64()[0]);
                            if (trans && !PermutedToNchw(layers, layer.src(), false, true, true, permuteMap))
                            {
                                Shape nchw = Shape({ 0, 3, 1, 2 });
                                layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                            }
                            layer.src().resize(1);
                        }
                    }
                    else if (layer.src().size() == 5)
                    {
                        if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta || src4->type() != LayerTypeMeta)
                            return false;
                        if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                            src3->meta().type() == Synet::MetaTypeConst && src4->meta().type() == Synet::MetaTypeConst)
                        {
                            if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 || 
                                src3->meta().alpha().i64().size() != 1 || src4->meta().alpha().i64().size() != 1)
                                return false;
                            layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                            layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                            layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                            layer.stridedSlice().strideDims().push_back(src4->meta().alpha().i64()[0]);
                            if (trans && !PermutedToNchw(layers, layer.src(), false, true, true, permuteMap))
                            {
                                Shape nchw = Shape({ 0, 3, 1, 2 });
                                layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                            }
                            layer.src().resize(1);
                        }
                    }
                }
            }
            return true;
        }

        bool ConvertSoftmaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            if (!ConvertAtrributeInt(node, "axis", layer.softmax().axis()))
                return false;
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, true))
            {
                CPL_LOG_SS(Error, "This layer can work only in NCHW format!");
                return false;
            }
            return true;
        }

        bool ConvertSplitNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
                if (!ConvertAtrributeInts(node, "split", layer.unpack().parts()))
                    return false;
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() == LayerTypeMeta &&  src1->meta().type() == Synet::MetaTypeConst)
                {
                    const TensorParam & alpha = src1->meta().alpha();
                    assert(alpha.shape().size() == 1);
                    for (size_t i = 0; i < alpha.shape()[0]; ++i)
                        layer.unpack().parts().push_back((int32_t)alpha.i64()[i]);
                    layer.src().resize(1);
                }
                else
                    assert(0);
            }
            if (!ConvertAtrributeInt(node, "axis", layer.unpack().axis()))
                return false;
            layer.type() = Synet::LayerTypeUnpack;
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, NULL) == TensorFormatNhwc)
            {
                Shape nchw = Shape({ 0, 3, 1, 2 });
                layer.unpack().axis() = nchw[layer.unpack().axis()];
            }
            return true;
        }

        bool ConvertSqrtNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeSqrt;
            return true;
        }

        bool ConvertSqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
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
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src0 == NULL || src1 == NULL)
                    return false;
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const TensorParam & alpha = src1->meta().alpha();
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = LayerTypeMeta;
                    layer.meta().type() = MetaTypeSqueeze;
                    layer.meta().alpha() = alpha;
                }
                else
                {
                    layer.type() = Synet::LayerTypeSqueeze;
                    if (alpha.type() == TensorType64i)
                    {
                        layer.squeeze().axes().resize(alpha.i64().size());
                        for (size_t i = 0; i < alpha.i64().size(); ++i)
                            layer.squeeze().axes()[i] = (int)alpha.i64()[i];
                    }
                    else
                        return false;
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertSubNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSub;
            }
            else if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
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

        bool ConvertTanhNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeTanh;
            return true;
        }

        bool ConvertTileNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            layer.type() = Synet::LayerTypeTile;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst && src1->meta().alpha().type() == TensorType64i)
            {
                Longs shape = src1->meta().alpha().i64();
                if (trans && !PermutedToNchw(layers, false, false, false))
                {
                    return false;
                }                
                for (size_t i = 0, already = 0; i < shape.size(); ++i)
                {
                    if (shape[i] != 1)
                    {
                        if (already)
                            return false;
                        layer.tile().axis() = i;
                        layer.tile().tiles() = (uint32_t)shape[i];
                        already = 1;
                    }
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertTopKNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;

            layer.type() = Synet::LayerTypeTopK;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst && src1->meta().alpha().type() == TensorType64i)
            {
                layer.topK().k() = src1->meta().alpha().i64()[0];
                layer.src().resize(1);
            }
            if (!ConvertAtrributeInt(node, "axis", layer.topK().axis()))
                return false;
            int64_t largest;
            if (!ConvertAtrributeInt(node, "largest", largest))
                return false;
            layer.topK().mode() = largest ? TopKModeMax : TopKModeMin;
            int64_t sorted;
            if (!ConvertAtrributeInt(node, "sorted", sorted))
                return false;
            layer.topK().sort() = sorted ? TopKSortValue : TopKSortIndex;
            layer.topK().indexElementType() = TensorType64i;

            return true;
        }

        bool ConvertTransposeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const OnnxParam& onnxParam, LayerParam& layer, TensorFormatMap* tensorFormatMap)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            Shape order;
            if (!ConvertAtrributeInts(node, "perm", order))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypePermute;
                layer.meta().alpha().shape() = Shp(order.size());
                layer.meta().alpha().type() = TensorType64i;
                for (size_t i = 0; i < order.size(); ++i)
                    layer.meta().alpha().i64().push_back(order[i]);
            }
            else
            {
                layer.type() = Synet::LayerTypePermute;
                if (trans)
                {
                    bool permutedToNchw = CurrentTensorFormat(layers, layer.src(), true, false, onnxParam.globalPoolingPermuteToNchw(), tensorFormatMap) != TensorFormatNhwc;
                    if (!permutedToNchw)
                    {
                        if (order == Shape({ 0, 2, 1, 3, 4 }))
                            order = Shape({ 0, 1, 2, 4, 3 });
                        if (order == Shp(0, 1, 3, 2))
                            order = Shp(0, 2, 1, 3);
                        else if (order == Shape({ 0, 2, 3, 1 }))
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
                    else 
                    {
                        if (order == Shape({ 0, 3, 1, 2 }) && onnxParam.transpose0312PermuteToNhwc())
                        {
                            order = Shape({ 0, 1, 2, 3 });
                            layer.permute().format() = TensorFormatNhwc;
                        }
                    }
                }
                layer.permute().order() = order;
            }
            return true;
        }

        bool ConvertUnsqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
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
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src0 == NULL || src1 == NULL)
                    return false;
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const TensorParam & alpha = src1->meta().alpha();
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeExpandDims;
                    layer.meta().alpha() = alpha;
                }
                else
                {
                    layer.type() = Synet::LayerTypeExpandDims;
                    if (alpha.type() == TensorType64i)
                    {
                        layer.expandDims().axes().resize(alpha.i64().size());
                        for (size_t i = 0; i < alpha.i64().size(); ++i)
                            layer.expandDims().axes()[i] = (int)alpha.i64()[i];
                    }
                    else
                        return false;
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertWhereNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta && src2->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypeSelect;
            }
            else
                layer.type() = Synet::LayerTypeWhere;
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool MoveDequantizeLinearToLayer(LayerParams& layers, LayerParam& layer)
        {
            for (int s = 0; s < (int)layer.src().size(); ++s)
            {
                LayerParam* dequantize = GetLayer(layers, layer.src()[s]);
                if (dequantize->type() != LayerTypeDequantizeLinear)
                    SYNET_ERROR("MoveDequantizeLinearToLayer can move only DequantizeLinearLayer layers!");
                layer.qSrc().push_back(dequantize->quantize());
                layer.qSrc().back().weights() = dequantize->weight().size();
                for (size_t w = 0; w < dequantize->weight().size(); ++w)
                    layer.weight().push_back(dequantize->weight()[w]);
                if (!dequantize->src().empty())
                {
                    layer.src()[s] = dequantize->src()[0];
                    dequantize->src().clear();
                }
                else
                {
                    layer.src().erase(layer.src().begin() + s, layer.src().begin() + s + 1);
                    --s;
                }
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
            case onnx::TensorProto_DataType_INT32: ss << "i32"; break;
            case onnx::TensorProto_DataType_INT64: ss << "i64"; break;
            case onnx::TensorProto_DataType_DOUBLE: ss << "f64"; break;
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
            case onnx::TensorProto_DataType_INT32:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((int32_t*)tensor.raw_data().c_str())[i];
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
            case onnx::TensorProto_DataType_DOUBLE:
            {
                if (tensor.double_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.double_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((double*)tensor.raw_data().c_str())[i];
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
            case onnx::AttributeProto_AttributeType_FLOATS:
                for (size_t i = 0; i < attribute.floats_size(); ++i)
                    ss << (i ? " " : "") << attribute.floats(i);
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
            SYNET_ERROR("Can't convert node[" << index << "]: " << NodeString(node) << " !");
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
                SYNET_ERROR("Can't find attribute '" << name << "' !");
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_INT)
                SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
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
                SYNET_ERROR("Can't find attribute '" << name << "' !");
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_FLOAT)
                SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
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
                SYNET_ERROR("Can't find attribute '" << name << "' !");
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_STRING)
                SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
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
                SYNET_ERROR("Can't find attribute '" << name << "' !");
            }
            if (attribute->type() != onnx::AttributeProto_AttributeType_INTS)
                SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
            values.resize(attribute->ints_size());
            for(size_t i = 0; i < attribute->ints_size(); ++i)
                values[i] = (T)attribute->ints(i);
            return true;
        }
    };

    //---------------------------------------------------------------------------------------------

    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, 
        const OnnxParam& onnxParam, const OptimizerParam& optParam)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcGraph, trans, dstXml, dstBin, onnxParam, optParam);
    }
}

#endif