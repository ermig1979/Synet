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

#include "Cvt/OnnxRuntime/OnnxRuntime.h"
#include "Cvt/OnnxRuntime/Convert.h"

namespace Synet
{
#if defined(SYNET_ONNXRUNTIME_ENABLE)
    bool OnnxToSynet::Convert(String srcGraphPath, bool trans, const String& dstModelPath, const String& dstWeightPath, const OnnxParam& onnxParam, const OptimizerParam& optParam)
    {
        if (!Cpl::FileExists(srcGraphPath))
        {
            String altGraphPath = Cpl::ChangeExtension(srcGraphPath, ".dat");
            if (altGraphPath != srcGraphPath)
            {
                if (!Cpl::FileExists(altGraphPath))
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

    bool OnnxToSynet::LoadModel(const String& path, onnx::ModelProto& model)
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

    bool OnnxToSynet::ConvertModel(const onnx::ModelProto& model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Bytes& reordered)
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
            if ((node.op_type() == "Gather" || node.op_type() == "GatherElements" || node.op_type() == "GatherND") && !ConvertGatherNode(node, network.layers(), layer))
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
            if ((node.op_type() == "QLinearAdd") && !ConvertQLinearAddNode(node, network.layers(), original, layer))
                return ErrorMessage(i, node);
            if ((node.op_type() == "QLinearConv") && !ConvertQLinearConvNode(node, trans, network.layers(), original, layer, reordered, &tensorFormatMap))
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
            if (node.op_type() == "Reshape" && !ConvertReshapeNode(node, trans, network.layers(), original, onnxParam, layer, &tensorFormatMap))
                return ErrorMessage(i, node);
            if (node.op_type() == "Resize" && !ConvertResizeNode(node, network.layers(), original, layer))
                return ErrorMessage(i, node);
            if (node.op_type() == "ScaledDotProductAttention" && !ConvertScaledDotProductAttentionNode(node, network.layers(), layer, reordered))
                return ErrorMessage(i, node);
            if (node.op_type() == "ScatterElements" && !ConvertScatterElementsNode(node, network.layers(), original, layer, reordered))
                return ErrorMessage(i, node);
            if (node.op_type() == "ScatterND" && !ConvertScatterNdNode(node, network.layers(), original, layer, reordered))
                return ErrorMessage(i, node);
            if (node.op_type() == "Shape" && !ConvertShapeNode(node, trans, network.layers(), onnxParam, layer))
                return ErrorMessage(i, node);
            if (node.op_type() == "Sigmoid" && !ConvertSigmoidNode(node, layer))
                return ErrorMessage(i, node);
            if (node.op_type() == "Sin" && !ConvertSinNode(node, layer))
                return ErrorMessage(i, node);
            if (node.op_type() == "Slice" && !ConvertSliceNode(node, trans, network.layers(), layer, &tensorFormatMap))
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

    //--------------------------------------------------------------------------------------------------

    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, const OnnxParam& onnxParam, const OptimizerParam& optParam)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcGraph, trans, dstXml, dstBin, onnxParam, optParam);
    }
#else
    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin,  const OnnxParam& onnxParam, const OptimizerParam& optParam)
    {
        return false;
    }
#endif
}
