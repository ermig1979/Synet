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

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

//#define SYNET_ONNX_PARSE_STOP_ON_ERROR

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(const String& srcParamPath, const String& srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            onnx::ModelProto model;
            if (!LoadModel(srcGraphPath, model))
                return false;

            Synet::NetworkParamHolder holder;
            Vector weight;
            if (!ConvertModel(model, trans, holder(), weight))
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

        bool ConvertModel(const onnx::ModelProto & model, bool trans, Synet::NetworkParam& network, Vector& reordered)
        {
            const onnx::GraphProto& graph = model.graph();

            //PrintGraph(graph, std::cout, false);

            network.name() = graph.name();

            Vector original;
            for (size_t i = 0; i < graph.initializer_size(); ++i)
            {
                const onnx::TensorProto& tensor = graph.initializer(i);
                if (!ConvertInitializer(tensor, network, original))
                {
                    std::cout << "Can't convert initializer '" << tensor.name() << "' !" << std::endl;
                    return false;
                }
            }
            reordered = original;

            for (size_t i = 0; i < graph.input_size(); ++i)
            {
                const onnx::ValueInfoProto& input = graph.input(i);
                if (!ConvertInput(input, trans, network))
                {
                    std::cout << "Can't convert input '" << input.name() << "' !" << std::endl;
                    return false;
                }
            }

            for (size_t i = 0; i < graph.node_size(); ++i)
            {
                const onnx::NodeProto& node = graph.node(i);
                LayerParam layer;

                layer.name() = node.name();
                for (size_t j = 0; j < node.input_size(); ++j)
                    layer.src().push_back(node.input(j));
                for (size_t j = 0; j < node.output_size(); ++j)
                    layer.dst().push_back(node.output(j));
                if (layer.dst().size() == 1)
                    layer.name() = layer.dst()[0];

                if (node.op_type() == "Add" && !ConvertAddNode(node, layer))
                    return ErrorMessage(node);
                if (node.op_type() == "Clip" && !ConvertClipNode(node, layer))
                    return ErrorMessage(node);
                if (node.op_type() == "Conv" && !ConvertConvNode(node, trans, network.layers(), original, layer, reordered))
                    return ErrorMessage(node);

#if defined(SYNET_ONNX_PARSE_STOP_ON_ERROR)
                if (layer.type() == LayerTypeUnknown)
                    return ErrorMessage(node);
#else
                if (layer.type() == LayerTypeUnknown)
                {
                    NotImplemented(node, layer);
                    std::cout << "Not implemented layer: " << NodeString(node) << std::endl;
                }
#endif
                network.layers().push_back(layer);
            }

            return true;
        }

        bool ConvertInitializer(const onnx::TensorProto& tensor, Synet::NetworkParam& network, Vector& weight)
        {
            LayerParam layer;
            layer.name() = tensor.name();
            if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
            {
                layer.type() = LayerTypeConst;
                layer.weight().resize(1);
                layer.weight()[0].type() = TensorType32f;
                uint64_t size = 1, offset = weight.size();
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= tensor.dims(i);
                    layer.weight()[0].dim().push_back(size_t(tensor.dims(i)));
                }
                layer.weight()[0].offset() = offset * sizeof(float);
                layer.weight()[0].size() = size * sizeof(float);
                if (tensor.has_raw_data() && size)
                {
                    weight.resize(offset + size);
                    memcpy(weight.data() + offset, tensor.raw_data().c_str(), layer.weight()[0].size());
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
                    size *= tensor.dims(i);
                    layer.meta().alpha().shape().push_back(size_t(tensor.dims(i)));
                }
                layer.meta().alpha().i64().resize(size);
                if (tensor.has_raw_data() && size)
                {
                    for (size_t i = 0; i < size; ++i)
                        layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
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
            layer.name() = input.name();
            layer.type() = LayerTypeInput;
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
            layer.input().shape()[0].dim() = shape;
            network.layers().push_back(layer);
            return true;
        }

        bool ConvertAddNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            if (node.input_size() != 2)
                return false;
            layer.type() = Synet::LayerTypeEltwise;
            layer.eltwise().operation() = EltwiseOperationTypeSum;
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
            if (trans && !PermutedToNchw(layers, layer.src(), true, false))
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool PrintGraph(const onnx::GraphProto& graph, std::ostream & os, bool init)
        {
            os << std::endl;
            os << "graph name: " << graph.name() << std::endl;
            for (size_t i = 0; i < graph.input_size(); ++i)
                os << " input[" << i << "] " << ValueInfoString(graph.input(i)) << std::endl;
            if (init)
            {
                for (size_t i = 0; i < graph.initializer_size(); ++i)
                    os << " const[" << i << "] " << TensorString(graph.initializer(i)) << std::endl;
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

        String TensorString(const onnx::TensorProto& tensor)
        {
            std::stringstream ss;
            ss << tensor.name() << " ";
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT: ss << "f32"; break;
            case onnx::TensorProto_DataType_INT64: ss << "i64"; break;
            default: ss << " unknown-" << tensor.data_type();
            }

            ss << " {";
            uint64_t size = 1;
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                ss << " " << tensor.dims(i);
                size *= tensor.dims(i);
            }
            size_t printSize = std::min<size_t>(3, size);
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
            ss << " ... ]";
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

        bool ErrorMessage(const onnx::NodeProto& node)
        {
            std::cout << "Can't convert node : " << NodeString(node) << " !" << std::endl;
            return false;
        }

        const onnx::AttributeProto * GetAtrribute(const onnx::NodeProto& node, const String& name)
        {
            for (size_t i = 0; i < node.attribute_size(); ++i)
                if (node.attribute(i).name() == name)
                    return &node.attribute(i);
            return NULL;
        }

        template<class T> bool ConvertAtrributeInt(const onnx::NodeProto& node, const String& name, T & value)
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
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

        bool ConvertAtrributeFloat(const onnx::NodeProto& node, const String& name, float & value)
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
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

        template<class T> bool ConvertAtrributeInts(const onnx::NodeProto& node, const String& name, std::vector<T>& values)
        {
            const onnx::AttributeProto* attribute = GetAtrribute(node, name);
            if (attribute == NULL)
            {
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
                values[i] = attribute->ints(i);
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