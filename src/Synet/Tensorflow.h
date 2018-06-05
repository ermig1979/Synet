/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

#include "Synet/Network.h"

#if defined(SYNET_TENSORFLOW_ENABLE)

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4267 4800 4554)
#endif

#include "tensorflow/core/public/session.h"

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#define SYNET_TENSORFLOW_DEBUG
#define SYNET_TENSORFLOW_DYNAMIC

namespace Synet
{
    class TensorflowToSynet
    {
    public:

        bool Convert(const String &, const String & srcGraphPath, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), srcGraphPath, &_graph);
            if (!status.ok())
            {
                status = tensorflow::ReadTextProto(tensorflow::Env::Default(), srcGraphPath, &_graph);
                if (!status.ok())
                {
                    std::cout << "Error in building graph: " << status.error_message() << std::endl;
                    return false;
                }
            }

            Synet::NetworkParamHolder holder;
            Tensors weight;
            if (!ConvertNetwork(holder(), weight))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(weight, dstWeightPath))
                return false;

            return true;
        }

    private:

        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<std::pair<String, int>> NameIndexVector;
        typedef std::map<String, int> NameIndexMap;
        typedef std::map<String, String> NameNameMap;
        typedef std::set<String> NameSet;
        typedef Synet::Tensor<int> ITensor;
        typedef std::map<String, ITensor> IConstMap;
        typedef std::map<String, Tensor> FConstMap;
        typedef std::vector<Shape> Shapes;
        typedef std::map<String, Shapes> ShapeMap;

        struct Pin
        {
            String name;
            int index;
            Pin(const String & name_, int index_ = 0) : name(name_), index(index_) {}
            Pin() : name(""), index(-1) {}
        };

        tensorflow::GraphDef _graph;
        NameIndexMap _valueId;
        NameSet _ignore, _meta;

        IConstMap _iConst;
        FConstMap _fConst;
        ShapeMap _shape;

        bool ConvertNetwork(Synet::NetworkParam & network, Tensors & weight)
        {
            RemoveUnused();

            AddConstFloat();

#ifdef SYNET_TENSORFLOW_DEBUG
            for (int i = 0; i < _graph.node_size(); ++i)
                PrintLayerAttr(_graph.node(i));
#endif //SYNET_TENSORFLOW_DEBUG

            NameNameMap shapes;
            network.name() = "tensorflow_unknown";
            for (int i = 0; i < _graph.node_size(); ++i)
            {
                const ::tensorflow::NodeDef & node = _graph.node(i);
                google::protobuf::Map<String, tensorflow::AttrValue> attr = node.attr();

                if (_ignore.find(node.name()) != _ignore.end())
                    continue;

                String type = node.op();
                LayerParam layer;
                layer.name() = node.name();
                if (type == "Placeholder")
                {
                    if (node.attr().at("dtype").type() == 1)
                    {
                        if (!ConvertInputLayer(node, layer))
                            return false;
                    }
                    else
                    {
                        layer.type() = LayerTypeMeta;
                        layer.meta().type() = MetaTypeInput;
                        layer.dst().push_back(layer.name());
                        _meta.insert(node.name());
                    }
                }
                else if (AllMeta(node) || type == "Shape" || type == "Const")
                {
                    if (!ConvertMetaLayer(node, layer))
                        return false;
                }
                else if (type == "Conv2D")
                {
                    if (!ConvertConvolutionLayer(node, layer, weight))
                        return false;
                }
                else if (type == "Abs")
                {
                    layer.type() = LayerTypeAbs;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Relu")
                {
                    layer.type() = LayerTypeRelu;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Relu6")
                {
                    layer.type() = LayerTypeRestrictRange;
                    layer.restrictRange().lower() = 0;
                    layer.restrictRange().upper() = 6;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "MaxPool")
                {
                    if (!ConvertPoolingLayer(node, layer))
                        return false;
                }
                else if (type == "Sigmoid")
                {
                    layer.type() = LayerTypeSigmoid;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Tanh")
                {
                    layer.type() = LayerTypeTanh;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Softmax")
                {
                    layer.type() = LayerTypeSoftmax;
                    layer.softmax().axis() = 0;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "BiasAdd" || type == "Add")
                {
                    bool haveConst = false;
                    for (int j = 0; !haveConst && j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        haveConst = _valueId.find(input.name) != _valueId.end();
                    }
                    if (haveConst)
                    {
                        layer.src().push_back(node.input(0));
                        layer.type() = LayerTypeBias;
                        layer.weight().resize(1);
                        weight.push_back(Tensor());
                        ConvertKernel(GetConst(_graph, node, _valueId), weight.back());
                        layer.weight()[0].dim() = weight.back().Shape();
                        layer.dst() = layer.src();
                    }
                    else
                    {
                        layer.type() = LayerTypeEltwise;
                        layer.eltwise().operation() = EltwiseOperationTypeSum;
                        layer.src().push_back(node.input(0));
                        layer.src().push_back(node.input(1));
                        layer.dst().push_back(layer.name());
                    }
                }
                else if (type == "Mul")
                {
                    assert(node.input_size() == 2);
                    ptrdiff_t constIndex = -1, constCount = 0;
                    for (int j = 0; j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        if (_valueId.find(input.name) != _valueId.end())
                        {
                            assert(constIndex == -1);
                            constIndex = j;
                            constCount++;
                        }
                    }
                    assert(constCount < 2);
                    if (constIndex < 0)
                    {
                        layer.type() = LayerTypeEltwise;
                        layer.eltwise().operation() = EltwiseOperationTypeProduct;
                        layer.src().push_back(node.input(0));
                        layer.src().push_back(node.input(1));
                    }
                    else
                    {
                        layer.type() = LayerTypeScale;
                        layer.src().push_back(node.input(1 - (int)constIndex));
                        layer.scale().biasTerm() = false;
                        Tensor scale;
                        ConvertKernel(GetConst(_graph, node, _valueId), scale);
                        if (scale.Size() == 1)
                            layer.scale().axis() = 0;
                        layer.weight().resize(1);
                        layer.weight()[0].dim() = scale.Shape();
                        weight.push_back(scale);
                    }
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Sub")
                {
                    assert(node.input_size() == 2);
                    ptrdiff_t constIndex = -1, constCount = 0;
                    for (int j = 0; j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        if (_valueId.find(input.name) != _valueId.end())
                        {
                            assert(constIndex == -1);
                            constIndex = j;
                            constCount++;
                        }
                    }
                    if (constCount == 2)
                    {
                        if (_iConst.find(node.input(0)) != _iConst.end() && _iConst.find(node.input(1)) != _iConst.end())
                        {
                            const ITensor & a = _iConst[node.input(0)];
                            const ITensor & b = _iConst[node.input(1)];
                            assert(a.Shape() == b.Shape());
                            ITensor c(a.Shape());
                            for (size_t j = 0; j < a.Size(); ++j)
                                c.CpuData()[j] = a.CpuData()[j] - b.CpuData()[j];
                            _iConst[node.name()] = c;
                            _ignore.insert(node.name());
                            continue;
                        }
                        SetNotImplemented(layer, node);
                    }
                    else if (constIndex < 0)
                    {
                        layer.type() = LayerTypeEltwise;
                        layer.eltwise().operation() = EltwiseOperationTypeSum;
                        layer.eltwise().coefficients() = Floats({ 1.0f, -1.0f });
                        layer.src().push_back(node.input(0));
                        layer.src().push_back(node.input(1));
                        layer.dst().push_back(layer.name());
                    }
                    else
                    {
                        layer.type() = LayerTypeScale;
                        layer.src().push_back(node.input(1 - (int)constIndex));
                        layer.scale().biasTerm() = true;
                        Tensor scale, bias;
                        ConvertKernel(GetConst(_graph, node, _valueId), bias);
                        if (constIndex)
                        {
                            for (size_t j = 0; j < bias.Size(); ++j)
                                bias.CpuData()[j] *= -1.0f;
                        }
                        scale.Reshape(bias.Shape(), constIndex ? 1.0f : -1.0f);
                        if (bias.Size() == 1)
                            layer.scale().axis() = 0;
                        layer.weight().resize(2);
                        layer.weight()[0].dim() = scale.Shape();
                        layer.weight()[1].dim() = bias.Shape();
                        weight.push_back(scale);
                        weight.push_back(bias);
                        layer.dst() = layer.src();
                    }
                }
                else if (type == "MatMul")
                {
                    if (!ConvertInnerProductLayer(node, layer, weight))
                        return false;
                }
                else if (type == "Reshape")
                {
                    assert(node.input_size() == 2);
                    layer.type() = LayerTypeReshape;
                    layer.src().push_back(node.input(0));
                    if (_meta.find(node.input(1)) != _meta.end())
                    {
                        layer.src().push_back(node.input(1));
                        layer.dst().push_back(layer.name());
                    }
                    else
                    {
                        SetNotImplemented(layer, node);
                    }
                }
                else if (type == "ExpandDims")
                {
                    if (!ConvertExpandDimsLayer(node, layer))
                        return false;
                }
                else if (type == "Squeeze")
                {
                    layer.type() = LayerTypeSqueeze;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Transpose")
                {
                    SetNotImplemented(layer, node);
                    //layer.type() = LayerTypePermute;
                    //layer.src().push_back(node.input(0));
                    //const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                    //layer.permute().order().resize(tensor.tensor_shape().dim(0).size());
                    //for (size_t j = 0; j < layer.permute().order().size(); ++j)
                    //    layer.permute().order()[j] = ((int*)tensor.tensor_content().c_str())[j];
                    //layer.dst().push_back(layer.name());
                }
                else if (type == "Fill")
                {
                    layer.type() = LayerTypeFill;
                    layer.src().push_back(node.input(0));
                    const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                    layer.fill().value() = tensor.float_val(0);
                    layer.dst() = layer.src();
                }
                else if (type == "Pad")
                {
                    assert(node.input_size() == 2);
                    layer.type() = LayerTypePad;
                    layer.src().push_back(node.input(0));
                    layer.src().push_back(node.input(1));
                    layer.dst().push_back(layer.name());
                }
                else
                {
                    SetNotImplemented(layer, node);
                }
                network.layers().push_back(layer);
            }

            return true;
        }

        //---------------------------------------------------------------------

        bool ConvertConvolutionLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Tensors & weight)
        {
            layer.type() = LayerTypeConvolution;
            layer.src().push_back(node.input(0));
            layer.convolution().biasTerm() = false;
            layer.weight().resize(1);
            weight.push_back(Tensor());
            ConvertKernel(GetConst(_graph, node, _valueId), weight.back());
            layer.weight()[0].dim() = weight.back().Shape();

            NameIndexVector nextLayers = NextLayers(node.name(), "BiasAdd");
            if (nextLayers.size() == 1)
            {
                layer.convolution().biasTerm() = true;
                layer.weight().resize(2);
                weight.push_back(Tensor());
                const ::tensorflow::NodeDef & bias = _graph.node(nextLayers[0].second);
                ConvertKernel(GetConst(_graph, bias, _valueId), weight.back());
                layer.weight()[1].dim() = weight.back().Shape();
                _ignore.insert(nextLayers[0].first);
                ExcludeLayer(nextLayers[0].second, 0, false);
            }

            layer.convolution().kernel() = Shape({ layer.weight()[0].dim()[2], layer.weight()[0].dim()[3] });
            layer.convolution().outputNum() = (uint32_t)layer.weight()[0].dim()[0];
            if (node.attr().find("strides") != node.attr().end())
            {
                const tensorflow::AttrValue_ListValue & list = node.attr().at("strides").list();
                assert(list.i_size() == 4 && list.i(0) == 1 && list.i(3) == 1);
                layer.convolution().stride() = Shape({ (size_t)list.i(1), (size_t)list.i(2) });
            }
            if (node.attr().find("padding") != node.attr().end())
            {
                const String & pad = node.attr().at("padding").s();
                if (pad == "SAME")
                {
                    Shape kernel = layer.convolution().kernel();
                    layer.convolution().pad() = Shape({ kernel[0] / 2, kernel[1] / 2 });
                }
            }
            layer.dst().push_back(layer.name());
            return true;
        }

        bool ConvertExpandDimsLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeExpandDims;
            layer.src().push_back(node.input(0));
            if (node.attr().find("Tdim") != node.attr().end())
            {
                layer.expandDims().axis() = (int)node.attr().at("Tdim").i();
            }
            else
            {
                const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId);
                layer.expandDims().axis() = tensor.int_val(0);
            }
            layer.dst().push_back(layer.name());
            return true;
        }

        bool ConvertInnerProductLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Tensors & weight)
        {
            assert(node.input_size() == 2);
            bool haveConst = false;
            for (int j = 0; !haveConst && j < node.input_size(); ++j)
            {
                Pin input = ParsePin(node.input(j));
                haveConst = _valueId.find(input.name) != _valueId.end();
            }
            layer.type() = LayerTypeInnerProduct;
            layer.innerProduct().biasTerm() = false;
            layer.src().push_back(node.input(0));
            if (node.attr().find("transpose_a") != node.attr().end())
                layer.innerProduct().transposeA() = node.attr().at("transpose_a").b();
            if (node.attr().find("transpose_b") != node.attr().end())
                layer.innerProduct().transposeB() = node.attr().at("transpose_b").b();
            if (haveConst)
            {
                layer.weight().resize(1);
                weight.push_back(Tensor());
                ConvertKernel(GetConst(_graph, node, _valueId), weight.back());
                layer.weight()[0].dim() = weight.back().Shape();
                layer.innerProduct().outputNum() = (uint32_t)(layer.innerProduct().transposeB() ? layer.weight()[0].dim()[1] : layer.weight()[0].dim()[0]);
                NameIndexVector nextLayers = NextLayers(node.name(), "BiasAdd");
                if (nextLayers.size() == 1)
                {
                    layer.innerProduct().biasTerm() = true;
                    layer.weight().resize(2);
                    weight.push_back(Tensor());
                    const ::tensorflow::NodeDef & bias = _graph.node(nextLayers[0].second);
                    ConvertKernel(GetConst(_graph, bias, _valueId), weight.back());
                    layer.weight()[1].dim() = weight.back().Shape();
                    _ignore.insert(nextLayers[0].first);
                    ExcludeLayer(nextLayers[0].second, 0, false);
                }
            }
            else
            {
                layer.src().push_back(node.input(1));
            }
            layer.dst().push_back(layer.name());
            return true;
        }

        bool ConvertInputLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeInput;
            layer.dst().push_back(layer.name());
            if (node.attr().find("shape") != node.attr().end())
            {
                const tensorflow::TensorShapeProto & src = node.attr().find("shape")->second.shape();
                layer.input().shape().resize(1);
                Shape & dst = layer.input().shape()[0].dim();
                dst.resize(src.dim_size());
                for (int d = 0; d < src.dim_size(); d++)
                    dst[d] = src.dim(d).size();
                if (dst.size() == 4)
                    dst = Shape({ size_t(1), dst[3], dst[1], dst[2] });
            }
            return true;
        }

        bool ConvertMetaLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeMeta;
            String type = node.op();
            if (type == "Concat" || type == "ConcatV2")
            {
                layer.meta().type() = MetaTypePack;
                int axisId = (type == "Concat" ? 0 : node.input_size() - 1);
                for (int j = 0; j < node.input_size(); ++j)
                {
                    if (j != axisId)
                        layer.src().push_back(node.input(j));
                }
            }
            else if (type == "Const")
            {
                layer.meta().type() = MetaTypeConst;
                const tensorflow::TensorProto & src = node.attr().at("value").tensor();
                ITensor dst;
                if (src.int_val_size())
                {
                    dst.Reshape({ (size_t)src.int_val_size() });
                    for (int j = 0; j < src.int_val_size(); j++)
                        dst.CpuData()[j] = src.int_val(j);
                }
                else
                    ConvertKernel<int, int>(src, dst);
                layer.meta().alpha().assign(dst.CpuData(), dst.CpuData() + dst.Size());
            }
            else if (type == "Pack")
            {
                layer.meta().type() = MetaTypePack;
                for (int j = 0; j < node.input_size(); ++j)
                    layer.src().push_back(node.input(j));
            }
            else if (type == "Range")
            {
                layer.meta().type() = MetaTypeRange;
                layer.src().push_back(node.input(0));
                layer.src().push_back(node.input(1));
                layer.src().push_back(node.input(2));
            }
            else if (type == "Shape")
            {
                layer.meta().type() = MetaTypeShape;
                layer.src().push_back(node.input(0));
            }
            else if (type == "Slice")
            {
                layer.meta().type() = MetaTypeSlice;
                layer.src().push_back(node.input(0));
                layer.src().push_back(node.input(1));
                layer.src().push_back(node.input(2));
            }
            else if (type == "StridedSlice")
            {
                layer.meta().type() = MetaTypeStridedSlice;
                layer.src().push_back(node.input(0));
                layer.src().push_back(node.input(1));
                layer.src().push_back(node.input(2));
                layer.src().push_back(node.input(3));
            }
            else if (type == "Sub")
            {
                layer.meta().type() = MetaTypeSub;
                layer.src().push_back(node.input(0));
                layer.src().push_back(node.input(1));
            }
            else if (type == "TensorArrayV3" || type == "Enter")
            {
                layer.meta().type() = MetaTypeStub;
            }
            else
            {
                SetNotImplemented(layer, node);
            }
            layer.dst().push_back(layer.name());
            _meta.insert(node.name());
            return true;
        }

        bool ConvertPoolingLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypePooling;
            layer.src().push_back(node.input(0));
            if (node.op() == "MaxPool")
                layer.pooling().method() = PoolingMethodTypeMax;
            else
                return false;
            const tensorflow::AttrValue_ListValue & kernel = node.attr().at("ksize").list();
            assert(kernel.i_size() == 4 && kernel.i(0) == 1 && kernel.i(3) == 1);
            layer.pooling().kernel() = Shape({ (size_t)kernel.i(1), (size_t)kernel.i(2) });
            if (node.attr().find("strides") != node.attr().end())
            {
                const tensorflow::AttrValue_ListValue & list = node.attr().at("strides").list();
                assert(list.i_size() == 4 && list.i(0) == 1 && list.i(3) == 1);
                layer.pooling().stride() = Shape({ (size_t)list.i(1), (size_t)list.i(2) });
            }
            if (node.attr().find("padding") != node.attr().end())
            {
                const String & pad = node.attr().at("padding").s();
                if (pad == "SAME")
                {
                    Shape kernel = layer.pooling().kernel();
                    layer.pooling().pad() = Shape({ kernel[0] / 2, kernel[1] / 2 });
                }
            }
            layer.dst().push_back(layer.name());
            return true;
        }

        //---------------------------------------------------------------------

        void RemoveUnused()
        {
            typedef std::map<String, String>  UnusedMap;
            UnusedMap unused;
            std::vector<int> unusedIndex;

            int layersCount = _graph.node_size();
            for (int i = 0; i < layersCount; i++)
            {
                const tensorflow::NodeDef & layer = _graph.node(i);
                String type = layer.op();
                if (type == "Identity" || type == "Dropout" || layer.name().find("dropout") == 0)
                {
                    unusedIndex.push_back(i);
                    if (layer.input_size())
                        unused[layer.name()] = layer.input(0);
                }
            }

            for (int i = 0; i < layersCount; i++)
            {
                tensorflow::NodeDef * layer = _graph.mutable_node(i);
                for (int j = 0; j < layer->input_size(); j++)
                {
                    String inputOpName = layer->input(j);
                    UnusedMap::iterator it = unused.find(inputOpName);
                    if (it != unused.end())
                        layer->set_input(j, it->second);
                }
            }

            std::sort(unusedIndex.begin(), unusedIndex.end());
            int removedNodes = 0;
            for (size_t i = 0; i < unusedIndex.size(); i++)
            {
                int startId = unusedIndex[i] - removedNodes;
                _graph.mutable_node()->DeleteSubrange(startId, 1);
                removedNodes++;
            }
        }

        Pin ParsePin(const String & name)
        {
            Pin pin(name);
            size_t delimiter = name.find_first_of(":");
            if (delimiter != std::string::npos)
            {
                pin.name = name.substr(0, delimiter);
                std::istringstream(name.substr(delimiter + 1)) >> pin.index;
            }
            return pin;
        }

        NameIndexVector NextLayers(const String & name, const String & type = "")
        {
            NameIndexVector layers;
            for (int i = 0; i < _graph.node_size(); i++)
            {
                const tensorflow::NodeDef & layer = _graph.node(i);
                for (int j = 0; j < layer.input_size(); j++)
                {
                    String input_op_name = ParsePin(layer.input(j)).name;
                    if (input_op_name == name && (type.empty() ? true : type == layer.op()))
                        layers.push_back(std::make_pair(layer.name(), i));
                }
            }
            return layers;
        }

        void AddConstFloat()
        {
            for (int i = 0; i < _graph.node_size(); i++)
            {
                const tensorflow::NodeDef & node = _graph.node(i);
                if (node.op() == "Const" && node.attr().at("dtype").type() == tensorflow::DT_FLOAT)
                {
                    _ignore.insert(node.name());
                    if (node.attr().find("value") != node.attr().end())
                        _valueId.insert(std::make_pair(node.name(), i));
                    const tensorflow::TensorProto & src = node.attr().at("value").tensor();
                    Tensor dst;
                    if (src.float_val_size())
                    {
                        dst.Reshape({ (size_t)src.float_val_size() });
                        for (int j = 0; j < src.float_val_size(); j++)
                            dst.CpuData()[j] = src.float_val(j);
                    }
                    else
                        ConvertKernel<float, float>(src, dst);
                    _fConst[node.name()] = dst;
                }
            }
        }

        void AddConstInt()
        {
            for (int i = 0; i < _graph.node_size(); i++)
            {
                const tensorflow::NodeDef & node = _graph.node(i);
                if (node.op() == "Const" && node.attr().at("dtype").type() == tensorflow::DT_INT32)
                {
                    _ignore.insert(node.name());
                    if (node.attr().find("value") != node.attr().end())
                        _valueId.insert(std::make_pair(node.name(), i));
                    const tensorflow::TensorProto & src = node.attr().at("value").tensor();
                    ITensor dst;
                    if (src.int_val_size())
                    {
                        dst.Reshape({ (size_t)src.int_val_size() });
                        for (int j = 0; j < src.int_val_size(); j++)
                            dst.CpuData()[j] = src.int_val(j);
                    }
                    else
                        ConvertKernel<int, int>(src, dst);
                    _iConst[node.name()] = dst;
                }
            }
        }

        const tensorflow::TensorProto & GetConst(const tensorflow::GraphDef & graph, const tensorflow::NodeDef & node, const NameIndexMap & consts, int index = -1, int * actual = NULL)
        {
            if (index == -1)
            {
                for (int i = 0; i < node.input_size(); i++)
                {
                    Pin input = ParsePin(node.input(i));
                    if (consts.find(input.name) != consts.end())
                    {
                        assert(index == -1);
                        index = i;
                        break;
                    }
                }
            }
            assert(index != -1);

            Pin kernel_inp = ParsePin(node.input(index));
            assert(consts.find(kernel_inp.name) != consts.end());
            assert(kernel_inp.index == 0);

            if (actual)
                *actual = index;

            int nodeIdx = consts.at(kernel_inp.name);
            return graph.node(nodeIdx).attr().at("value").tensor();
        }

        bool AllConstInt(const tensorflow::NodeDef & node) const
        {
            for (int j = 0; j < node.input_size(); ++j)
            {
                if (_iConst.find(node.input(j)) == _iConst.end())
                    return false;
            }
            return true;
        }

        bool AllMeta(const tensorflow::NodeDef & node) const
        {
            for (int j = 0; j < node.input_size(); ++j)
            {
                String name = node.input(j);
                if (name.find(":") != String::npos)
                    name = name.substr(0, name.find(":"));
                if (_meta.find(name) == _meta.end())
                    return false;
            }
            return true;
        }

        template <class TS, class TD> void ConvertKernel(const tensorflow::TensorProto & src, Synet::Tensor<TD> & dst)
        {
            Shape shape = GetShape(src); 
            if(shape.size() == 4)
                shape = Shape({shape[3], shape[2], shape[0], shape[1]});
            dst.Reshape(shape);    
            TD * pDst = dst.CpuData();

            const String & content = src.tensor_content();
            const TS * pSrc = (TS*)content.c_str();

            if (shape.size() == 4)
            {
                size_t out_c = shape[0], input_c = shape[1], height = shape[2], width = shape[3];
                for (size_t i_oc = 0; i_oc < out_c; i_oc++)
                {
                    for (size_t i_ic = 0; i_ic < input_c; i_ic++)
                    {
                        for (size_t i_h = 0; i_h < height; i_h++)
                        {
                            for (size_t i_w = 0; i_w < width; i_w++)
                            {
                                size_t dst_i = input_c*height*width*i_oc + height*width*i_ic + width*i_h + i_w;
                                size_t src_i = out_c*input_c*width*i_h + out_c*input_c*i_w + out_c*i_ic + i_oc;
                                pDst[dst_i] = (TD)pSrc[src_i];
                            }
                        }
                    }
                }
            }
            else
            {
                size_t size = 1;
                for (size_t i = 0; i < shape.size(); i++)
                    size *= shape[i];
                for (size_t i = 0; i < size; i++)
                    pDst[i] = (TD)pSrc[i];
            }
        }

        void ConvertKernel(const tensorflow::TensorProto & src, Tensor & dst)
        {
            switch (src.dtype())
            {
            case tensorflow::DT_FLOAT:
                if (src.float_val_size())
                {
                    Shape shape(1, src.float_val_size());
                    dst.Reshape(shape);
                    for (int i = 0; i < src.float_val_size(); i++)
                        dst.CpuData()[i] = src.float_val(i);
                }
                else
                    ConvertKernel<float, float>(src, dst);
                break;
            case tensorflow::DT_INT32:
                if (src.int_val_size())
                {
                    Shape shape(1, src.int_val_size());
                    dst.Reshape(shape);
                    for (int i = 0; i < src.int_val_size(); i++)
                        dst.CpuData()[i] = (float)src.int_val(i);
                }
                else
                    ConvertKernel<int, float>(src, dst);
                break;
            default: 
                assert(0);
            }
        }

        Shape GetShape(const tensorflow::TensorProto & tensor)
        {
            Shape shape;
            if (tensor.has_tensor_shape())
            {
                const tensorflow::TensorShapeProto & ts = tensor.tensor_shape();
                int i, n = ts.dim_size();
                if (n)
                {
                    shape.resize(n);
                    for (i = 0; i < n; i++)
                        shape[i] = (int)ts.dim(i).size();
                }
                else
                    shape.resize(1, 1);
            }
            return shape;
        }

        void ExcludeLayer(const int layerIndex, const int inputIndex, bool remove = true) 
        {
            String layerName = _graph.node(layerIndex).name();
            NameIndexVector layers = NextLayers(layerName);
            String removedInput = _graph.node(layerIndex).input(inputIndex);
            for (size_t i = 0; i < layers.size(); i++)
            {
                tensorflow::NodeDef * layer = _graph.mutable_node(layers[i].second);
                for (int input_id = 0; input_id < layer->input_size(); input_id++) 
                {
                    String inputName = layer->input(input_id);
                    if (inputName == layerName)
                        layer->set_input(input_id, removedInput);
                }
            }
            if (remove)
                _graph.mutable_node()->DeleteSubrange(layerIndex, 1);
        }

        void RenameInput(const String & oldName, const String & newName)
        {
            for (int i = 0; i < _graph.node_size(); ++i)
            {
                tensorflow::NodeDef * node = _graph.mutable_node(i);
                for (int j = 0; j < node->input_size(); ++j)
                {
                    if (node->input(j) == oldName)
                        node->set_input(j, newName);
                }
            }
        }

        bool SetShape(const LayerParam & param)
        {
            if (param.type() == LayerTypeInput)
            {
                _shape[param.name()].push_back(param.input().shape()[0].dim());
            }
            else
            {
                typedef Synet::Network<float> Net;
                std::unique_ptr<Net::Layer> layer(Net::Create(param));
                if (!layer)
                    return false;
                std::vector<std::shared_ptr<Tensor>> tensors;
                Net::TensorPtrs src, buf, dst;
                tensors.push_back(std::make_shared<Tensor>());
                buf.push_back(tensors.back().get());
                tensors.push_back(std::make_shared<Tensor>());
                dst.push_back(tensors.back().get());
                for (size_t i = 0; i < param.src().size(); ++i)
                {
                    if (_shape.find(param.src()[i]) == _shape.end())
                        return false;
                    const Shape & shape = _shape[param.src()[i]][0];
                    tensors.push_back(std::make_shared<Tensor>(shape));
                    src.push_back(tensors.back().get());
                }
                layer->Setup(src, buf, dst);
                layer->Reshape(src, buf, dst);
                _shape[param.name()].push_back(dst[0]->Shape());            
            }
            return true;
        }

        String NotImplementedMarker()
        {
            return "~~~NOT_IMPLEMENTED~~~";
        }

        bool IsNotImplemented(const LayerParam & layer)
        {
            for (size_t i = 0; i < layer.debug().size(); ++i)
            {
                if (layer.debug()[i] == NotImplementedMarker())
                    return true;
            }
            return false;
        }

        void SetNotImplemented(LayerParam & layer, const tensorflow::NodeDef & node, const String & info = String())
        {
#ifdef SYNET_TENSORFLOW_DEBUG
            if (!IsNotImplemented(layer))
            {
                layer.debug().clear();
                layer.debug().push_back(NotImplementedMarker());
                layer.debug().push_back(node.op());
                if(!info.empty())
                    layer.debug().push_back(info);
                layer.src().clear();
                for (size_t j = 0; j < (size_t)node.input_size(); ++j)
                    layer.src().push_back(node.input((int)j));
            }
#else
            assert(0);
#endif
        }

#ifdef SYNET_TENSORFLOW_DEBUG
        void PrintLayerAttr(const tensorflow::NodeDef & layer)
        {
            std::cout << std::endl << layer.name() << ":" << layer.op();
            for (int ii = 0; ii < layer.input_size(); ii++)
                std::cout << "(" << layer.input(ii) << ")";
            std::cout << std::endl;
            const google::protobuf::Map<String, tensorflow::AttrValue> & attrs = layer.attr();
            for (google::protobuf::Map<String, tensorflow::AttrValue>::const_iterator ai = attrs.begin(); ai != attrs.end(); ++ai)
            {
                const String & name = ai->first;
                std::cout << name << ":";
                const tensorflow::AttrValue & attr = ai->second;
                switch (attr.value_case())
                {
                case tensorflow::AttrValue::kS:
                    std::cout << attr.s();
                    break;
                case tensorflow::AttrValue::kI:
                    std::cout << attr.i();
                    break;
                case tensorflow::AttrValue::kF:
                    std::cout << attr.f();
                    break;
                case tensorflow::AttrValue::kB:
                    std::cout << attr.b();
                    break;
                case tensorflow::AttrValue::kType:
                    std::cout << attr.type();
                    break;
                case tensorflow::AttrValue::kShape:
                    PrintTensorShape(attr.shape());
                    break;
                case tensorflow::AttrValue::kTensor:
                    PrintTensor(attr.tensor());
                    break;
                case tensorflow::AttrValue::kList:
                    PrintList(attr.list());
                    break;
                default:
                    assert(0);
                    break;
                }
                std::cout << std::endl;
            }
        }

        void PrintList(const tensorflow::AttrValue::ListValue & value)
        {
            std::cout << "(";
            for (int i = 0; i < value.i_size(); i++)
                std::cout << " " << value.i(i);
            for (int i = 0; i < value.s_size(); i++)
                std::cout << " " << value.s(i);
            std::cout << " )";
        }

        void PrintTensorShape(const tensorflow::TensorShapeProto & shape)
        {
            std::cout << "[ ";
            for (int d = 0; d < shape.dim_size(); d++)
                std::cout << shape.dim(d).name() <<
                ":" << shape.dim(d).size() << " ";
            std::cout << "]";
        }

        void PrintTensor(const tensorflow::TensorProto & tensor)
        {
            if (tensor.int_val_size())
            {
                for (int i = 0; i < tensor.int_val_size(); i++)
                    std::cout << " " << tensor.int_val(i);
                return;
            }

            if (tensor.float_val_size())
            {
                for (int i = 0; i < tensor.float_val_size(); i++)
                    std::cout << " " << tensor.float_val(i);
                return;
            }

            PrintTensorShape(tensor.tensor_shape());

            switch (tensor.dtype())
            {
            case tensorflow::DT_FLOAT:
            {
                const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                size_t size = tensor.tensor_content().size() / sizeof(float);
                for (size_t i = 0; i < std::min(size_t(10), size); i++)
                    std::cout << " " << data[i];
                if (size > size_t(10))
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_INT32:
            {
                const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                size_t size = tensor.tensor_content().size() / sizeof(int);
                for (size_t i = 0; i < std::min(size_t(10), size); i++)
                    std::cout << " " << data[i];
                if (size > size_t(10))
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_BOOL:
            {
                const bool * data = reinterpret_cast<const bool*>(tensor.tensor_content().c_str());
                size_t size = tensor.tensor_content().size() / sizeof(bool);
                for (size_t i = 0; i < std::min(size_t(10), size); i++)
                    std::cout << " " << (int)data[i];
                if (size > size_t(10))
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_STRING:
            {
                Strings values;
                std::stringstream stream(tensor.tensor_content());
                while (!stream.eof())
                {
                    String value;
                    stream >> value;
                    if (value.size())
                        values.push_back(value);
                }
                size_t size = values.size();
                if (size == 0)
                {
                    std::cout << tensor.tensor_content();
                }
                else
                {
                    for (size_t i = 0; i < std::min(size_t(10), size); i++)
                        std::cout << " " << values[i];
                    if (size > size_t(10))
                        std::cout << " ... " << size - 10 << " more";
                }

                break;
            }
            default:
                assert(0);
                break;
            }
        }
#endif //SYNET_TENSORFLOW_DEBUG

        bool SaveWeight(const Tensors & weight, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (ofs.is_open())
            {
                for (size_t i = 0; i < weight.size(); ++i)
                {
                    ofs.write((const char*)weight[i].CpuData(), weight[i].Size()*sizeof(float));
                }
                ofs.close();
                return true;
            }
            return false;
        }
    };

    bool ConvertTensorflowToSynet(const String & srcParam, const String & srcGraph, const String & dstXml, const String & dstBin)
    {
        TensorflowToSynet tensorflowToSynet;
        return tensorflowToSynet.Convert(srcParam, srcGraph, dstXml, dstBin);
    }
}

#endif