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

#include "Synet/Common.h"
#include "Synet/Params.h"

#if defined(SYNET_TENSORFLOW_ENABLE)

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4267 4800)
#endif

#include "tensorflow/core/public/session.h"

#ifdef _MSC_VER
#pragma warning (pop)
#endif

#define SYNET_TENSORFLOW_DEBUG

namespace Synet
{
    class TensorflowToSynet
    {
    public:
        struct SizeParam
        {
            SYNET_PARAM_VALUE(String, name, String());
            SYNET_PARAM_VALUE(int32_t, size, 0);
        };

        struct ShapeParam
        {
            SYNET_PARAM_VALUE(String, name, String());
            SYNET_PARAM_VECTOR(SizeParam, shape);
        };

        struct TensorflowParam
        {
            SYNET_PARAM_VECTOR(ShapeParam, input);
            SYNET_PARAM_VECTOR(ShapeParam, output);
        };

        SYNET_PARAM_HOLDER(TensorflowParamHolder, TensorflowParam, network);


        bool Convert(const String & srcParamPath, const String & srcGraphPath, const String & dstModelPath, const String & dstWeightPath)
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

            TensorflowParamHolder param;
            if (!param.Load(srcParamPath))
            {
                std::cout << "Can't load '" << srcParamPath << "' file!" << std::endl;
                return false;
            }

            tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), srcGraphPath, &_graph);
            if (!status.ok()) 
            {
                std::cout << "Error in building graph: " << status.error_message() << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Tensors weight;
            if (!ConvertNetwork(param(), holder(), weight))
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

        struct Pin
        {
            String name;
            int index;
            Pin(const String & name_, int index_ = 0) : name(name_), index(index_) {}
            Pin() : name(""), index(-1) {}
        };

        tensorflow::GraphDef _graph;
        NameIndexMap _valueId;
        NameSet _ignore;

        bool ConvertNetwork(const TensorflowParam & param, Synet::NetworkParam & network, Tensors & weight)
        {
            RemoveUnused();
            AddConst();

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
                    if (!ConvertInputLayer(param, node, layer))
                        return false;
                }
                else if (type == "Conv2D")
                {
                    if (!ConvertConvolutionLayer(node, layer, weight))
                        return false;
                }
                else if (type == "Relu")
                {
                    layer.type() = LayerTypeRelu;
                    layer.src().push_back(node.input(0));
                    layer.dst() = layer.src();
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
                    layer.dst() = layer.src();
                }
                else if (type == "Tanh")
                {
                    layer.type() = LayerTypeTanh;
                    layer.src().push_back(node.input(0));
                    layer.dst() = layer.src();
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
                else if (type == "MatMul")
                {
                    if (!ConvertInnerProductLayer(node, layer, weight))
                        return false;
                }
                else if (type == "Reshape")
                {
                    layer.type() = LayerTypeReshape;
                    layer.src().push_back(node.input(0));
                    bool haveConst = false;
                    for (int j = 0; !haveConst && j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        haveConst = _valueId.find(input.name) != _valueId.end();
                    }
                    if (haveConst)
                    {
                        const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                        assert(tensor.tensor_shape().dim_size() == 1);
                        layer.reshape().shape().resize(tensor.tensor_shape().dim(0).size());
                        for (size_t j = 0; j < layer.reshape().shape().size(); ++j)
                            layer.reshape().shape()[j] = ((int*)tensor.tensor_content().c_str())[j];
                    }
                    else
                    {
                        NotImplemented(layer);
                    }
                    layer.dst().push_back(layer.name());
                } 
                else if (type == "ExpandDims")
                {
                    layer.type() = LayerTypeExpandDims;
                    layer.src().push_back(node.input(0));
                    const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId);
                    layer.expandDims().axis() = tensor.int_val(0);
                    layer.dst().push_back(layer.name());
                }                
                else if (type == "Transpose")
                {
                    layer.type() = LayerTypePermute;
                    layer.src().push_back(node.input(0));
                    const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                    layer.permute().order().resize(tensor.tensor_shape().dim(0).size());
                    for (size_t j = 0; j < layer.permute().order().size(); ++j)
                        layer.permute().order()[j] = ((int*)tensor.tensor_content().c_str())[j];
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Fill")
                {
                    layer.type() = LayerTypeFill;
                    layer.src().push_back(node.input(0));
                    const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                    layer.fill().value() = tensor.float_val(0);
                    layer.dst() = layer.src();
                }
                //else if (type == "Split")
                //{
                //    //layer.type() = LayerTypeFill;

                //}
                //else if (type == "Switch")
                //{
                //    //layer.type() = LayerTypeFill;

                //}
                //else if (type == "Shape")
                //{
                //    _ignore.insert(node.name());
                //    shapes[node.name()] = node.input(0);
                //    continue;
                //}
                else
                {
                    NotImplemented(layer);
                    for (size_t j = 0; j < (size_t)node.input_size(); ++j)
                        layer.src().push_back(node.input((int)j));
                    layer.dst().push_back(type);
                }
                network.layers().push_back(layer);
            }

            return true;
        }

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

        bool ConvertInputLayer(const TensorflowParam & param, const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeInput;
            layer.dst().push_back(layer.name());
            bool found = false;
            for (size_t j = 0; j < param.input().size(); ++j)
            {
                if (param.input()[j].name() == layer.name())
                {
                    const ShapeParam & shape = param.input()[j];
                    layer.input().shape().resize(1);
                    for (size_t k = 0; k < shape.shape().size(); ++k)
                    {
                        ptrdiff_t size = shape.shape()[k].size();
                        const String & name = shape.shape()[k].name();
                        if (size > 0)
                        {
                            layer.input().shape()[0].dim().push_back(size);
                        }
                        else if (name.size() > 0 && _valueId.find(name) != _valueId.end())
                        {
                            int nodeIdx = _valueId.at(name);
                            const tensorflow::TensorProto & tensor = _graph.node(nodeIdx).attr().at("value").tensor();
                            layer.input().shape()[0].dim().push_back(tensor.int_val(0));
                        }
                        else
                            return false;
                    }
                    found = true;
                }
            }
            return found;
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
                    if(layer.input_size())
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

        void AddConst()
        {
            for (int i = 0; i < _graph.node_size(); i++)
            {
                const tensorflow::NodeDef & node = _graph.node(i);
                if (node.op() == "Const")
                {
                    _ignore.insert(node.name());
                    if (node.attr().find("value") != node.attr().end())
                        _valueId.insert(std::make_pair(node.name(), i));
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

        template <class T> void ConvertKernel(const tensorflow::TensorProto & src, Tensor & dst)
        {
            Shape shape = GetShape(src); 
            if(shape.size() == 4)
                shape = Shape({shape[3], shape[2], shape[0], shape[1]});
            dst.Reshape(shape);    
            float * pDst = dst.Data();

            const String & content = src.tensor_content();
            const T * pSrc = (T*)content.c_str();
            size_t size = content.size() / sizeof(T);
            assert(size = dst.Size());

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
                                pDst[dst_i] = (float)pSrc[src_i];
                            }
                        }
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < size; i++)
                    pDst[i] = (float)pSrc[i];
            }
        }

        void ConvertKernel(const tensorflow::TensorProto & src, Tensor & dst)
        {
            switch (src.dtype())
            {
            case tensorflow::DT_FLOAT:
                ConvertKernel<float>(src, dst);
                break;
            case tensorflow::DT_INT32:
                ConvertKernel<int>(src, dst);
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

        void NotImplemented(LayerParam & layer)
        {
            layer.dst().push_back("~~~NOT_IMPLEMENTED~~~");
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
                    ofs.write((const char*)weight[i].Data(), weight[i].Size()*sizeof(float));
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