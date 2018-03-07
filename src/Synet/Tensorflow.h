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
#include "tensorflow/core/public/session.h"

#define SYNET_TENSORFLOW_DEBUG

namespace Synet
{
    class TensorflowToSynet
    {
    public:
        bool Convert(const String & srcPath, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcPath))
            {
                std::cout << "File '" << srcPath << "' is not exist!" << std::endl;
                return false;
            }

            tensorflow::GraphDef graph;
            tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), srcPath, &graph);
            if (!status.ok()) 
            {
                std::cout << "Error in building graph: " << status.error_message() << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Tensors weight;
            if (!ConvertNetwork(graph, holder(), weight))
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
        typedef std::set<String> NameSet;

        struct Pin
        {
            String name;
            int index;
            Pin(const String & name_, int index_ = 0) : name(name_), index(index_) {}
            Pin() : name(""), index(-1) {}
        };

        bool ConvertNetwork(tensorflow::GraphDef & graph, Synet::NetworkParam & network, Tensors & weight)
        {
            RemoveUnused(graph);

#ifdef SYNET_TENSORFLOW_DEBUG
            for (int i = 0; i < graph.node_size(); ++i)
                PrintLayerAttr(graph.node(i));
#endif //SYNET_TENSORFLOW_DEBUG

            NameIndexMap valueId;
            std::set<String> ignore;

            AddConst(graph, ignore, valueId);

            network.name() = "tensorflow_unknown";
            for (int i = 0; i < graph.node_size(); ++i)
            {
                const ::tensorflow::NodeDef & node = graph.node(i);
                google::protobuf::Map<String, tensorflow::AttrValue> attr = node.attr();

                if (ignore.find(node.name()) != ignore.end())
                    continue;

                String type = node.op();
                LayerParam layer;
                layer.name() = node.name();
                if (type == "Placeholder")
                {
                    layer.type() = LayerTypeInput;
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Conv2D")
                {
                    layer.type() = LayerTypeConvolution;
                    layer.src().push_back(node.input(0));
                    layer.convolution().biasTerm() = false;
                    layer.weight().resize(1);
                    weight.push_back(Tensor());
                    ConvertKernel(GetConst(graph, node, valueId), weight.back());
                    layer.weight()[0].dim() = weight.back().Shape();

                    NameIndexVector nextLayers = NextLayers(graph, node.name(), "BiasAdd");
                    if (nextLayers.size() == 1)
                    {
                        layer.convolution().biasTerm() = true;
                        layer.weight().resize(2);
                        weight.push_back(Tensor());
                        const ::tensorflow::NodeDef & bias = graph.node(nextLayers[0].second);
                        ConvertKernel(GetConst(graph, bias, valueId), weight.back());
                        layer.weight()[1].dim() = weight.back().Shape();
                        ignore.insert(nextLayers[0].first);
                        ExcludeLayer(graph, nextLayers[0].second, 0, false);
                    }

                    layer.convolution().kernel() = Shape({ layer.weight()[0].dim()[2], layer.weight()[0].dim()[3]});
                    layer.convolution().outputNum() = layer.weight()[0].dim()[0];
                    if (attr.find("strides") != attr.end())
                    {
                        const tensorflow::AttrValue_ListValue & list = attr.at("strides").list();
                        assert(list.i_size() == 4 && list.i(0) == 1 && list.i(3) == 1);
                        layer.convolution().stride() = Shape({ (size_t)list.i(1), (size_t)list.i(2) });
                    }
                    if (attr.find("padding") != attr.end())
                    {
                        const String & pad = attr.at("padding").s();
                        if (pad == "SAME")
                        {
                            Shape kernel = layer.convolution().kernel();
                            layer.convolution().pad() = Shape({ kernel[0] / 2, kernel[1] / 2 });
                        }
                    }
                    layer.dst().push_back(layer.name()); 
                }
                else if (type == "Relu")
                {
                    layer.type() = LayerTypeRelu;
                    layer.src().push_back(node.input(0));
                    layer.dst() = layer.src();
                }
                else if (type == "MaxPool")
                {
                    layer.type() = LayerTypePooling;
                    layer.src().push_back(node.input(0));
                    layer.pooling().method() = PoolingMethodTypeMax;
                    const tensorflow::AttrValue_ListValue & kernel = attr.at("ksize").list();
                    assert(kernel.i_size() == 4 && kernel.i(0) == 1 && kernel.i(3) == 1);
                    layer.pooling().kernel() = Shape({ (size_t)kernel.i(1), (size_t)kernel.i(2) });
                    if (attr.find("strides") != attr.end())
                    {
                        const tensorflow::AttrValue_ListValue & list = attr.at("strides").list();
                        assert(list.i_size() == 4 && list.i(0) == 1 && list.i(3) == 1);
                        layer.pooling().stride() = Shape({ (size_t)list.i(1), (size_t)list.i(2) });
                    }
                    if (attr.find("padding") != attr.end())
                    {
                        const String & pad = attr.at("padding").s();
                        if (pad == "SAME")
                        {
                            Shape kernel = layer.convolution().kernel();
                            layer.pooling().pad() = Shape({ kernel[0] / 2, kernel[1] / 2 });
                        }
                    }
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Sigmoid")
                {
                    layer.type() = LayerTypeSigmoid;
                    layer.src().push_back(node.input(0));
                    layer.dst() = layer.src();
                }
                else if (type == "BiasAdd" || type == "Add")
                {
                    bool haveConst = false;
                    for (int j = 0; !haveConst && j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        haveConst = valueId.find(input.name) != valueId.end();
                    }
                    if (haveConst)
                    {
                        layer.src().push_back(node.input(0));
                        layer.type() = LayerTypeBias;
                        layer.weight().resize(1);
                        weight.push_back(Tensor());
                        ConvertKernel(GetConst(graph, node, valueId), weight.back());
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
                    assert(node.input_size() == 2);
                    bool haveConst = false;
                    for (int j = 0; !haveConst && j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        haveConst = valueId.find(input.name) != valueId.end();
                    }
                    layer.type() = LayerTypeInnerProduct;
                    layer.innerProduct().biasTerm() = false;
                    layer.src().push_back(node.input(0));
                    if (attr.find("transpose_a") != attr.end())
                        layer.innerProduct().transposeA() = attr.at("transpose_a").b();
                    if (attr.find("transpose_b") != attr.end())
                        layer.innerProduct().transposeB() = attr.at("transpose_b").b();
                    if (haveConst)
                    {
                        layer.weight().resize(1);
                        weight.push_back(Tensor());
                        ConvertKernel(GetConst(graph, node, valueId), weight.back());
                        layer.weight()[0].dim() = weight.back().Shape();
                        layer.innerProduct().outputNum() = layer.innerProduct().transposeB() ? layer.weight()[0].dim()[1] : layer.weight()[0].dim()[0];
                        NameIndexVector nextLayers = NextLayers(graph, node.name(), "BiasAdd");
                        if (nextLayers.size() == 1)
                        {
                            layer.innerProduct().biasTerm() = true;
                            layer.weight().resize(2);
                            weight.push_back(Tensor());
                            const ::tensorflow::NodeDef & bias = graph.node(nextLayers[0].second);
                            ConvertKernel(GetConst(graph, bias, valueId), weight.back());
                            layer.weight()[1].dim() = weight.back().Shape();
                            ignore.insert(nextLayers[0].first);
                            ExcludeLayer(graph, nextLayers[0].second, 0, false);
                        }
                    }
                    else
                    {
                        layer.src().push_back(node.input(1));
                    }
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Reshape")
                {
                    layer.type() = LayerTypeReshape;
                    layer.src().push_back(node.input(0));
                    bool haveConst = false;
                    for (int j = 0; !haveConst && j < node.input_size(); ++j)
                    {
                        Pin input = ParsePin(node.input(j));
                        haveConst = valueId.find(input.name) != valueId.end();
                    }
                    if (haveConst)
                    {
                        const tensorflow::TensorProto & tensor = GetConst(graph, node, valueId, 1);
                        assert(tensor.tensor_shape().dim_size() == 1);
                        layer.reshape().shape().resize(tensor.tensor_shape().dim(0).size());
                        for (size_t j = 0; j < layer.reshape().shape().size(); ++j)
                            layer.reshape().shape()[j] = ((int*)tensor.tensor_content().c_str())[j];
                    }
                    else
                    {

                    }
                    layer.dst().push_back(layer.name());
                }                
                else
                {
                    layer.dst().push_back(type);
                }
                network.layers().push_back(layer);
            }

            return true;
        }

        void RemoveUnused(tensorflow::GraphDef & graph)
        {
            typedef std::map<String, String>  UnusedMap;
            UnusedMap unused;
            std::vector<int> unusedIndex;

            int layersCount = graph.node_size();
            for (int i = 0; i < layersCount; i++)
            {
                const tensorflow::NodeDef & layer = graph.node(i);
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
                tensorflow::NodeDef * layer = graph.mutable_node(i);
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
                graph.mutable_node()->DeleteSubrange(startId, 1);
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

        NameIndexVector NextLayers(const tensorflow::GraphDef & graph, const String & name, const String & type = "")
        {
            NameIndexVector layers;
            for (int i = 0; i < graph.node_size(); i++)
            {
                const tensorflow::NodeDef & layer = graph.node(i);
                for (int j = 0; j < layer.input_size(); j++) 
                {
                    String input_op_name = ParsePin(layer.input(j)).name;
                    if (input_op_name == name && (type.empty() ? true : type == layer.op()))
                        layers.push_back(std::make_pair(layer.name(), i));
                }
            }
            return layers;
        }

        void AddConst(const tensorflow::GraphDef & graph, NameSet & ignore, NameIndexMap & consts)
        {
            for (int i = 0; i < graph.node_size(); i++)
            {
                const tensorflow::NodeDef & node = graph.node(i);
                if (node.op() == "Const")
                {
                    ignore.insert(node.name());
                    if (node.attr().find("value") != node.attr().end())
                        consts.insert(std::make_pair(node.name(), i));
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
                                pDst[dst_i] = pSrc[src_i];
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

        void ExcludeLayer(tensorflow::GraphDef & graph, const int layerIndex, const int inputIndex, bool remove = true) 
        {
            String layerName = graph.node(layerIndex).name();
            NameIndexVector layers = NextLayers(graph, layerName);
            String removedInput = graph.node(layerIndex).input(inputIndex);
            for (size_t i = 0; i < layers.size(); i++)
            {
                tensorflow::NodeDef * layer = graph.mutable_node(layers[i].second);
                for (int input_id = 0; input_id < layer->input_size(); input_id++) 
                {
                    String inputName = layer->input(input_id);
                    if (inputName == layerName)
                        layer->set_input(input_id, removedInput);
                }
            }
            if (remove)
                graph.mutable_node()->DeleteSubrange(layerIndex, 1);
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
                int size = tensor.tensor_content().size() / sizeof(float);
                for (int i = 0; i < std::min(10, size); i++)
                    std::cout << " " << data[i];
                if (size > 10)
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_INT32:
            {
                const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                int size = tensor.tensor_content().size() / sizeof(int);
                for (int i = 0; i < std::min(10, size); i++)
                    std::cout << " " << data[i];
                if (size > 10)
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

    bool ConvertTensorflowToSynet(const String & src, const String & dstXml, const String & dstBin)
    {
        TensorflowToSynet tensorflowToSynet;
        return tensorflowToSynet.Convert(src, dstXml, dstBin);
    }
}

#endif