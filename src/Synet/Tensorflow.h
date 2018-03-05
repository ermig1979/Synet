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

            //RemoveUnused(graph);

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

        bool ConvertNetwork(const tensorflow::GraphDef & graph, Synet::NetworkParam & network, Tensors & weight)
        {
            NameIndexMap valueId;
            std::set<String> ignore;

            //AddConst(graph, ignore, valueId);

            network.name() = "tensorflow_unknown";
            for (int i = 0; i < graph.node_size(); ++i)
            {
                const ::tensorflow::NodeDef & node = graph.node(i);
                if (ignore.find(node.name()) != ignore.end())
                    continue;

                PrintLayerAttr(node);
                continue;

                String type = node.op();
                LayerParam layer;
                layer.name() = node.name();
                if (type == "Conv2D")
                {
                    layer.type() = LayerTypeConvolution;
                    layer.src().push_back(node.input(0));
                    layer.convolution().biasTerm() = false;
                    layer.weight().resize(1);


                    //const tensorflow::TensorProto & tensor = node.attr().at("value").tensor();
                    //Shape shape = GetShape(tensor);
                    //layer.weight()[0].dim() = shape;

                    NameIndexVector nextLayers = NextLayers(graph, node.name(), "BiasAdd");
                    if (nextLayers.size() == 1)
                    {
                        layer.convolution().biasTerm() = true;
                        layer.weight().resize(2);
                        //int weights_layer_index = next_layers[0].second;
                        //blobFromTensor(getConstBlob(net.node(weights_layer_index), value_id), layerParams.blobs[1]);
                        //ExcludeLayer(net, weights_layer_index, 0, false);
                        ignore.insert(nextLayers[0].first);
                    }
                }
                //for (int j = 0; j < node.input_size(); ++j)
                //    layer.src().push_back(node.input(j));
                layer.dst().push_back(node.op());

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
                if (type == "Identity" || type == "Dropout") 
                {
                    unusedIndex.push_back(i);
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

        void PrintLayerAttr(const tensorflow::NodeDef & layer)
        {
            std::cout << std::endl << layer.name() << ":" << layer.op();
            for (int ii = 0; ii < layer.input_size(); ii++)
                std::cout << "(" << layer.input(ii) << ")";
            std::cout << std::endl;
            google::protobuf::Map<std::string, tensorflow::AttrValue> attr
                = layer.attr();
            for (google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator ai = attr.begin();
                ai != attr.end(); ++ai)
            {
                std::cout << ai->first << ":";
                if (ai->first == "dtype" || ai->first == "T")
                    std::cout << ai->second.i();
                else if (ai->first == "padding")
                    std::cout << ai->second.s();
                else if (ai->first == "transpose_a" || ai->first == "transpose_b")
                    std::cout << ai->second.b();
                else if (ai->first == "shape")
                     PrintTensorShape(ai->second.shape());
                else if (ai->first == "strides" || ai->first == "ksize")
                    PrintList(ai->second.list());
                else
                    PrintTensor(ai->second.tensor());
                std::cout << std::endl;
            }
        }

        void PrintList(const tensorflow::AttrValue::ListValue & val)
        {
            std::cout << "(";
            for (int i = 0; i < val.i_size(); i++)
                std::cout << " " << val.i(i);
            std::cout << " )";
        }

        void PrintTensorShape(const tensorflow::TensorShapeProto &shape)
        {
            std::cout << "[ ";
            for (int d = 0; d < shape.dim_size(); d++)
                std::cout << shape.dim(d).name() <<
                ":" << shape.dim(d).size() << " ";
            std::cout << "]";
        }

        void PrintTensor(const tensorflow::TensorProto &tensor)
        {
            PrintTensorShape(tensor.tensor_shape());

            if (tensor.tensor_content().empty())
                return;

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