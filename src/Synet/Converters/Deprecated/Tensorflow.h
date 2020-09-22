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

#include "Synet/Converters/Optimizer.h"

#if defined(SYNET_TENSORFLOW_ENABLE)

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4267 4800 4554 4244)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "tensorflow/core/public/session.h"

#if defined(_MSC_VER)
#pragma warning (pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

//#define SYNET_TENSORFLOW_DEBUG
//#define SYNET_TENSORFLOW_DYNAMIC

namespace Synet
{
    class TensorflowToSynet
    {
    public:

        struct ShapeTipParam
        {
            SYNET_PARAM_VALUE(String, name, String());
            SYNET_PARAM_VALUE(Shape, shape, Shape());
        };

        struct ConvertParam
        {
            SYNET_PARAM_VALUE(Strings, ignore, Strings());
            SYNET_PARAM_VALUE(Strings, output, Strings());
            SYNET_PARAM_VECTOR(ShapeTipParam, tips);
        };

        SYNET_PARAM_HOLDER(ConvertParamHolder, ConvertParam, convert);

        bool Convert(const String & srcParamPath, const String & srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
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

            if (!_param.Load(srcParamPath))
            {
                std::cout << "Can't load converion partameters!" << std::endl;
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

            _trans = trans;

            Synet::NetworkParamHolder holder;
            Vector weight;
            if (!ConvertNetwork(holder(), weight))
                return false;

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
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
        typedef std::vector<float> Vector;
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

        ConvertParamHolder _param;
        tensorflow::GraphDef _graph;
        bool _trans;
        NameIndexMap _valueId;
        NameSet _ignore, _meta, _fMeta;

        IConstMap _iConst;
        FConstMap _fConst;
        ShapeMap _shape;

        bool ConvertNetwork(Synet::NetworkParam & network, Vector & weight)
        {
            RemoveUnused();

            AddConstFloat();

#ifdef SYNET_TENSORFLOW_DEBUG
            for (int i = 0; i < _graph.node_size(); ++i)
                PrintLayerAttr(_graph.node(i));
#endif //SYNET_TENSORFLOW_DEBUG

            NameNameMap shapes;
            network.version() = 1;
            network.name() = "tensorflow_unknown";
            for (size_t i = 0, offset = 0; i < (size_t)_graph.node_size(); ++i)
            {
                const ::tensorflow::NodeDef & node = _graph.node((int)i);
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
                else if (IsMeta(node))
                {
                    if (!ConvertMetaConstFloatLayer(node, network))
                        return false;
                    if (!ConvertMetaLayer(node, layer))
                        return false;
                }
                else if (type == "Add" || type == "BiasAdd")
                {
                    if (!ConvertAddLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "Cast")
                {
                    if (!ConvertCastLayer(node, layer))
                        return false;
                }
                else if (type == "Conv2D" || type == "DepthwiseConv2dNative")
                {
                    if (!ConvertConvolutionLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "ExpandDims")
                {
                    if (!ConvertExpandDimsLayer(node, layer))
                        return false;
                }
                else if (type == "Fill")
                {
                    layer.type() = LayerTypeFill;
                    layer.src().push_back(node.input(0));
                    const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId, 1);
                    layer.fill().value() = tensor.float_val(0);
                    layer.dst() = layer.src();
                }
                else if (type == "Gather")
                {
                    layer.type() = LayerTypeGather;
                    layer.src().push_back(node.input(0));
                    layer.src().push_back(node.input(1));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "MaxPool" || type == "AvgPool")
                {
                    if (!ConvertPoolingLayer(node, layer))
                        return false;
                }
                else if (type == "MatMul")
                {
                    if (!ConvertInnerProductLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "Maximum" || type == "Minimum")
                {
                    if (!ConvertEltwiseLayer(node, layer))
                        return false;
                }
                else if (type == "Mul")
                {
                    if (!ConvertMulLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "Pad")
                {
                    assert(node.input_size() == 2);
                    layer.type() = LayerTypePad;
                    layer.src().push_back(node.input(0));
                    layer.src().push_back(node.input(1));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "RealDiv")
                {
                    if (!ConvertRealDivLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "Max" || type == "Sum")
                {
                    if (!ConvertReductionLayer(node, layer, network))
                        return false;
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
                else if (type == "Sigmoid")
                {
                    layer.type() = LayerTypeSigmoid;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Softmax")
                {
                    layer.type() = LayerTypeSoftmax;
                    layer.softmax().axis() = _trans ? 0 : 1;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Split")
                {
                    if (!ConvertSplitLayer(node, layer, network))
                        return false;
                }                
                else if (type == "Squeeze")
                {
                    layer.type() = LayerTypeSqueeze;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if (type == "Sub")
                {
                    if (!ConvertSubLayer(node, layer, weight, offset))
                        return false;
                }
                else if (type == "Switch")
                {
                    layer.type() = LayerTypeSwitch;
                    layer.src().push_back(node.input(0));
                    layer.src().push_back(node.input(1));
                    layer.dst().push_back(layer.name());
                    layer.dst().push_back(layer.name() + ":1");
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
                else if (type == "Unpack")
                {
                    if (!ConvertUnpackLayer(node, layer))
                        return false;
                }
                else if (type == "Abs" || type == "Exp" || type == "Neg" || type == "Rsqrt" || type == "Sqrt" || type == "Tanh" || type == "ZerosLike")
                {
                    if (!ConvertUnaryOperationLayer(node, layer))
                        return false;
                }
                else if (type == "NextIteration" || type == "TensorArrayScatterV3" || type == "Identity" || type == "Enter" || type == "LoopCond" || type == "Exit")
                {
                    layer.type() = LayerTypeStub;
                    layer.src().push_back(node.input(0));
                    layer.dst().push_back(layer.name());
                }
                else if(type == "Merge")
                {
                    layer.type() = LayerTypeStub;
                    for(int j = 0; j < node.input_size(); ++j)
                        layer.src().push_back(node.input(j));
                    layer.dst().push_back(layer.name());
                    layer.dst().push_back(layer.name() + ":1");
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

        void AddSrcDst(const ::tensorflow::NodeDef & node, int src, int dst, Synet::LayerParam & layer)
        {
            for(int i = 0; i < src; ++i)
                layer.src().push_back(node.input(i));
            if(dst)
                layer.dst().push_back(layer.name());
            for (int i = 1; i < dst; ++i)
                layer.dst().push_back(layer.name() + ":" + ValueToString(i));
        }

        bool ConvertAddLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
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
                if (!ConvertWeight(GetConst(_graph, node, _valueId), String(), layer.weight()[0], weight, offset))
                    return false;
                layer.dst() = layer.src();
            }
            else
            {
                if (!ConvertEltwiseLayer(node, layer))
                    return false;
            }
            return true;
        }

        bool ConvertCastLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeCast;
            const tensorflow::AttrValue & attr = node.attr().at("DstT");
            if (attr.type() == tensorflow::DT_FLOAT)
                layer.cast().type() = TensorType32f;
            else if (attr.type() == tensorflow::DT_INT32)
                layer.cast().type() = TensorType32i;
            else
                assert(0);
            AddSrcDst(node, 1, 1, layer);
            return true;
        }

        bool ConvertConvolutionLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
        {
            layer.type() = LayerTypeConvolution;
            layer.convolution().biasTerm() = false;
            layer.weight().resize(1);
            if (!ConvertWeight(GetConst(_graph, node, _valueId), String(), layer.weight()[0], weight, offset))
                return false;
            NameIndexVector nextLayers = NextLayers(node.name(), "BiasAdd");
            if (nextLayers.size() == 1)
            {
                layer.convolution().biasTerm() = true;
                layer.weight().resize(2);
                const ::tensorflow::NodeDef & bias = _graph.node(nextLayers[0].second);
                if (!ConvertWeight(GetConst(_graph, bias, _valueId), String(), layer.weight()[1], weight, offset))
                    return false;
                _ignore.insert(nextLayers[0].first);
                ExcludeLayer(nextLayers[0].second, 0, false);
                for (size_t j = 0; j < _param().output().size(); ++j)
                {
                    if (_param().output()[j] == nextLayers[0].first)
                    {
                        layer.name() = nextLayers[0].first;
                        break;
                    }
                }
            }

            const Shape & shape = layer.weight()[0].dim();
            layer.convolution().kernel() = _trans ? Shape({ shape[2], shape[3] }) : Shape({ shape[0], shape[1] });
            if (node.op() == "Conv2D")
                layer.convolution().outputNum() = _trans ? (uint32_t)shape[0] : (uint32_t)shape[3];
            else if (node.op() == "DepthwiseConv2dNative")
            {
                layer.convolution().outputNum() = _trans ? (uint32_t)shape[1] : (uint32_t)shape[2];
                layer.convolution().group() = _trans ? (uint32_t)shape[1] : (uint32_t)shape[2];
            }
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
            AddSrcDst(node, 1, 1, layer);
            return true;
        }

        bool ConvertEltwiseLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeEltwise;
            for (int j = 0; j < node.input_size(); ++j)
            {
                Pin input = ParsePin(node.input(j));
                assert(_valueId.find(input.name) == _valueId.end());
            }
            if (node.op() == "BiasAdd" || node.op() == "Add")
                layer.eltwise().operation() = EltwiseOperationTypeSum;
            else if (node.op() == "Maximum")
                layer.eltwise().operation() = EltwiseOperationTypeMax;
            else if (node.op() == "Minimum")
                layer.eltwise().operation() = EltwiseOperationTypeMin;
            else if(node.op() == "Mul")
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
            else if (node.op() == "Sub")
            {
                layer.eltwise().operation() = EltwiseOperationTypeSum;
                layer.eltwise().coefficients() = Floats({ 1.0f, -1.0f });
            }
            else
                assert(0);
            AddSrcDst(node, 2, 1, layer);
            return true;
        }

        bool ConvertExpandDimsLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeExpandDims;
            if (node.attr().find("Tdim") != node.attr().end())
                layer.expandDims().axis() = (int)node.attr().at("Tdim").i();
            else
            {
                const tensorflow::TensorProto & tensor = GetConst(_graph, node, _valueId);
                layer.expandDims().axis() = tensor.int_val(0);
            }
            AddSrcDst(node, 1, 1, layer);
            return true;
        }

        bool ReorderWeightSpecial(const String & name, Tensor & weight)
        {
            assert(weight.Count() == 2);
            Shape shape;
            for (size_t j = 0; j < _param().tips().size(); ++j)
            {
                if (_param().tips()[j].name() == name)
                {
                    shape = _param().tips()[j].shape();
                    break;
                }
            } 
            if (shape.empty())
                return true;
            assert(shape.size() == 2 && shape[0] * shape[1] == weight.Axis(1));
            Tensor buf(shape);
            for (size_t j = 0; j < weight.Axis(0); ++j)
            {
                float * pSrc = weight.CpuData({j, size_t(0)});
                float * pDst = buf.CpuData();
                size_t on = shape[1], in = shape[0];
                for (size_t o = 0; o < on; o++)
                    for (size_t i = 0; i < in; i++)
                        pDst[in*o + i] = pSrc[on*i + o];
                memcpy(pSrc, pDst, on*in * sizeof(float));
            }
            return true;
        }

        bool ConvertInnerProductLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
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
            if (node.attr().find("transpose_a") != node.attr().end())
                layer.innerProduct().transposeA() = node.attr().at("transpose_a").b();
            if (node.attr().find("transpose_b") != node.attr().end())
                layer.innerProduct().transposeB() = node.attr().at("transpose_b").b();
            //layer.innerProduct().axis() = 0;
            if (haveConst)
            {
                layer.weight().resize(1);
                if (!ConvertWeight(GetConst(_graph, node, _valueId), layer.name(), layer.weight()[0], weight, offset))
                    return false;
                const Shape & shape = layer.weight()[0].dim();
                layer.innerProduct().outputNum() = (uint32_t)(/*(!_trans) ^ */layer.innerProduct().transposeB() ? shape[1] : shape[0]);
                NameIndexVector nextLayers = NextLayers(node.name(), "BiasAdd");
                if (nextLayers.size() == 1)
                {
                    layer.innerProduct().biasTerm() = true;
                    layer.weight().resize(2);
                    const ::tensorflow::NodeDef & bias = _graph.node(nextLayers[0].second);
                    if (!ConvertWeight(GetConst(_graph, bias, _valueId), String(), layer.weight()[1], weight, offset))
                        return false;
                    _ignore.insert(nextLayers[0].first);
                    ExcludeLayer(nextLayers[0].second, 0, false);
                    for (size_t j = 0; j < _param().output().size(); ++j)
                    {
                        if (_param().output()[j] == nextLayers[0].first)
                        {
                            layer.name() = nextLayers[0].first;
                            break;
                        }
                    }
                }
                AddSrcDst(node, 1, 1, layer);
            }
            else
                AddSrcDst(node, 2, 1, layer);
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
                if(dst.size())
                    dst[0] = 1;
                if (_trans && dst.size() == 4)
                    dst = Shape({ dst[0], dst[3], dst[1], dst[2] });
                layer.input().shape()[0].format() = _trans ? TensorFormatNchw : TensorFormatNhwc;
            }
            return true;
        }

        bool ConvertMetaConstFloatLayer(const ::tensorflow::NodeDef & node, Synet::NetworkParam & network)
        {
            for (int j = 0; j < node.input_size(); ++j)
            {
                String name = node.input(j);
                Pin input = ParsePin(name);
                if (_valueId.find(input.name) != _valueId.end() && _fMeta.find(name) == _fMeta.end())
                {
                    Synet::LayerParam layer;
                    layer.name() = name;
                    layer.type() = LayerTypeMeta;
                    layer.meta().type() = MetaTypeConst;
                    Tensor alpha;
                    ConvertKernel(GetConst(_graph, node, _valueId), alpha);
                    alpha.Export(layer.meta().alpha());
                    network.layers().push_back(layer);
                    _fMeta.insert(name);
                }
            }
            return true;
        }

        bool ConvertMetaLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeMeta;
            String type = node.op();
            if (type == "Add")
            {
                layer.meta().type() = MetaTypeAdd;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Cast")
            {
                layer.meta().type() = MetaTypeCast;
                const tensorflow::AttrValue & attr = node.attr().at("DstT");
                if (attr.type() == tensorflow::DT_FLOAT)
                    layer.meta().alpha().type() = TensorType32f;
                else if (attr.type() == tensorflow::DT_INT32)
                    layer.meta().alpha().type() = TensorType32i;
                else
                    assert(0);
                AddSrcDst(node, 1, 1, layer);
            }
            else if (type == "Concat" || type == "ConcatV2")
            {
                layer.meta().type() = MetaTypePack;
                int axisId = (type == "Concat" ? 0 : node.input_size() - 1);
                for (int j = 0; j < node.input_size(); ++j)
                {
                    if (j != axisId)
                        layer.src().push_back(node.input(j));
                }
                AddSrcDst(node, 0, 1, layer);
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
                dst.Export(layer.meta().alpha());
                AddSrcDst(node, 0, 1, layer);
            }
            else if (type == "ExpandDims")
            {
                layer.meta().type() = MetaTypeExpandDims;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Fill")
            {
                layer.meta().type() = MetaTypeFill;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Gather")
            {
                layer.meta().type() = MetaTypeGather;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Greater")
            {
                layer.meta().type() = MetaTypeGreater;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Maximum")
            {
                layer.meta().type() = MetaTypeMaximum;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Minimum")
            {
                layer.meta().type() = MetaTypeMinimum;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Mul")
            {
                layer.meta().type() = MetaTypeMul;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Pack")
            {
                layer.meta().type() = MetaTypePack;
                AddSrcDst(node, node.input_size(), 1, layer);
            }
            else if (type == "PlaceholderWithDefault")
            {
                layer.meta().type() = MetaTypeInputWithDefault;
                AddSrcDst(node, 1, 1, layer);
            }
            else if (type == "Range")
            {
                layer.meta().type() = MetaTypeRange;
                AddSrcDst(node, 3, 1, layer);
            }
            else if (type == "RealDiv")
            {
                layer.meta().type() = MetaTypeRealDiv;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Reshape")
            {
                layer.meta().type() = MetaTypeReshape;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Rsqrt")
            {
                layer.meta().type() = MetaTypeRsqrt;
                AddSrcDst(node, 1, 1, layer);
            }
            else if (type == "Shape")
            {
                layer.meta().type() = MetaTypeShape;
                AddSrcDst(node, 1, 1, layer);
            }
            else if (type == "Slice")
            {
                layer.meta().type() = MetaTypeSlice;
                AddSrcDst(node, 3, 1, layer);
            }
            else if (type == "Sqrt")
            {
                layer.meta().type() = MetaTypeSqrt;
                AddSrcDst(node, 1, 1, layer);
            }
            else if (type == "StridedSlice")
            {
                layer.meta().type() = MetaTypeStridedSlice;
                AddSrcDst(node, 3, 1, layer);
            }
            else if (type == "Sub")
            {
                layer.meta().type() = MetaTypeSub;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Switch")
            {
                layer.meta().type() = MetaTypeSwitch;
                AddSrcDst(node, 2, 2, layer);
            }
            else if (type == "TensorArrayV3")
            {
                layer.meta().type() = MetaTypeTensorArray;
                tensorflow::AttrValue attr = node.attr().at("dtype");
                tensorflow::DataType dtype = attr.type();
                if (dtype == tensorflow::DT_FLOAT)
                    layer.meta().alpha().type() = TensorType32f;
                else if (dtype == tensorflow::DT_INT32)
                    layer.meta().alpha().type() = TensorType32i;
                else
                    assert(0);
                AddSrcDst(node, 1, 2, layer);
            }
            else if (type == "TensorArrayReadV3")
            {
                layer.meta().type() = MetaTypeTensorArrayRead;
                AddSrcDst(node, 3, 1, layer);
            }
            else if (type == "TensorArraySizeV3")
            {
                layer.meta().type() = MetaTypeTensorArraySize;
                AddSrcDst(node, 2, 2, layer);
            }
            else if (type == "TensorArrayWriteV3")
            {
                layer.meta().type() = MetaTypeTensorArrayWrite;
                AddSrcDst(node, 4, 1, layer);
            }
            else if (type == "Tile")
            {
                layer.meta().type() = MetaTypeTile;
                AddSrcDst(node, 2, 1, layer);
            }
            else if (type == "Unpack")
            {
                layer.meta().type() = MetaTypeUnpack;
                int axis = (int)node.attr().at("axis").i();
                layer.meta().alpha().type() = TensorType32i;
                layer.meta().alpha().shape().resize(1, 1);
                layer.meta().alpha().i32().push_back(axis);
                int num = (int)node.attr().at("num").i();
                AddSrcDst(node, 1, num, layer);
            }
            else if (type == "Enter" || type == "Squeeze")
            {
                layer.meta().type() = MetaTypeStub;
                AddSrcDst(node, 1, 1, layer);
            }
            else
            {
                SetNotImplemented(layer, node);
            }
            _meta.insert(node.name());
            return true;
        }

        bool ConvertMulLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
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
                if (!ConvertEltwiseLayer(node, layer))
                    return false;
            }
            else
            {
                layer.type() = LayerTypeScale;
                layer.src().push_back(node.input(1 - (int)constIndex));
                layer.scale().biasTerm() = false;
                layer.weight().resize(1);
                if (!ConvertWeight(GetConst(_graph, node, _valueId), String(), layer.weight()[0], weight, offset))
                    return false;
                if (layer.weight()[0].dim()[0] == 1)
                    layer.scale().axis() = 0;
                layer.dst().push_back(layer.name());
            }
            return true;
        }

        bool ConvertPoolingLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypePooling;
            if (node.op() == "MaxPool")
                layer.pooling().method() = PoolingMethodTypeMax;
            else if (node.op() == "AvgPool")
                layer.pooling().method() = PoolingMethodTypeAverage;
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
                    layer.pooling().padType() = PoolingPadTypeTensorflowSame;
            }
            layer.pooling().yoloCompatible() = layer.pooling().kernel()[0] == 2;
            AddSrcDst(node, 1, 1, layer);
            return true;
        }

        bool ConvertRealDivLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
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
            if (constCount == 1 && constIndex == 1)
            {
                layer.type() = LayerTypeScale;
                layer.scale().biasTerm() = false;
                layer.scale().axis() = 0;
                layer.weight().resize(1);
                Tensor divider;
                ConvertKernel(GetConst(_graph, node, _valueId), divider);
                assert(divider.Size() == 1);
                divider.CpuData()[0] = 1.0f / divider.CpuData()[0];
                UpdateWeight(divider, layer.weight()[0], weight, offset);
                AddSrcDst(node, 1, 1, layer);
            }
            else
            {
                layer.type() = LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeDiv;
                AddSrcDst(node, 2, 1, layer);
            }
            return true;
        }

        bool ConvertReductionLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Synet::NetworkParam & network)
        {
            assert(node.input_size() == 2);
            String name = ParsePin(node.input(1)).name;
            ptrdiff_t index = -1;
            for (size_t i = 0; i < network.layers().size(); ++i)
            {
                if (network.layers()[i].name() == name)
                {
                    index = i;
                    break;
                }
            }
            assert(index >= 0);
            layer.type() = LayerTypeReduction;
            AddSrcDst(node, 1, 1, layer);
            Ints axis = network.layers()[index].meta().alpha().i32();
            if (_trans && axis.size() == 1)
            {
                //if (axis[0] == 0)
                //    axis[0] = 1;
                //if (axis[0] == 1)
                //    axis[0] = 0;
                if (axis[0] == 3)
                    axis[0] = 1;
            }
            layer.reduction().axis() = axis;
            network.layers().erase(network.layers().begin() + index);
            if (node.attr().find("keep_dims") != node.attr().end())
                layer.reduction().keepDims() = node.attr().at("keep_dims").b();
            if (node.op() == "Max")
                layer.reduction().type() = ReductionTypeMax;
            else if (node.op() == "Sum")
                layer.reduction().type() = ReductionTypeSum;
            else
                assert(0);
            return true;
        }

        bool ConvertSplitLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Synet::NetworkParam & network)
        {
            assert(node.input_size() == 2);
            String name = ParsePin(node.input(0)).name;
            ptrdiff_t index = -1;
            for (size_t i = 0; i < network.layers().size(); ++i)
            {
                if (network.layers()[i].name() == name)
                {
                    index = i; 
                    break;
                }
            }
            assert(index >= 0);
            TensorParam axis = network.layers()[index].meta().alpha();
            layer.type() = LayerTypeSlice;
            layer.src().push_back(node.input(1));
            layer.slice().axis() = axis.i32()[0];
            network.layers().erase(network.layers().begin() + index);
            if (node.attr().find("num_split") != node.attr().end())
            {
                int num = (int)node.attr().at("num_split").i();
                AddSrcDst(node, 0, num, layer);
            }
            else
                return false;
            return true;
        }

        bool ConvertSubLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer, Vector & weight, size_t & offset)
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
                layer.type() = LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeSub;
                AddSrcDst(node, 2, 1, layer);
                //if (!ConvertEltwiseLayer(node, layer))
                //    return false;
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
                UpdateWeight(scale, layer.weight()[0], weight, offset);
                UpdateWeight(bias, layer.weight()[1], weight, offset);
                layer.dst() = layer.src();//?
            }
            return true;
        }

        bool ConvertUnpackLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeUnpack;
            layer.unpack().axis() = (int)node.attr().at("axis").i();
            int num = (int)node.attr().at("num").i();
            AddSrcDst(node, 1, num, layer);
            return true;
        }

        bool ConvertUnaryOperationLayer(const ::tensorflow::NodeDef & node, Synet::LayerParam & layer)
        {
            layer.type() = LayerTypeUnaryOperation;
            if (node.op() == "Abs")
                layer.unaryOperation().type() = UnaryOperationTypeAbs;
            else if (node.op() == "Exp")
                layer.unaryOperation().type() = UnaryOperationTypeExp;
            else if (node.op() == "Neg")
                layer.unaryOperation().type() = UnaryOperationTypeNeg;
            else if (node.op() == "Rsqrt")
                layer.unaryOperation().type() = UnaryOperationTypeRsqrt;
            else if (node.op() == "Sqrt")
                layer.unaryOperation().type() = UnaryOperationTypeSqrt;
            else if (node.op() == "Tanh")
                layer.unaryOperation().type() = UnaryOperationTypeTanh;
            else if (node.op() == "ZerosLike")
                layer.unaryOperation().type() = UnaryOperationTypeZero;
            else
                return false;
            AddSrcDst(node, 1, 1, layer);
            return true;
        }

        //---------------------------------------------------------------------

        bool Output(const String & name)
        {
            for (size_t i = 0; i < _param().output().size(); ++i)
            {
                if (name == _param().output()[i])
                    return true;
            }
            return false;
        }

        bool Ignore(const String & name)
        {
            for (size_t i = 0; i < _param().ignore().size(); ++i)
            {
                if (name.find(_param().ignore()[i]) == 0)
                    return true;
            }
            return false;
        }

        void GetUsed(NameSet & used)
        {
            typedef std::map<String, int> Index;
            Index index;
            for (int i = 0; i < _graph.node_size(); i++)
                index[_graph.node(i).name()] = i;

            typedef std::queue<String> Queue;
            Queue queue;
            used.clear();
            for (size_t i = 0; i < _param().output().size(); ++i)
            {
                const String & output = _param().output()[i];
                used.insert(output);
                queue.push(output);
            }

            while (queue.size())
            {
                const tensorflow::NodeDef & node = _graph.node(index[queue.front()]);
                for (int i = 0; i < node.input_size(); ++i)
                {
                    String name = node.input(i);
                    size_t delimiter = name.find_first_of(":");
                    if (delimiter != std::string::npos)
                        name = name.substr(0, delimiter);
                    if (used.find(name) == used.end())
                    {
                        used.insert(name);
                        queue.push(name);
                    }
                }
                queue.pop();
            }
        }

        void RemoveUnused()
        {
            typedef std::map<String, String>  UnusedMap;
            UnusedMap unused;
            std::vector<int> unusedIndex;

            NameSet used;
            GetUsed(used);

            int layersCount = _graph.node_size();
            for (int i = 0; i < layersCount; i++)
            {
                const tensorflow::NodeDef & layer = _graph.node(i);
                String type = layer.op();
                String name = layer.name();
                if (Output(name))
                    continue;
                if(type == "Identity" || type == "Dropout" || type == "Assert" || type == "NextIteration" ||
                    used.find(name) == used.end() || (Ignore(name) && type != "Const"))
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

        bool IsMeta(const tensorflow::NodeDef & node)
        {
            if (node.op() == "Shape" || node.op() == "Const" || node.op() == "TensorArraySizeV3" || node.op() == "TensorArrayReadV3" || node.op() == "TensorArrayV3")
                return true;

            int meta = 0, fConst = 0;
            for (int j = 0; j < node.input_size(); ++j)
            {
                String name = node.input(j);
                if (name.find(":") != String::npos)
                    name = name.substr(0, name.find(":"));
                if (_meta.find(name) != _meta.end())
                    meta++;
                if (_fConst.find(name) != _fConst.end())
                    fConst++;
            }
            return meta + fConst == node.input_size();
        }

        template <class TS, class TD> void ConvertKernel(const tensorflow::TensorProto & src, Synet::Tensor<TD> & dst)
        {
            Shape shape = GetShape(src); 
            if(_trans && shape.size() == 4)
                shape = Shape({shape[3], shape[2], shape[0], shape[1]});
            if (/*_trans &&*/ shape.size() == 2)
                shape = Shape({ shape[1], shape[0] });
            dst.Reshape(shape);    
            TD * pDst = dst.CpuData();

            const String & content = src.tensor_content();
            const TS * pSrc = (TS*)content.c_str();

            if (_trans && shape.size() == 4)
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
            else if (/*_trans &&*/ shape.size() == 2)
            {
                size_t out_c = shape[0], in_c = shape[1];
                for (size_t i_oc = 0; i_oc < out_c; i_oc++)
                {
                    for (size_t i_ic = 0; i_ic < in_c; i_ic++)
                    {
                        size_t dst_i = in_c*i_oc + i_ic;
                        size_t src_i = out_c*i_ic + i_oc;
                        pDst[dst_i] = (TD)pSrc[src_i];                        
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

            dst.SetFormat((!_trans && shape.size() == 4) ? TensorFormatNhwc : TensorFormatNchw);
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
                for (size_t i = 0; i < Min(size_t(10), size); i++)
                    std::cout << " " << data[i];
                if (size > size_t(10))
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_INT32:
            {
                const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                size_t size = tensor.tensor_content().size() / sizeof(int);
                for (size_t i = 0; i < Min(size_t(10), size); i++)
                    std::cout << " " << data[i];
                if (size > size_t(10))
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
            case tensorflow::DT_BOOL:
            {
                const bool * data = reinterpret_cast<const bool*>(tensor.tensor_content().c_str());
                size_t size = tensor.tensor_content().size() / sizeof(bool);
                for (size_t i = 0; i < Min(size_t(10), size); i++)
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
                    for (size_t i = 0; i < Min(size_t(10), size); i++)
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

        bool UpdateWeight(const Tensor & src, WeightParam & param, Vector & weight, size_t & offset)
        {
            if (offset + src.Size() > weight.size())
                weight.resize(offset + src.Size());
            memcpy(weight.data() + offset, src.CpuData(), src.Size()*sizeof(float));
            param.dim() = src.Shape();
            param.format() = src.Format();
            param.offset() = offset * sizeof(float);
            param.size() = src.Size() * sizeof(float);
            offset += src.Size();
            return true;
        }

        bool ConvertWeight(const tensorflow::TensorProto & src, const String & name, WeightParam & param, Vector & weight, size_t & offset)
        {
            Tensor dst;
            ConvertKernel(src, dst);
            if (name.size() && _trans && !ReorderWeightSpecial(name, dst))
                return false;
            return UpdateWeight(dst, param, weight, offset);
        }

        bool SaveWeight(const Vector & bin, const String & path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (!ofs.is_open())
                return false;
            ofs.write((const char*)bin.data(), bin.size() * sizeof(float));
            bool result = (bool)ofs;
            ofs.close();
            return result;
        }
    };

    bool ConvertTensorflowToSynet(const String & srcParam, const String & srcGraph, bool trans, const String & dstXml, const String & dstBin)
    {
        TensorflowToSynet tensorflowToSynet;
        return tensorflowToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif