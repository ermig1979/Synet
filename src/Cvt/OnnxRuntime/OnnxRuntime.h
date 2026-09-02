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
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/SynetUtils.h"
#include "Cvt/Optimizer/Optimizer.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(String srcGraphPath, bool trans, const String& dstModelPath, const String& dstWeightPath, const OnnxParam& onnxParam, const OptimizerParam& optParam);

    private:
        bool LoadModel(const String& path, onnx::ModelProto& model);

        bool ConvertModel(const onnx::ModelProto& model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Bytes& reordered);

        void SetSrcAndDst(const onnx::NodeProto& node, Renames& renames, LayerParam& layer);

        bool ManualInsertToNchwPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        bool ManualInsertToNhwcPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        //-----------------------------------------------------------------------------------------

        bool ConvertSliceNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap *tensorFormatMap)
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
                            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
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
                            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
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

        bool ConvertSplitNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
                if (!ConvertAtrributeInts(node, "split", layer.unpack().parts(), true))
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
                Shape shape = Synet::Convert(info.type().tensor_type().shape());
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
            case onnx::TensorProto_DataType_UINT8: ss << "u8"; break;
            case onnx::TensorProto_DataType_INT8: ss << "i8"; break;
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
            case onnx::TensorProto_DataType_UINT8:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << (int)((uint8_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_INT8:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << (int)((int8_t*)tensor.raw_data().c_str())[i];
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
    };

    //---------------------------------------------------------------------------------------------

    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, const OnnxParam& onnxParam, const OptimizerParam& optParam);
}

#endif