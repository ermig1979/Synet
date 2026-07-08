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

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    bool ConvertConstantOfShapeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& original, Bytes& reordered)
    {
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        if (src0 == NULL)// || src0->type() != Synet::LayerTypeMeta)
            return false;
        Shape shape;
        if (IsMetaConst64i(*src0))
            shape = Shp(src0->meta().alpha().i64());
        const onnx::AttributeProto* attribute = GetAtrribute(node, "value");
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
}

#endif