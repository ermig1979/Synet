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

#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
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
            if (size == 0)
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
        const onnx::AttributeProto* value = GetAtrribute(node, name);
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
}

#endif