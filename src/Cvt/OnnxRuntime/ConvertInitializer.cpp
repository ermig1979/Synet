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

namespace Synet
{
    bool ConvertInitializer(const onnx::TensorProto& tensor, Synet::NetworkParam& network, Bytes& weight, Renames& renames)
    {
        LayerParam layer;
        layer.name() = ValidName(tensor.name(), renames);
        layer.dst().push_back(layer.name());
        if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
        {
            layer.type() = LayerTypeConst;
            layer.weight().resize(1);
            layer.weight()[0].type() = TensorType32f;
            uint64_t size = 1, offset = weight.size();
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                size *= (size_t)tensor.dims(i);
                layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
            }
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size * sizeof(float);
            if (size)
            {
                if (size == 1 && layer.weight()[0].dim().empty())
                {
                    layer.weight()[0].dim().push_back(1);
                    layer.weight()[0].scalar() = true;
                }
                if (tensor.has_raw_data())
                    Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                else if (tensor.float_data_size())
                {
                    if (size != tensor.float_data_size())
                        SYNET_ERROR("Wrong tensor float_data_size " << tensor.float_data_size() << " != " << size << " !");
                    for (size_t i = 0; i < size; ++i)
                        PushBack<float>(weight, tensor.float_data(i));
                }
                else
                    SYNET_ERROR("Can't parse '" << layer.name() << "' FP32 tensor!");
            }
            }
        else if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
        {
            layer.type() = LayerTypeConst;
            layer.weight().resize(1);
            layer.weight()[0].type() = TensorType32i;
            uint64_t size = 1, offset = weight.size();
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                size *= (size_t)tensor.dims(i);
                layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
            }
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size * sizeof(int32_t);
            if (size)
            {
                if (size == 1 && layer.weight()[0].dim().empty())
                {
                    layer.weight()[0].dim().push_back(1);
                    layer.weight()[0].scalar() = true;
                }
                if (tensor.has_raw_data())
                    Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                else if (tensor.int32_data_size())
                {
                    if (size != tensor.int32_data_size())
                        SYNET_ERROR("Wrong tensor int32_data_size " << tensor.int32_data_size() << " != " << size << " !");
                    for (size_t i = 0; i < size; ++i)
                        PushBack<int32_t>(weight, tensor.int32_data(i));
                }
                else
                    SYNET_ERROR("Can't parse '" << layer.name() << "' INT32 tensor!");
            }
        }
        else if (tensor.data_type() == onnx::TensorProto_DataType_UINT8)
        {
            layer.type() = LayerTypeConst;
            layer.weight().resize(1);
            layer.weight()[0].type() = TensorType8u;
            uint64_t size = 1, offset = weight.size();
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                size *= (size_t)tensor.dims(i);
                layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
            }
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size * sizeof(uint8_t);
            if (size)
            {
                if (size == 1 && layer.weight()[0].dim().empty())
                {
                    layer.weight()[0].dim().push_back(1);
                    layer.weight()[0].scalar() = true;
                }
                if (tensor.has_raw_data())
                    Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                else if (tensor.int32_data_size())
                {
                    if (size != tensor.int32_data_size())
                        SYNET_ERROR("Wrong tensor int32_data_size " << tensor.int32_data_size() << " != " << size << " !");
                    for (size_t i = 0; i < size; ++i)
                        PushBack<uint8_t>(weight, tensor.int32_data(i));
                }
                else
                    SYNET_ERROR("Can't parse '" << layer.name() << "' UINT8 tensor!");
            }
        }
        else if (tensor.data_type() == onnx::TensorProto_DataType_INT8)
        {
            layer.type() = LayerTypeConst;
            layer.weight().resize(1);
            layer.weight()[0].type() = TensorType8i;
            uint64_t size = 1, offset = weight.size();
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                size *= (size_t)tensor.dims(i);
                layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
            }
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size * sizeof(int8_t);
            if (size)
            {
                if (size == 1 && layer.weight()[0].dim().empty())
                {
                    layer.weight()[0].dim().push_back(1);
                    layer.weight()[0].scalar() = true;
                }
                if (tensor.has_raw_data())
                    Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                else if (tensor.int32_data_size())
                {
                    if (size != tensor.int32_data_size())
                        SYNET_ERROR("Wrong tensor int32_data_size " << tensor.int32_data_size() << " != " << size << " !");
                    for (size_t i = 0; i < size; ++i)
                        PushBack<int8_t>(weight, tensor.int32_data(i));
                }
                else
                    SYNET_ERROR("Can't parse '" << layer.name() << "' INT8 tensor!");
            }
        }
        else if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
        {
            ptrdiff_t size = 1;
            for (size_t i = 0; i < tensor.dims_size(); ++i)
                size *= (size_t)tensor.dims(i);
            if (size < 16)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeConst;
                layer.meta().alpha().type() = TensorType64i;
                uint64_t size = 1;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                {
                    size *= (size_t)tensor.dims(i);
                    layer.meta().alpha().shape().push_back(size_t(tensor.dims(i)));
                }
                layer.meta().alpha().i64().resize(size);
                if (size)
                {
                    if (layer.meta().alpha().shape().empty())
                    {
                        layer.meta().alpha().shape().push_back(1);
                        layer.meta().alpha().scalar() = true;
                    }
                    if (tensor.has_raw_data())
                    {
                        for (size_t i = 0; i < size; ++i)
                            layer.meta().alpha().i64()[i] = ((int64_t*)tensor.raw_data().c_str())[i];
                    }
                    else if (tensor.int64_data_size())
                    {
                        for (size_t i = 0; i < size; ++i)
                            layer.meta().alpha().i64()[i] = tensor.int64_data(i);
                    }
                    else
                        SYNET_ERROR("Can't parse '" << layer.name() << "' INT64 tensor!");
                }
            }
            else
            {
                size_t offset = weight.size();
                layer.type() = LayerTypeConst;
                layer.weight().resize(1);
                layer.weight()[0].type() = TensorType64i;
                for (size_t i = 0; i < tensor.dims_size(); ++i)
                    layer.weight()[0].dim().push_back((size_t)tensor.dims(i));
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * sizeof(int64_t);
                if (size)
                {
                    if (tensor.has_raw_data())
                        Append(weight, layer.weight()[0], tensor.raw_data().c_str());
                    else if (tensor.int64_data_size())
                    {
                        if (size != tensor.int64_data_size())
                            SYNET_ERROR("Wrong tensor int64_data_size " << tensor.int64_data_size() << " != " << size << " !");
                        for (size_t i = 0; i < size; ++i)
                            PushBack<int64_t>(weight, tensor.int64_data(i));
                    }
                    else
                        SYNET_ERROR("Can't parse '" << layer.name() << "' INT64 tensor!");
                }
            }
        }
        else
            SYNET_ERROR(" Unknown tensor type " << tensor.data_type() << " !");
        network.layers().push_back(layer);
        return true;
    }
}

#endif