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
    bool ConvertCastNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 1))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        if (src0 == NULL)
            return false;
        int to;
        if (!ConvertAtrributeInt(node, "to", to))
            return false;
        if (src0->type() == LayerTypeMeta)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeCast;
            if (to == onnx::TensorProto_DataType_FLOAT)
                layer.meta().alpha().type() = TensorType32f;
            else if (to == onnx::TensorProto_DataType_INT32)
                layer.meta().alpha().type() = TensorType32i;
            else if (to == onnx::TensorProto_DataType_INT64)
                layer.meta().alpha().type() = TensorType64i;
            else
                SYNET_ERROR("Unsupported cast type!");
        }
        else
        {
            layer.type() = Synet::LayerTypeCast;
            if (to == onnx::TensorProto_DataType_FLOAT)
                layer.cast().type() = TensorType32f;
            else if (to == onnx::TensorProto_DataType_INT32)
                layer.cast().type() = TensorType32i;
            else if (to == onnx::TensorProto_DataType_INT64)
            {
                layer.cast().type() = TensorType64i;
                for (size_t i = 0; i < onnxParam.cast64iTo32i().size(); ++i)
                {
                    if (layer.name() == onnxParam.cast64iTo32i()[i])
                    {
                        layer.cast().type() = TensorType32i;
                        break;
                    }
                }
            }
            else if (to == onnx::TensorProto_DataType_UINT8)
                layer.cast().type() = TensorType8u;
            else
                SYNET_ERROR("Unsupported cast type!");
            if (src0->type() == LayerTypeConst && src0->weight().size() && src0->weight()[0].type() == layer.cast().type())
            {
                layer.type() = Synet::LayerTypeStub;
                layer.cast().type() = TensorTypeUnknown;
                }
            if (src0->type() == LayerTypeQuantizeLinear && src0->quantize().type() == layer.cast().type())
            {
                layer.type() = Synet::LayerTypeStub;
                layer.cast().type() = TensorTypeUnknown;
            }
            }
        return true;
        }
}

#endif