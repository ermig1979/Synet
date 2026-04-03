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
    bool ConvertQuantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 3))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src0 == NULL || src1 == NULL || src2 == NULL)
            return false;
        layer.type() = Synet::LayerTypeQuantizeLinear;
        if (!ConvertAtrributeInt(node, "axis", layer.quantize().axis(), true, 0))
            return false;
        if (src1->type() == LayerTypeConst && src2->type() == LayerTypeConst)
        {
            if (TensorSize(src1->weight()[0].dim()) == 1 && TensorSize(src2->weight()[0].dim()) == 1)
            {
                layer.quantize().scale() = GetWeight<float>(original, src1->weight()[0])[0];
                layer.quantize().type() = src2->weight()[0].type();
                switch (layer.quantize().type())
                {
                case TensorType8u:
                    layer.quantize().zero() = GetWeight<uint8_t>(original, src2->weight()[0])[0];
                    break;
                case TensorType8i:
                    layer.quantize().zero() = GetWeight<int8_t>(original, src2->weight()[0])[0];
                    break;
                default:
                    SYNET_ERROR("QuantizeLinear: unsupported src[2] type: " << Cpl::ToStr(layer.quantize().type())  << " !");
                }
            }
            else
            {
                layer.weight().push_back(src1->weight()[0]);
                layer.weight().push_back(src2->weight()[0]);
            }
            layer.src().resize(1);
        }
        else
            SYNET_ERROR("QuantizeLinear: src[1] or src[2] is not const!");
        return true;
    }
}

#endif