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
    bool ConvertDequantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 3))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetWeightLayer(layers, layer.src()[1]);
        const LayerParam* src2 = GetWeightLayer(layers, layer.src()[2]);
        if (src0 == NULL || src1 == NULL || src2 == NULL)
            return false;
        layer.type() = Synet::LayerTypeDequantizeLinear;
        if (!ConvertAtrributeInt(node, "axis", layer.quantize().axis(), true, 0))
            return false;
        if (src0->type() == LayerTypeConst)
        {
            layer.weight().push_back(src0->weight()[0]);
            layer.weight().push_back(src1->weight()[0]);
            layer.weight().push_back(src2->weight()[0]);
            layer.src().resize(0);
        }
        else
        {
            layer.weight().push_back(src1->weight()[0]);
            layer.weight().push_back(src2->weight()[0]);
            layer.src().resize(1);
        }
        if (TensorSize(layer.weight().back().dim()) == 1 || layer.weight().back().scalar())
        {
            switch (layer.weight().back().type())
            {
            case TensorType8u:
                layer.quantize().zero() = GetWeight<uint8_t>(original, layer.weight().back())[0];
                break;
            case TensorType32i:
                layer.quantize().zero() = GetWeight<int32_t>(original, layer.weight().back())[0];
                break;
            case TensorType8i:
                layer.quantize().zero() = GetWeight<int8_t>(original, layer.weight().back())[0];
                break;
            default:
                SYNET_ERROR("Unsupported dequantization zero type: " << Cpl::ToStr(layer.weight().back().type()) << " !");
            }
            layer.quantize().type() = layer.weight().back().type();
            layer.weight().resize(layer.weight().size() - 1);
            if (TensorSize(layer.weight().back().dim()) == 1 || layer.weight().back().scalar())
            {
                switch (layer.weight().back().type())
                {
                case TensorType32f:
                    layer.quantize().scale() = GetWeight<float>(original, layer.weight().back())[0];
                    break;
                default:
                    SYNET_ERROR("Unsupported dequantization scale type: " << Cpl::ToStr(layer.weight().back().type()) << " !");
                }
                layer.weight().resize(layer.weight().size() - 1);
            }
        }
        return true;
    }
}

#endif