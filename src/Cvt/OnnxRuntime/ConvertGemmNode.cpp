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
    bool ConvertGemmNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered, TensorFormatMap* tensorFormatMap, UniqNames& merged)
    {
        layer.type() = Synet::LayerTypeInnerProduct;
        int transB;
        if (!ConvertAtrributeInt(node, "transB", transB))
            return false;
        layer.innerProduct().transposeB() = !transB;
        if (layer.src().size() < 2 || layer.src().size() > 3)
            return false;
        if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
            GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
        {
            layer.type() = Synet::LayerTypeQuantizedInnerProduct;
            const LayerParam* dequantized = GetLayer(layers, layer.src()[1]);
            if (dequantized->weight().empty())
                return false;
            const Shape& shape = dequantized->weight()[0].dim();
            if (!CheckDims(shape, 2, "quantized inner product weight"))
                return false;
            layer.innerProduct().outputNum() = (uint32_t)(transB ? shape[0] : shape[1]);
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (!MoveDequantizeLinearToLayer(layers, layer, merged))
                return false;
            return true;
        }
        layer.weight().resize(layer.src().size() - 1);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL || src1->type() != LayerTypeConst)
            return false;
        const Shape& weight = src1->weight()[0].dim();
        if (!CheckDims(weight, 2, "inner product weight"))
            return false;
        layer.weight()[0] = src1->weight()[0];
        layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[0] : weight[1]);
        if (layer.src().size() > 2)
        {
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = src2->weight()[0];
        }
        else
            layer.innerProduct().biasTerm() = false;
        layer.src().resize(1);
        return true;
        if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
            SYNET_ERROR("Can 't convert Gemm node for NHWC format!");
        return true;
    }
}

#endif