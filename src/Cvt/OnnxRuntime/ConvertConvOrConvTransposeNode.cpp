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
    bool ConvertConvOrConvTransposeNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap)
    {
        if (node.op_type() == "Conv")
            layer.type() = Synet::LayerTypeConvolution;
        else if (node.op_type() == "ConvTranspose")
            layer.type() = Synet::LayerTypeDeconvolution;
        else
            return false;
        if (layer.src().size() < 2 || layer.src().size() > 3)
            return false;
        if (!ConvertAtrributeInts(node, "dilations", layer.convolution().dilation(), true))
            return false;
        if (!ConvertAtrributeInt(node, "group", layer.convolution().group(), true, 1u))
            return false;
        if (!ConvertAtrributeInts(node, "kernel_shape", layer.convolution().kernel(), true))
            return false;
        if (!ConvertAtrributeInts(node, "pads", layer.convolution().pad()))
            return false;
        if (!ConvertAtrributeInts(node, "strides", layer.convolution().stride()))
            return false;
        if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
            GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
        {
            const LayerParam* dequantized = GetLayer(layers, layer.src()[1]);
            if (dequantized->weight().empty())
                return false;
            const Shape& shape = dequantized->weight()[0].dim();
            layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.type() == Synet::LayerTypeConvolution)
            {
                layer.type() = Synet::LayerTypeQuantizedConvolution;
                if (!MoveDequantizeLinearToLayer(layers, layer))
                    return false;
            }
            else
                return false;
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }
        const LayerParam* weight = GetWeightLayer(layers, layer.src()[1]);
        if (weight == NULL || weight->type() != LayerTypeConst)
            return false;
        const Shape& shape = weight->weight()[0].dim();
        if (layer.convolution().kernel().empty())
        {
            if (shape.size() != 4)
                SYNET_ERROR("Convolution weight must be 4D tensor!");
            layer.convolution().kernel() = Shp(shape[2], shape[3]);
        }
        layer.weight().resize(layer.src().size() - 1);
        layer.weight()[0] = weight->weight()[0];
        layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
        layer.convolution().biasTerm() = layer.src().size() > 2;
        if (layer.convolution().biasTerm())
        {
            const LayerParam* bias = GetWeightLayer(layers, layer.src()[2]);
            if (bias == NULL || bias->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = bias->weight()[0];
        }
        layer.src().resize(1);
        if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
            return ReorderWeight(srcBin, Shape(), layer, dstBin);
        return true;
    }
}

#endif