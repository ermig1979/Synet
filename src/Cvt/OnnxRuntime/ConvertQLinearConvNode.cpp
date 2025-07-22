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
    bool ConvertQLinearConvNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 8, 9))
            return false;
        layer.type() = Synet::LayerTypeQuantizedConvolution;
        if (!ConvertAtrributeInts(node, "dilations", layer.convolution().dilation()))
            return false;
        if (!ConvertAtrributeInt(node, "group", layer.convolution().group()))
            return false;
        if (!ConvertAtrributeInts(node, "kernel_shape", layer.convolution().kernel(), true))
            return false;
        if (!ConvertAtrributeInts(node, "pads", layer.convolution().pad()))
            return false;
        if (!ConvertAtrributeInts(node, "strides", layer.convolution().stride()))
            return false;

        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src1 == NULL || src2 == NULL)
            return false;
        if (src1->type() == LayerTypeConst && src2->type() == LayerTypeConst)
        {
            layer.qSrc().resize(1);
            if (TensorSize(src1->weight()[0].dim()) == 1 && TensorSize(src2->weight()[0].dim()) == 1)
            {
                layer.qSrc()[0].scale() = GetWeight<float>(srcBin, src1->weight()[0])[0];
                layer.qSrc()[0].type() = src2->weight()[0].type();
                switch (layer.qSrc()[0].type())
                {
                case TensorType8u:
                    layer.qSrc()[0].zero() = GetWeight<uint8_t>(srcBin, src2->weight()[0])[0];
                    break;
                default:
                    SYNET_ERROR("QLinearConv: unsupported src[2] type!");
                }
            }
            else
                SYNET_ERROR("QLinearConv: support only uniform quantized input!");
        }
        else
            SYNET_ERROR("QuantizeLinear: src[1] or src[2] is not const!");

        const LayerParam* src3 = GetWeightLayer(layers, layer.src()[3]);
        const LayerParam* src4 = GetWeightLayer(layers, layer.src()[4]);
        const LayerParam* src5 = GetWeightLayer(layers, layer.src()[5]);
        if (src3 == NULL || src4 == NULL || src5 == NULL)
            return false;
        const Shape& shape = src3->weight()[0].dim();
        layer.convolution().outputNum() = uint32_t(shape[0]);
        layer.weight().push_back(src3->weight()[0]);
        layer.weight().push_back(src4->weight()[0]);
        layer.weight().push_back(src5->weight()[0]);
        layer.qSrc().resize(2);
        layer.qSrc()[1].weights() = 3;
        if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
            return ReorderWeight(srcBin, Shape(), layer, dstBin);

        const LayerParam* src6 = GetLayer(layers, layer.src()[6]);
        const LayerParam* src7 = GetLayer(layers, layer.src()[7]);
        if (src6 == NULL || src7 == NULL)
            return false;
        if (src6->type() == LayerTypeConst && src7->type() == LayerTypeConst)
        {
            layer.qDst().resize(1);
            if (TensorSize(src6->weight()[0].dim()) == 1 && TensorSize(src7->weight()[0].dim()) == 1)
            {
                layer.qDst()[0].scale() = GetWeight<float>(srcBin, src6->weight()[0])[0];
                layer.qDst()[0].type() = src7->weight()[0].type();
                switch (layer.qDst()[0].type())
                {
                case TensorType8u:
                    layer.qDst()[0].zero() = GetWeight<uint8_t>(srcBin, src7->weight()[0])[0];
                    break;
                default:
                    SYNET_ERROR("QLinearConv: unsupported src[7] type!");
                }
            }
            else
                SYNET_ERROR("QLinearConv: support only uniform quantized output!");
        }
        else
            SYNET_ERROR("QuantizeLinear: src[6] or src[7] is not const!");

        if (layer.src().size() > 8)
        {
            const LayerParam* src8 = GetWeightLayer(layers, layer.src()[8]);
            if (src8 == NULL)
                return false;
            if (src8->weight()[0].type() != TensorType32i)
                SYNET_ERROR("QLinearConv: support only INT32 bias!");
            layer.weight().push_back(src8->weight()[0]);
            layer.convolution().biasTerm() = true;
            layer.qSrc().resize(3);
            layer.qSrc()[2].weights() = 1;
            layer.qSrc()[2].type() = TensorType32i;
        }
        else
            layer.convolution().biasTerm() = false;

        layer.src().resize(1);

        return true;
    }
}

#endif