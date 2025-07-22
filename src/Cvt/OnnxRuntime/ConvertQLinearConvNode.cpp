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
        if (!CheckSourceNumber(layer, 9))
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

        return true;
    }
}

#endif