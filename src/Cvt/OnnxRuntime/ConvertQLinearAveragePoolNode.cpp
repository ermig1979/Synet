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
    bool ConvertQLinearAveragePoolNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 5))
            return false;
        layer.type() = Synet::LayerTypeQuantizedPooling;
        layer.pooling().method() = PoolingMethodTypeAverage;

        if (!ConvertAtrributeInts(node, "kernel_shape", layer.pooling().kernel()))
            return false;
        if (!ConvertAtrributeInts(node, "pads", layer.pooling().pad(), true))
            return false;
        if (!ConvertAtrributeInts(node, "strides", layer.pooling().stride()))
            return false;
        if (GetAtrribute(node, "ceil_mode") == NULL)
            layer.pooling().roundingType() = RoundingTypeFloor;
        else
        {
            int ceilMode;
            if (!ConvertAtrributeInt(node, "ceil_mode", ceilMode))
                return false;
            layer.pooling().roundingType() = ceilMode ? RoundingTypeCeil : RoundingTypeFloor;
        }
        if (GetAtrribute(node, "count_include_pad"))
        {
            int64_t countIncludePad;
            if (!ConvertAtrributeInt(node, "count_include_pad", countIncludePad))
                return false;
            layer.pooling().excludePad() = (countIncludePad == 0);
        }

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
                    SYNET_ERROR("QLinearAveragePool: unsupported src[2] type!");
                }
            }
            else
                SYNET_ERROR("QLinearAveragePool: support only uniform quantized input!");
        }
        else
            SYNET_ERROR("QuantizeAveragePool: src[1] or src[2] is not const!");

        const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
        const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
        if (src3 == NULL || src4 == NULL)
            return false;
        if (src3->type() == LayerTypeConst && src4->type() == LayerTypeConst)
        {
            layer.qDst().resize(1);
            if (TensorSize(src3->weight()[0].dim()) == 1 && TensorSize(src4->weight()[0].dim()) == 1)
            {
                layer.qDst()[0].scale() = GetWeight<float>(srcBin, src3->weight()[0])[0];
                layer.qDst()[0].type() = src4->weight()[0].type();
                switch (layer.qDst()[0].type())
                {
                case TensorType8u:
                    layer.qDst()[0].zero() = GetWeight<uint8_t>(srcBin, src4->weight()[0])[0];
                    break;
                default:
                    SYNET_ERROR("QLinearAveragePool: unsupported src[4] type!");
                }
            }
            else
                SYNET_ERROR("QLinearAveragePool: support only uniform quantized output!");
        }
        else
            SYNET_ERROR("QuantizeAveragePool: src[3] or src[4] is not const!");

        layer.src().resize(1);

        return true;
    }
}

#endif