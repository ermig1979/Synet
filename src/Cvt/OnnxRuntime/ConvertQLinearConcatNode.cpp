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
    bool ConvertQLinearConcatNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        size_t sn = layer.src().size();
        if (sn < 5 || sn % 3 != 2)
            SYNET_ERROR("QLinearConcat: Wrong number of sources (" << sn << ") !");

        layer.type() = Synet::LayerTypeQuantizedConcat;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        if (src0->type() == LayerTypeConst && src1->type() == LayerTypeConst)
        {
            layer.qDst().resize(1);
            if (TensorSize(src0->weight()[0].dim()) == 1 && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.qDst()[0].scale() = GetWeight<float>(srcBin, src0->weight()[0])[0];
                layer.qDst()[0].type() = src1->weight()[0].type();
                switch (layer.qDst()[0].type())
                {
                case TensorType8u:
                    layer.qDst()[0].zero() = GetWeight<uint8_t>(srcBin, src1->weight()[0])[0];
                    break;
                default:
                    SYNET_ERROR("QLinearConcat: unsupported src[1] type!");
                }
            }
            else
                SYNET_ERROR("QLinearConcat: support only uniform quantized output!");
        }
        else
            SYNET_ERROR("QuantizeConcat: src[0] or src[1] is not const!");

        Strings sources;
        for (size_t ls = 0, ns = 2; ns < sn; ls += 1, ns += 3)
        {
            sources.push_back(layer.src()[ns + 0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[ns + 1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[ns + 2]);
            if (src1 == NULL || src2 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && src2->type() == LayerTypeConst)
            {
                layer.qSrc().resize(ls + 1);
                if (TensorSize(src1->weight()[0].dim()) == 1 && TensorSize(src2->weight()[0].dim()) == 1)
                {
                    layer.qSrc()[ls].scale() = GetWeight<float>(srcBin, src1->weight()[0])[0];
                    layer.qSrc()[ls].type() = src2->weight()[0].type();
                    switch (layer.qSrc()[ls].type())
                    {
                    case TensorType8u:
                        layer.qSrc()[ls].zero() = GetWeight<uint8_t>(srcBin, src2->weight()[0])[0];
                        break;
                    default:
                        SYNET_ERROR("QLinearConcat: unsupported src[" << sn + 2 << "] type!");
                    }
                }
                else
                    SYNET_ERROR("QLinearConcat: support only uniform quantized input!");
            }
            else
                SYNET_ERROR("QuantizeConcat: src[" << sn + 1 << "] or src[" << sn + 2 << "] is not const!");
        }

        if (!ConvertAtrributeInt(node, "axis", layer.concat().axis()))
            return false;
        if (trans && CurrentTensorFormat(layers, layer.src(), true, true, true, tensorFormatMap) == TensorFormatNhwc)
        {
            Shape nchw = Shape({ 0, 3, 1, 2 });
            if (layer.concat().axis() >= 0 && layer.concat().axis() < 4)
                layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
        }
        layer.src() = sources;

        return true;
    }
}

#endif