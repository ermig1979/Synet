/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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
    bool ConvertScaledDotProductAttentionNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& reordered)
    {
        if (!CheckSourceNumber(layer, 4))
            return false;
        const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
        if (src3 == NULL)
            return false;
        if (src3->type() != LayerTypeConst)
            SYNET_ERROR("ScaledDotProductAttention src[3] must be Const type!");
        if (src3->weight().empty())
            SYNET_ERROR("ScaledDotProductAttention src[3] Const layer has no weight!");
        if (src3->weight()[0].type() != TensorType32f)
            SYNET_ERROR("ScaledDotProductAttention src[3] must have FP32 type!");
        if (src3->weight()[0].dim() != Shp(1))
            SYNET_ERROR("ScaledDotProductAttention src[3] must have shape {1} !");
        layer.type() = Synet::LayerTypeScaledDotProductAttention;
        layer.src().resize(3);
        return true;
    }
}

#endif
