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
    bool ConvertClipNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
    {
        layer.type() = Synet::LayerTypeRestrictRange;
        if (layer.src().size() > 1)
        {
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst || src1->weight().size() != 1)
                return false;
            if (node.input(1).empty())
            {
                const float* max = GetWeight<float>(original, src1->weight()[0]);
                layer.restrictRange().upper() = max[0];
            }
            else
            {
                const float* min = GetWeight<float>(original, src1->weight()[0]);
                layer.restrictRange().lower() = min[0];
            }
            if (layer.src().size() > 2)
            {
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                if (src2 == NULL || src2->type() != LayerTypeConst || src2->weight().size() != 1)
                    return false;
                const float* max = GetWeight<float>(original, src2->weight()[0]);
                layer.restrictRange().upper() = max[0];
            }
            layer.src().resize(1);
        }
        else
        {
            if (!ConvertAtrributeFloat(node, "min", layer.restrictRange().lower(), true, -FLT_MAX))
                return false;
            if (!ConvertAtrributeFloat(node, "max", layer.restrictRange().upper(), true, +FLT_MAX))
                return false;
        }
        return true;
    }
}

#endif