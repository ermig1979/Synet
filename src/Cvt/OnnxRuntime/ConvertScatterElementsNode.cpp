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
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    bool ConvertScatterElementsNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered)
    {
        if (!CheckSourceNumber(layer, 3))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src0 == NULL || src1 == NULL || src2 == NULL)
            return false;
        layer.type() = Synet::LayerTypeScatterNd;
        layer.scatter().version() = 1;
        if (!ConvertAtrributeInt(node, "axis", layer.scatter().axis()))
            return false;
        String reduction;
        if (!ConvertAtrributeString(node, "reduction", reduction))
            return false;
        if (reduction != "none")
            SYNET_ERROR("Unsupported ScatterElements reduction '" << reduction << "' !");
        return true;
    }
}

#endif
