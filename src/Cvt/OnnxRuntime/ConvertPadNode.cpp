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
    bool ConvertPadNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 1, 3))
            return false;

        layer.type() = Synet::LayerTypePad;
        String mode;
        if (!ConvertAtrributeString(node, "mode", mode, true, "constant"))
            return false;
        if (mode == "constant")
            layer.pad().mode() = PadModeConstant;
        else if (mode == "reflect")
            layer.pad().mode() = PadModeReflect;
        else if (mode == "edge")
            layer.pad().mode() = PadModeEdge;
        else if (mode == "wrap")
            layer.pad().mode() = PadModeWrap;
        else
            SYNET_ERROR("Unknown type of pad mode: " << mode << " !");

        if (layer.src().size() > 1)
        {
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() != LayerTypeMeta)
                SYNET_ERROR("Pad src[1] must be Meta type!");
            if (layer.src().size() > 2)
            {
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                if (src2 == NULL)
                    return false;
                if (src2->type() != LayerTypeConst)
                    SYNET_ERROR("Pad src[2] must be Const type!");
                if (src2->weight()[0].type() != TensorType32f)
                    SYNET_ERROR("Pad src[2] must have FP32 type!");
                if (GetWeight<float>(original, src2->weight()[0])[0] != 0)
                    SYNET_ERROR("Synet support only pad value == 0!");
                layer.src().resize(2);
            }
        }
        else
        {
            if (!ConvertAtrributeInts(node, "pads", layer.pad().pads()))
                return false;
        }

        return true;
    }
}

#endif
