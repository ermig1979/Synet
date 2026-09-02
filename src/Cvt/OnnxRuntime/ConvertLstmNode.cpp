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
    bool ConvertLstmNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
    {
        layer.type() = Synet::LayerTypeLstm;
        if (!ConvertAtrributeInt(node, "hidden_size", layer.lstm().hiddenSize()))
            return false;
        String direction;
        if (!ConvertAtrributeString(node, "direction", direction))
            return false;
        if (direction == "forward")
            layer.lstm().direction() = LstmDirectionTypeForward;
        else if (direction == "reverse")
            layer.lstm().direction() = LstmDirectionTypeReverse;
        else if (direction == "bidirectional")
            layer.lstm().direction() = LstmDirectionTypeBidirectional;
        else
            SYNET_ERROR("Unsupported LSTM direction '" << direction << "' !");
        if (!CheckSourceNumber(layer, 6))
            return false;
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL)
            return false;
        if (src1->type() != LayerTypeConst)
            SYNET_ERROR("LSTM src[1] must be Const type!");
        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src2 == NULL)
            return false;
        if (src2->type() != LayerTypeConst)
            SYNET_ERROR("LSTM src[2] must be Const type!");
        const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
        if (src3 == NULL)
            return false;
        if (src3->type() != LayerTypeConst)
            SYNET_ERROR("LSTM src[3] must be Const type!");
        layer.weight().resize(3);
        layer.weight()[0] = src1->weight()[0];
        layer.weight()[1] = src2->weight()[0];
        layer.weight()[2] = src3->weight()[0];
        layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 4);
        layer.dst().resize(1);
        return true;
    }
}

#endif
