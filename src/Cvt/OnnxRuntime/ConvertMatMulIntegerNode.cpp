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
    bool ConvertMatMulIntegerNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 4))
            return false;
        layer.type() = Synet::LayerTypeMatMulInteger;
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
        if (src1 == NULL || src3 == NULL)
            return false;
        if (src1->type() == LayerTypeConst || src3->type() == LayerTypeConst)
        {
            layer.weight().resize(2);
            layer.weight()[0] = src1->weight()[0];
            layer.weight()[1] = src3->weight()[0];
            layer.src().erase(layer.src().begin() + 3, layer.src().begin() + 4);
            layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 2);
        }
        if (trans && CurrentTensorFormat(layers, layer.src(), false, false, false, tensorFormatMap) == TensorFormatNhwc)
            SYNET_ERROR("Can 't convert MatMulInteger node for NHWC format!");
        return true;
    }
}

#endif
