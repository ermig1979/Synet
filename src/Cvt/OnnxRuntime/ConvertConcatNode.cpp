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
    bool ConvertConcatNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        if (src0 == NULL)
            return false;
        const LayerParam* src1 = layer.src().size() < 2 ? 0 : GetLayer(layers, layer.src()[1]);
        if (src0->type() == Synet::LayerTypeMeta || (src1 && src1->type() == Synet::LayerTypeMeta))
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypePack;
        }
        else
        {
            layer.type() = Synet::LayerTypeConcat;
            if (!ConvertAtrributeInt(node, "axis", layer.concat().axis()))
                return false;
            if (trans && CurrentTensorFormat(layers, layer.src(), true, true, true, tensorFormatMap) == TensorFormatNhwc)
            {
                Shape nchw = Shape({ 0, 3, 1, 2 });
                if (layer.concat().axis() >= 0 && layer.concat().axis() < 4)
                    layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
            }
        }
        return true;
    }
}

#endif