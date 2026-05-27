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
    bool ConvertSqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 1, 2))
            return false;
        if (layer.src().size() == 1)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSqueeze;
            }
            else
            {
                layer.type() = Synet::LayerTypeSqueeze;
                if (!ConvertAtrributeInts(node, "axes", layer.squeeze().axes()))
                    return false;
            }
        }
        else if (layer.src().size() == 2)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                return false;
            const TensorParam& alpha = src1->meta().alpha();
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSqueeze;
                layer.meta().alpha() = alpha;
            }
            else
            {
                layer.type() = Synet::LayerTypeSqueeze;
                if (alpha.type() == TensorType64i)
                {
                    layer.squeeze().axes().resize(alpha.i64().size());
                    for (size_t i = 0; i < alpha.i64().size(); ++i)
                        layer.squeeze().axes()[i] = (int)alpha.i64()[i];
                }
                else
                    return false;
            }
            layer.src().resize(1);
        }
        return true;
    }
}

#endif