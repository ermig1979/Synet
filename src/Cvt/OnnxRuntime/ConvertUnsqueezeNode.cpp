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
    bool ConvertUnsqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
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
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeExpandDims;
                layer.meta().alpha().type() = TensorType64i;
                if (!ConvertAtrributeInts(node, "axes", layer.meta().alpha().i64()))
                    return false;
                layer.meta().alpha().shape().resize(1, layer.meta().alpha().i64().size());
            }
            else
            {
                layer.type() = Synet::LayerTypeExpandDims;
                if (!ConvertAtrributeInts(node, "axes", layer.expandDims().axes()))
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
                SYNET_ERROR("Unsqueeze src[1] must be Meta Const type!");
            const TensorParam & alpha = src1->meta().alpha();
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeExpandDims;
                layer.meta().alpha() = alpha;
            }
            else
            {
                layer.type() = Synet::LayerTypeExpandDims;
                if (alpha.type() == TensorType64i)
                {
                    layer.expandDims().axes().resize(alpha.i64().size());
                    for (size_t i = 0; i < alpha.i64().size(); ++i)
                        layer.expandDims().axes()[i] = (int)alpha.i64()[i];
                }
                else
                    SYNET_ERROR("Unsqueeze src[1] alpha must be TensorType64i!");
            }
            layer.src().resize(1);
        }
        return true;
    }
}

#endif
