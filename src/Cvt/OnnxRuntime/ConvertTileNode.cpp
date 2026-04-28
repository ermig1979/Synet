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
    bool ConvertTileNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL)
            return false;
        layer.type() = Synet::LayerTypeTile;
        if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst && src1->meta().alpha().type() == TensorType64i)
        {
            Longs shape = src1->meta().alpha().i64();
            if (trans && CurrentTensorFormat(layers, layer.src(), false, false, false, tensorFormatMap) == TensorFormatNhwc)
                SYNET_ERROR("Unsupport NHWC format!");
            for (size_t i = 0, already = 0; i < shape.size(); ++i)
            {
                if (shape[i] != 1)
                {
                    if (already)
                        return false;
                    layer.tile().axis() = i;
                    layer.tile().tiles() = (uint32_t)shape[i];
                    already = 1;
                }
            }
            layer.src().resize(1);
        }
        return true;
    }
}

#endif