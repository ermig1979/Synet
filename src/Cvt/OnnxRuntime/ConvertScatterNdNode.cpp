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
    bool ConvertScatterNdNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered)
    {
        if (!CheckSourceNumber(layer, 3))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src0 == NULL || src1 == NULL || src2 == NULL)
            return false;
        layer.type() = Synet::LayerTypeScatterNd;
        if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst)
        {
            const TensorParam& alpha = src1->meta().alpha();
            size_t size = TensorSize(alpha.shape()), offset = reordered.size();
            layer.type() = Synet::LayerTypeScatterNd;
            layer.weight().resize(1);
            layer.weight()[0].dim() = alpha.shape();
            layer.weight()[0].type() = TensorType32i;
            layer.weight()[0].offset() = offset;
            layer.weight()[0].size() = size * 4;
            layer.src().erase(layer.src().begin() + 1);
            original.resize(offset + size * 4);
            reordered.resize(offset + size * 4);
            if (alpha.type() == TensorType64i)
            {
                const int64_t* src = alpha.i64().data();
                int32_t* dst = GetWeight<int32_t>(reordered, layer.weight()[0]);
                for (size_t i = 0; i < size; ++i)
                    dst[i] = (int32_t)src[i];
            }
            else
                SYNET_ERROR("ScatterND src[1] type must be meta const int64!");
        }
        return true;
    }
}

#endif
