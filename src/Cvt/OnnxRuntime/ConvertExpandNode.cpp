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
    bool ConvertExpandNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        if (src1->type() == LayerTypeMeta)
        {
            const MetaParam & meta = src1->meta();
            if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i && AllEqualTo(meta.alpha().i64(), int64_t(1)))
            {
                layer.type() = Synet::LayerTypeStub;
                layer.src().resize(1);
            }
            else if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i &&
                src0->type() == LayerTypeConst && Shp(meta.alpha().i64()) == src0->weight()[0].dim())
            {
                layer.type() = Synet::LayerTypeStub;
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeTile;
            }
        }
        else
            SYNET_ERROR("Unsupported src[1] type!");
        return true;
    }
}

#endif
