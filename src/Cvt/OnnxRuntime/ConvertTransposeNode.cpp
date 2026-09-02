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
    bool ConvertTransposeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const OnnxParam& onnxParam, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 1))
            return false;
        Shape order;
        if (!ConvertAtrributeInts(node, "perm", order))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        if (src0 == NULL)
            return false;
        if (src0->type() == LayerTypeMeta)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = MetaTypePermute;
            layer.meta().alpha().shape() = Shp(order.size());
            layer.meta().alpha().type() = TensorType64i;
            for (size_t i = 0; i < order.size(); ++i)
                layer.meta().alpha().i64().push_back(order[i]);
        }
        else
        {
            layer.type() = Synet::LayerTypePermute;
            if (trans)
            {
                bool permutedToNchw = CurrentTensorFormat(layers, layer.src(), true, false, onnxParam.globalPoolingPermuteToNchw(), tensorFormatMap) != TensorFormatNhwc;
                if (!permutedToNchw)
                {
                    if (order == Shape({ 0, 2, 1, 3, 4 }))
                        order = Shape({ 0, 1, 2, 4, 3 });
                    if (order == Shp(0, 1, 3, 2))
                        order = Shp(0, 2, 1, 3);
                    else if (order == Shape({ 0, 2, 3, 1 }))
                    {
                        order = Shape({ 0, 1, 2, 3 });
                        layer.permute().format() = TensorFormatNchw;
                    }
                    if (order == Shape({ 0, 2, 1 }))
                    {
                        order = Shape({ 0, 1, 2 });
                        layer.permute().format() = TensorFormatNchw;
                    }
                }
                else
                {
                    if (order == Shape({ 0, 3, 1, 2 }) && onnxParam.transpose0312PermuteToNhwc())
                    {
                        order = Shape({ 0, 1, 2, 3 });
                        layer.permute().format() = TensorFormatNhwc;
                    }
                }
            }
            layer.permute().order() = order;
        }
        return true;
    }
}

#endif