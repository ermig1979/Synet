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
    bool ConvertSliceNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap *tensorFormatMap)
    {
        if (layer.src().size() == 1)
        {
            if (!ConvertAtrributeInts(node, "axes", layer.stridedSlice().axes()))
                return false;
            if (!ConvertAtrributeInts(node, "starts", layer.stridedSlice().beginDims()))
                return false;
            if (!ConvertAtrributeInts(node, "ends", layer.stridedSlice().endDims()))
                return false;
            layer.type() = Synet::LayerTypeStridedSlice;
        }
        else if (layer.src().size() == 3)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypeSlice;
            }
            else
                SYNET_ERROR("Slice src[0] must be Meta type!");
        }
        else if(layer.src().size() >= 4 && layer.src().size() <= 5)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src0 == NULL || src1 == NULL || src2 == NULL || src3 == NULL)
                return false;
            const LayerParam* src4 = NULL;
            if (layer.src().size() > 4)
            {
                src4 = GetLayer(layers, layer.src()[4]);
                if (src4 == NULL)
                    return false;
            }
            if (src0->type() == LayerTypeMeta)
            {
                if (!CheckSourceNumber(layer, 4, 5))
                    return false;
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypeSlice;
            }
            else
            {
                layer.type() = Synet::LayerTypeStridedSlice;
                if (layer.src().size() == 4)
                {
                    if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta)
                        SYNET_ERROR("Slice src[1], src[2] and src[3] must be Meta type!");
                    if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                        src3->meta().type() == Synet::MetaTypeConst)
                    {
                        if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 ||
                            src3->meta().alpha().i64().size() != 1)
                            SYNET_ERROR("Slice starts, ends and axes must have 1 value!");
                        layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                        layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                        layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                        //layer.stridedSlice().strideDims().push_back((size_t)src4->meta().alpha().i64()[0]);
                        if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
                        {
                            Shape nchw = Shape({ 0, 3, 1, 2 });
                            layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                        }
                        layer.src().resize(1);
                    }
                }
                else if (layer.src().size() == 5)
                {
                    if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta || src4->type() != LayerTypeMeta)
                        SYNET_ERROR("Slice src[1], src[2], src[3] and src[4] must be Meta type!");
                    if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                        src3->meta().type() == Synet::MetaTypeConst && src4->meta().type() == Synet::MetaTypeConst)
                    {
                        if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 ||
                            src3->meta().alpha().i64().size() != 1 || src4->meta().alpha().i64().size() != 1)
                            SYNET_ERROR("Slice starts, ends, axes and steps must have 1 value!");
                        layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                        layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                        layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                        layer.stridedSlice().strideDims().push_back(src4->meta().alpha().i64()[0]);
                        if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
                        {
                            Shape nchw = Shape({ 0, 3, 1, 2 });
                            layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                        }
                        layer.src().resize(1);
                    }
                }
            }
        }
        return true;
    }
}

#endif
