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
    bool ConvertReshapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        if (src1->type() != LayerTypeMeta)
            SYNET_ERROR("Reshape src[1] must be Meta type!");
        if (src1->meta().type() == MetaTypeStub)
        {
            src1 = GetLayer(layers, src1->src()[0]);
            if (src1 == NULL)
                return false;
        }
        if (src0->type() == Synet::LayerTypeMeta)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeReshape;
        }
        else if (src1->meta().type() == MetaTypeConst)
        {
            const TensorParam& alpha = src1->meta().alpha();
            if (alpha.shape().size() != 1)
                SYNET_ERROR("Reshape src[1] alpha must have 1D shape!");
            Shape& shape = layer.reshape().shape();
            layer.type() = LayerTypeReshape;
            shape = Shp(alpha.i64().data(), alpha.shape()[0]);
            layer.src().resize(1);
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, tensorFormatMap) == TensorFormatNhwc)
            {
                if (shape.size() == 5)
                {
                    shape = Shp(shape[0], shape[3], shape[4], shape[1], shape[2]);
                }
                if (shape.size() == 4)
                {
                    shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                }
                if (shape.size() == 3)
                {
                    shape = Shape({ shape[0], shape[2] , shape[1] });
                }
            }
        }
        else
        {
            layer.type() = LayerTypeReshape;
        }
        if (onnxParam.setReshapeAxis1() && layer.type() == LayerTypeReshape)
        {
            layer.reshape().axis() = 1;
            //if (layer.reshape().shape().size() > 1 && layer.reshape().shape()[0] == 1)
            //    layer.reshape().shape().erase(layer.reshape().shape().begin(), layer.reshape().shape().begin() + 1);
        }
        return true;
    }
}

#endif
