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
    bool ConvertReduceMaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 1, 2))
            return false;
        layer.type() = Synet::LayerTypeReduction;
        layer.reduction().type() = ReductionTypeMax;
        if (layer.src().size() == 1)
        {
            if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                return false;
        }
        else
        {
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 && IsMetaConst64i(*src1))
            {
                const Longs& val = src1->meta().alpha().i64();
                for (size_t i = 0; i < val.size(); ++i)
                    layer.reduction().axis().push_back((int)val[i]);
                layer.src().resize(1);
            }
            else
                SYNET_ERROR("ReduceMaxNode: src[1] is not const!");
        }
        if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
            return false;
        if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
        {
            Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
            for (size_t i = 0; i < axis.size(); ++i)
                layer.reduction().axis()[i] = nchw[axis[i]];
        }
        return true;
    }
}

#endif