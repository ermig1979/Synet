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
    bool ConvertReduceMeanNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, UniqNames& merged)
    {
        if (!CheckSourceNumber(layer, 1, 2))
            return false;
        Ints axes;
        if (layer.src().size() == 1)
        {
            if (!ConvertAtrributeInts(node, "axes", axes))
                return false;
        }
        else
        {
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 && IsMetaConst64i(*src1))
            {
                const Longs& val = src1->meta().alpha().i64();
                for (size_t i = 0; i < val.size(); ++i)
                    axes.push_back((int)val[i]);
                layer.src().resize(1);
            }
            else
                SYNET_ERROR("ReduceMeanNode: src[1] is not const!");
        }

        if (axes == Ints({ 2, 3 }) || axes == Ints({ -1, -2 }))
        {
            if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear)
            {
                layer.type() = Synet::LayerTypeQuantizedPooling;
                layer.pooling().method() = PoolingMethodTypeAverage;
                layer.pooling().globalPooling() = true;
                if (!MoveDequantizeLinearToLayer(layers, layer, merged))
                    return false;
            }
            else
            {
                layer.type() = Synet::LayerTypePooling;
                layer.pooling().method() = PoolingMethodTypeAverage;
                layer.pooling().globalPooling() = true;
            }
        }
        else
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeMean;
            for (size_t i = 0; i < axes.size(); ++i)
                layer.reduction().axis().push_back(axes[i]);
            ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims(), true, true);
            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true) == TensorFormatNhwc)
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for (size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
        }
        return true;
    }
}

#endif