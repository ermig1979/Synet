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
    bool ConvertAddNode(const onnx::NodeProto& node, LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
            GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
        {
            layer.type() = Synet::LayerTypeAdd;
            //layer.type() = Synet::LayerTypeQuantizedAdd;
            //if (!MoveDequantizeLinearToLayer(layers, layer))
            //    return false;
            return true;
        }
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeAdd;
        }
        else if (src1->type() == LayerTypeConst && src1->weight()[0].dim() == Shp(1))
        {
            const float* shift = GetWeight<float>(original, src1->weight()[0]);
            layer.type() = Synet::LayerTypePower;
            layer.power().power() = 1.0f;
            layer.power().scale() = 1.0f;
            layer.power().shift() = shift[0];
            layer.src().resize(1);
        }
        else
        {
            if (onnxParam.addToEltwise())
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeSum;
            }
            else
                layer.type() = Synet::LayerTypeAdd;
            if (src0->type() == LayerTypeConst && src1->type() != LayerTypeConst)
                std::swap(layer.src()[0], layer.src()[1]);
        }
        return true;
    }
}

#endif