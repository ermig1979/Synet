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
    bool ConvertMulNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        if (src0->type() == LayerTypeConst)
        {
            std::swap(src0, src1);
            std::swap(layer.src()[0], layer.src()[1]);
        }
        if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
        {
            layer.type() = Synet::LayerTypePower;
            layer.power().power() = 1.0f;
            layer.power().shift() = 0.0f;
            if (src1->weight()[0].type() == TensorType32f)
            {
                const float* scale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = scale[0];
            }
            else if (src1->weight()[0].type() == TensorType32i)
            {
                const int32_t* scale = GetWeight<int32_t>(original, src1->weight()[0]);
                layer.power().scale() = (float)scale[0];
            }
            layer.src().resize(1);
        }
        else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 3 && src1->weight()[0].dim()[0] != 1)
        {
            layer.type() = Synet::LayerTypeScale;
            layer.weight() = src1->weight();
            //if (!CompactShape(layer.weight()[0].dim()))
            //    return false;
            layer.src().resize(1);
        }
        else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 4 && src1->weight()[0].dim()[1] != 1)
        {
            layer.type() = Synet::LayerTypeScale;
            layer.weight() = src1->weight();
            if (!CompactShape(layer.weight()[0].dim()))
                return false;
            layer.src().resize(1);
        }
        else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeMul;
        }
        else
        {
            if (onnxParam.mulToEltwise())
            {
                layer.type() = Synet::LayerTypeEltwise;
                layer.eltwise().operation() = EltwiseOperationTypeProduct;
            }
            else
                layer.type() = Synet::LayerTypeMul;
            if (src0->type() == LayerTypeConst && src1->type() != LayerTypeConst)
                std::swap(layer.src()[0], layer.src()[1]);
        }
        return true;
        }
}

#endif