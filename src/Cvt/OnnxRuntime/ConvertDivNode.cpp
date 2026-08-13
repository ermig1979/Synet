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
    bool ConvertDivNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;
        if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
        {
            layer.type() = Synet::LayerTypePower;
            const float* pScale = GetWeight<float>(original, src1->weight()[0]);
            layer.power().scale() = float(1.0 / double(pScale[0]));
            layer.src().resize(1);
        }
        else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
        {
            const float* pSrc0 = GetWeight<float>(original, src0->weight()[0]);
            if (pSrc0[0] != 1.0f)
                SYNET_ERROR("DivNode: unsupported constant src[0] value " << pSrc0[0] << " (only 1.0 is supported)!");
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeRcp;
            layer.src().erase(layer.src().begin());
            return true;
        }
        else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
        {
            const float* pSrc = GetWeight<float>(original, src1->weight()[0]);
            size_t size = TensorSize(src1->weight()[0].dim());
            bool uniform = true;
            for (size_t i = 1; i < size && uniform; ++i)
                uniform = (pSrc[i] == pSrc[0]);
            if (uniform)
            {
                layer.type() = Synet::LayerTypePower;
                layer.power().scale() = 1.0f / pSrc[0];
            }
            else
            {
                layer.weight() = src1->weight();
                const Shape& dim = src1->weight()[0].dim();
                if ((dim.size() == 4 && dim[1] != 1) || (dim.size() == 3 && dim[0] != 1) || dim.size() == 1)
                {
                    layer.type() = Synet::LayerTypeScale;
                    layer.scale().biasTerm() = false;
                    if (dim.size() == 1)
                        layer.scale().axis() = -1;
                    if (!CompactShape(layer.weight()[0].dim()))
                        return false;                    
                }
                else
                {
                    layer.type() = Synet::LayerTypeMul;
                }
                float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = 1.0f / pSrc[i];
            }
            layer.src().resize(1);
        }
        else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
        {
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeDiv;
        }
        else
        {
            layer.type() = Synet::LayerTypeBinaryOperation;
            layer.binaryOperation().type() = BinaryOperationTypeDiv;
        }
        return true;
    }
}

#endif
