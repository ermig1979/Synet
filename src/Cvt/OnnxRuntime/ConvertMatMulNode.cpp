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
    bool ConvertMatMulNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, TensorFormatMap* tensorFormatMap)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        layer.type() = Synet::LayerTypeInnerProduct;
        int transB = false;
        layer.weight().resize(layer.src().size() - 1);
        layer.innerProduct().biasTerm() = false;
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL)
            return false;
        if (src1->type() == LayerTypeConst)
        {
            if (src1->weight().empty())
                SYNET_ERROR("MatMul src[1] Const layer has no weight!");
            layer.weight()[0] = src1->weight()[0];
        }
        else if (src1->type() == LayerTypePermute)
        {
            if (!CheckSourceNumber(*src1, 1))
                return false;
            const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
            if (src10 == NULL)
                return false;
            if (src10->type() == LayerTypeConst)
            {
                if (src10->weight().empty())
                    SYNET_ERROR("MatMul permute src[1] Const source has no weight!");
                transB = true;
                layer.weight() = src10->weight();
                layers.erase(layers.begin() + (src1 - layers.data()));
            }
        }
        Shape weight = layer.weight()[0].dim();
        layer.innerProduct().transposeB() = !transB;
        if (weight.empty())
        {
            layer.weight().clear();
            layer.innerProduct().outputNum() = 0;
            layer.innerProduct().axis() = -1;
        }
        else
        {
            //if (!CheckSignificantDims(weight, 2, "MatMul weight"))
            //    return false;
            if (weight.size() > 2)
                layer.innerProduct().axis() = weight.size() - 1;
            layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[weight.size() - 2] : weight[weight.size() - 1]);
            layer.src().resize(1);
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, tensorFormatMap) == TensorFormatNhwc)
                SYNET_ERROR("Can 't convert MatMul node for NHWC format!");
        }
        return true;
    }
}

#endif
