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

#include "Cvt/Optimizer/Common.h"
#include "Cvt/Optimizer/Optimizer.h"

namespace Synet
{
    bool MergeDynamicQuantizedInnerProduct(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 6)
            return false;
        if (src[index + 0].type() != LayerTypeDynamicQuantizeLinear || src[index + 0].dst().size() != 3)
            return false;
        if (src[index + 1].type() != LayerTypeEltwise || src[index + 1].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 1].src()[0] != src[index + 0].dst()[1] || src[index + 1].src().size() != 2)
            return false;
        size_t is11 = GetLayerIndex(src, src[index + 1].src()[1]);
        if (is11 >= src.size() || is11 >= dst.size() || src[is11].type() != LayerTypeConst)
            return false;
        if (src[index + 2].type() != LayerTypeMatMulInteger ||
            src[index + 2].src()[0] != src[index + 0].dst()[0] || src[index + 2].src()[1] != src[index + 0].dst()[2])
            return false;
        if (src[index + 3].type() != LayerTypeCast || src[index + 3].cast().type() != TensorType32f ||
            src[index + 3].src()[0] != src[index + 2].dst()[0])
            return false;
        if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 4].src()[0] != src[index + 3].dst()[0] || src[index + 4].src()[1] != src[index + 1].dst()[0])
            return false;

        LayerParam layer;
        layer.type() = LayerTypeDynamicQuantizedInnerProduct;
        layer.name() = src[index + 2].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.weight() = src[index + 2].weight();
        layer.weight().push_back(src[is11].weight()[0]);
        layer.innerProduct().outputNum() = src[index + 2].weight()[0].dim()[1];
        layer.innerProduct().biasTerm() = false;
        if (src[index + 5].type() == LayerTypeEltwise && src[index + 5].eltwise().operation() == EltwiseOperationTypeSum &&
            src[index + 5].src().size() == 2 && src[index + 5].src()[0] == src[index + 4].dst()[0])
        {
            size_t is51 = GetLayerIndex(src, src[index + 5].src()[1]);
            if (is51 < src.size() && src[is51].type() == LayerTypeConst)
            {
                layer.weight().push_back(src[is51].weight()[0]);
                layer.innerProduct().biasTerm() = true;
                layer.dst() = src[index + 5].dst();
            }
        }
        dst.push_back(layer);
        index += layer.innerProduct().biasTerm() ? 5 : 4;
        return true;
    }
}
