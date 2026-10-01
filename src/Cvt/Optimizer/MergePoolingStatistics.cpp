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
    bool MergePoolingStatistics(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 8)
            return false;
        if (src[index + 0].type() != LayerTypePooling || !src[index + 1].pooling().globalPooling()
            || src[index + 0].pooling().method() != PoolingMethodTypeAverage)
            return false;
        if (src[index + 1].type() != LayerTypePooling || !src[index + 1].pooling().globalPooling()
            || src[index + 1].pooling().method() != PoolingMethodTypeAverage || src[index + 1].src()[0] != src[index + 0].src()[0])
            return false;
        if (src[index + 2].type() != LayerTypeBinaryOperation || src[index + 2].binaryOperation().type() != BinaryOperationTypeSub ||
            src[index + 2].src()[0] != src[index + 0].src()[0] || src[index + 2].src()[1] != src[index + 1].dst()[0])
            return false;
        if (src[index + 3].type() != LayerTypeEltwise || src[index + 3].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 3].src()[0] != src[index + 2].dst()[0] || src[index + 3].src()[1] != src[index + 2].dst()[0])
            return false;
        if (src[index + 4].type() != LayerTypePooling || !src[index + 4].pooling().globalPooling()
            || src[index + 4].pooling().method() != PoolingMethodTypeAverage || src[index + 4].src()[0] != src[index + 3].dst()[0])
            return false;
        if (src[index + 5].type() != LayerTypePower || src[index + 5].power().power() != 1.0f || src[index + 5].power().scale() != 1.0f ||
            src[index + 5].src()[0] != src[index + 4].dst()[0])
            return false;
        if (src[index + 6].type() != LayerTypeUnaryOperation || src[index + 6].unaryOperation().type() != UnaryOperationTypeSqrt ||
            src[index + 6].src()[0] != src[index + 5].dst()[0])
            return false;
        if (src[index + 7].type() != LayerTypeConcat || src[index + 7].src().size() != 2 ||
            src[index + 7].src()[0] != src[index + 0].dst()[0] || src[index + 7].src()[1] != src[index + 6].dst()[0])
            return false;

        LayerParam layer;
        layer.type() = LayerTypePoolingStatistics;
        layer.name() = src[index + 7].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(src[index + 7].dst()[0]);
        layer.shared().floats().push_back(src[index + 5].power().shift());
        dst.push_back(layer);
        index += 7;
        return true;
    }
}
