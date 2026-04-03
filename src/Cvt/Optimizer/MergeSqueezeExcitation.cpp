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
    bool MergeSqueezeExcitation(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() <= index + 4)
            return false;
        if (src[index + 0].type() != LayerTypePooling || src[index + 0].pooling().method() != PoolingMethodTypeAverage)
            return false;
        if (src[index + 1].type() != LayerTypeConvolution || src[index + 1].convolution().kernel() != Shp(1, 1) ||
            src[index + 1].src()[0] != src[index + 0].name())
            return false;
        if (src[index + 2].type() != LayerTypeConvolution || src[index + 2].convolution().kernel() != Shp(1, 1) ||
            src[index + 2].src()[0] != src[index + 1].name() || src[index + 2].convolution().activationType() != ActivationFunctionTypeIdentity)
            return false;
        if (src[index + 3].type() != LayerTypeSigmoid || src[index + 3].src()[0] != src[index + 2].name())
            return false;
        if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 4].src()[0] != src[index + 0].src()[0] || src[index + 4].src()[1] != src[index + 3].dst()[0])
            return false;
        if (InsideLink(src, index + 1, 4))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeSqueezeExcitation;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        for(size_t i = 0; i < src[index + 1].weight().size(); ++i)
            layer.weight().push_back(src[index + 1].weight()[i]);
        for (size_t i = 0; i < src[index + 2].weight().size(); ++i)
            layer.weight().push_back(src[index + 2].weight()[i]);
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.squeezeExcitation().biasTerm0() = src[index + 1].convolution().biasTerm();
        layer.squeezeExcitation().activationType() = src[index + 1].convolution().activationType();
        layer.squeezeExcitation().activationParam0() = src[index + 1].convolution().activationParam0();
        layer.squeezeExcitation().activationParam1() = src[index + 1].convolution().activationParam1();
        layer.squeezeExcitation().biasTerm1() = src[index + 2].convolution().biasTerm();
        dst.push_back(layer);
        index += 4;
        return true;
    }
}