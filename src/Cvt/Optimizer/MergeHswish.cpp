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
    bool MergeHswish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 4)
            return false;
        if (src[index + 0].type() != LayerTypePower || src[index + 0].power().power() != 1.0f ||
            src[index + 0].power().scale() != 1.0f)
            return false;
        if (src[index + 1].type() != LayerTypeRestrictRange || src[index + 1].src()[0] != src[index + 0].name() ||
            src[index + 1].restrictRange().lower() != 0.0f)
            return false;
        if (src[index + 2].type() != LayerTypePower || src[index + 2].power().power() != 1.0f ||
            src[index + 2].power().shift() != 0.0f || src[index + 2].src()[0] != src[index + 1].name())
            return false;
        if (src[index + 3].type() != LayerTypeEltwise || src[index + 3].src().size() != 2 ||
            src[index + 3].src()[0] != src[index + 0].src()[0] || src[index + 3].src()[1] != src[index + 2].name() ||
            src[index + 3].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        if (!Equal(src[index + 0].power().shift() * 2.0f, src[index + 1].restrictRange().upper()))
            return false;
        if (InsideLink(src, index + 1, 3))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeHswish;
        layer.name() = src[index + 3].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.hswish().shift() = src[index + 0].power().shift();
        layer.hswish().scale() = src[index + 2].power().scale();
        dst.push_back(layer);
        index += 3;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeHswishV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        if (src[index + 0].type() != LayerTypeHardSigmoid || src[index + 0].hardSigmoid().scale() != 1.0f / 6.0f ||
            src[index + 0].hardSigmoid().shift() != 0.5f)
            return false;
        if (src[index + 1].type() != LayerTypeEltwise || src[index + 1].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 1].src().size() != 2 || src[index + 1].src()[0] != src[index + 0].src()[0] || src[index + 1].src()[1] != src[index + 0].dst()[0])
            return false;
        if (InsideLink(src, index + 1, 1))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeHswish;
        layer.name() = src[index + 1].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        //layer.hswish().shift() = src[index + 0].power().shift();
        //layer.hswish().scale() = src[index + 2].power().scale();
        dst.push_back(layer);
        index += 1;
        return true;
    }
}
