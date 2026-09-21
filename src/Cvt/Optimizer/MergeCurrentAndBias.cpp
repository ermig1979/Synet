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
    static const WeightParam* GetEltwiseWeight(size_t index, const LayerParams& layers)
    {
        if (index < layers.size() && (layers[index].type() == LayerTypeEltwise && layers[index].src().size() == 2) || layers[index].type() == LayerTypeAdd)
        {
            const LayerParam* src0 = GetLayer(layers, layers[index].src()[0]);
            if (src0 && src0->type() == LayerTypeConst)
                return src0->weight().data() + 0;
            const LayerParam* src1 = GetLayer(layers, layers[index].src()[1]);
            if (src1 && src1->type() == LayerTypeConst)
                return src1->weight().data() + 0;
        }
        return NULL;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeCurrentAndBias(const LayerParams& src, size_t& index, Bytes& bin, LayerParams& dst, Changes& changes)
    {
        if (index == 0)
            return false;
        const LayerParam & current = src[index - 1];
        const LayerParam & bias = src[index];
        const WeightParam* weight = GetEltwiseWeight(index, src);
        if (bias.type() == LayerTypeBias)
            weight = &bias.weight()[0];
        if(weight == NULL)
            return false;
        if (!(bias.src()[0] == current.name() || (bias.src().size() == 2 && bias.src()[1] == current.name())))
            return false;
        if (InsideLink(src, index - 1, 2))
            return false;
        switch (current.type())
        {
        case LayerTypeConvolution:
            if (current.convolution().biasTerm() || current.convolution().outputNum() != weight->dim()[0])
                return false;
            dst.back().convolution().biasTerm() = true;
            break;
        case LayerTypeInnerProduct:
            if (current.innerProduct().biasTerm() || current.src().size() != 1 || current.innerProduct().outputNum() != weight->dim()[0])
                return false;
            dst.back().innerProduct().biasTerm() = true;
            break;
        case LayerTypePower:
            if (current.power().power() != 1.0f || current.power().shift() != 0.0f || bias.type() != LayerTypeBias)
                return false;
            dst.back().type() = LayerTypeScale;
            dst.back().scale().biasTerm() = true;
            dst.back().weight().push_back(*weight);
            dst.back().weight()[0].offset() = bin.size();
            for (size_t i = 0; i < dst.back().weight()[0].dim()[0]; ++i)
                PushBack<float>(bin, current.power().scale());
            dst.back().power().scale() = 1.0f;
            break;
        case LayerTypeScale:
            if (current.scale().biasTerm())
                return false;
            dst.back().scale().biasTerm() = true;
            break;
            default:
                return false;
        }
        dst.back().name() = bias.name();
        dst.back().dst() = bias.dst();
        dst.back().weight().push_back(*weight);
        return true;
    }
}
