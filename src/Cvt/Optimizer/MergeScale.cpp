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

    bool MergeScale(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        const WeightParam* scale = GetEltwiseWeight(index + 0, src);
        if (scale == NULL || src[index + 0].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        const WeightParam* shift = GetEltwiseWeight(index + 1, src);
        if (shift == NULL || !IsAdd(src[index + 1]))
            return false;
        if(src[index + 1].src()[0] != src[index + 0].dst()[0])
            return false;
        if (scale->dim() != shift->dim())
            return false;

        LayerParam layer;
        layer.type() = LayerTypeScale;
        layer.name() = src[index + 1].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.scale().biasTerm() = true;
        layer.weight().push_back(*scale);
        layer.weight().push_back(*shift);
        dst.push_back(layer);
        index += 1;
        return true;
    }
}
