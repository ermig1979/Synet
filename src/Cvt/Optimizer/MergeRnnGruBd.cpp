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
    bool MergeRnnGruBd(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        const size_t RNN_GRU_BD_SIZE = 19;
        if (index == 0 || index + RNN_GRU_BD_SIZE >= src.size())
            return false;
        const LayerParam& parent = src[index - 1];
        if (parent.type() != LayerTypeTensorIterator || parent.src().size() != 2 || 
            parent.dst().size() != 1 || parent.tensorIterator().back().size() != 1)
            return false;
        for (size_t i = 0; i < RNN_GRU_BD_SIZE; ++i)
        {
            if (src[index + i].parent() != parent.name())
                return false;
        }
        if (src[index + 0].type() != LayerTypeInput || src[index + 1].type() != LayerTypeMeta)
            return false;
        if (src[index + 2].type() != LayerTypeSqueeze || src[index + 3].type() != LayerTypeInput)
            return false;
        if (src[index + 4].type() != LayerTypeConcat || src[index + 5].type() != LayerTypeInnerProduct || src[index + 5].weight().size() != 2)
            return false;
        if (src[index + 6].type() != LayerTypeSigmoid || src[index + 7].type() != LayerTypeUnpack)
            return false;
        if (src[index + 8].type() != LayerTypeEltwise || src[index + 9].type() != LayerTypePower)
            return false;
        if (src[index + 10].type() != LayerTypeEltwise || src[index + 11].type() != LayerTypeConcat)
            return false;
        if (src[index + 12].type() != LayerTypeInnerProduct || src[index + 12].weight().size() != 2 || src[index + 13].type() != LayerTypeUnaryOperation)
            return false;
        if (src[index + 14].type() != LayerTypeEltwise || src[index + 15].type() != LayerTypeEltwise)
            return false;
        if (src[index + 16].type() != LayerTypeStub || src[index + 17].type() != LayerTypeExpandDims || src[index + 18].type() != LayerTypeStub)
            return false;
        if (!src[index + RNN_GRU_BD_SIZE].parent().empty())
            return false;

        dst.push_back(src[index + 0]);
        dst.push_back(src[index + 3]);

        LayerParam layer;
        layer.type() = LayerTypeRnnGruBd;
        layer.parent() = parent.name();
        layer.name() = parent.name() + "_RnnGruBd";
        layer.src().push_back(src[index + 0].dst()[0]);
        layer.src().push_back(src[index + 3].dst()[0]);
        layer.dst().push_back(src[index + 18].dst()[0]);
        layer.dst().push_back(src[index + 16].dst()[0]);
        layer.weight().push_back(src[index + 5].weight()[0]);
        layer.weight().push_back(src[index + 5].weight()[1]);
        layer.weight().push_back(src[index + 12].weight()[0]);
        layer.weight().push_back(src[index + 12].weight()[1]);
        dst.push_back(layer);

        index += RNN_GRU_BD_SIZE - 1;
        return true;
    }
}
