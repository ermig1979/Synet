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
    bool MergeUnpack4(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 19)
            return false;
        if (src[index + 0].type() != LayerTypeMeta || src[index + 0].meta().type() != MetaTypeShape)
            return false;
        if (!IsMetaConst64i(src[index + 1]))
            return false;
        if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeGather)
            return false;
        if (!IsMetaConst64i(src[index + 3]))
            return false;
        if (!IsMetaConst64i(src[index + 4]))
            return false;
        if (src[index + 5].type() != LayerTypeMeta || src[index + 5].meta().type() != MetaTypeAdd)
            return false;
        if (!IsMetaConst64i(src[index + 6], Lng(4)))
            return false;
        if (src[index + 7].type() != LayerTypeMeta || src[index + 7].meta().type() != MetaTypeDiv)
            return false;
        if (!IsMetaConst64i(src[index + 8], Lng(1)))
            return false;
        if (src[index + 9].type() != LayerTypeMeta || src[index + 9].meta().type() != MetaTypeMul)
            return false;
        if (src[index + 10].type() != LayerTypeStridedSlice || src[index + 10].src()[0] != src[index + 0].src()[0])
            return false;
        if (!IsMetaConst64i(src[index + 11], Lng(2)))
            return false;
        if (src[index + 12].type() != LayerTypeMeta || src[index + 12].meta().type() != MetaTypeMul)
            return false;
        if (src[index + 13].type() != LayerTypeStridedSlice || src[index + 13].src()[0] != src[index + 0].src()[0])
            return false;
        if (!IsMetaConst64i(src[index + 14], Lng(3)))
            return false;
        if (src[index + 15].type() != LayerTypeMeta || src[index + 15].meta().type() != MetaTypeMul)
            return false;
        if (src[index + 16].type() != LayerTypeStridedSlice || src[index + 16].src()[0] != src[index + 0].src()[0])
            return false;
        if (!IsMetaConst64i(src[index + 17], Lng(4)))
            return false;
        if (src[index + 18].type() != LayerTypeMeta || src[index + 18].meta().type() != MetaTypeMul)
            return false;
        if (src[index + 19].type() != LayerTypeStridedSlice || src[index + 19].src()[0] != src[index + 0].src()[0])
            return false;

        LayerParam layer;
        layer.type() = LayerTypeUnpack;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.unpack().axis() = isNhwc ? 3 : 1;
        layer.dst().push_back(src[index + 10].dst()[0]);
        layer.dst().push_back(src[index + 13].dst()[0]);
        layer.dst().push_back(src[index + 16].dst()[0]);
        layer.dst().push_back(src[index + 19].dst()[0]);
        dst.push_back(layer);
        index += 19;
        return true;
    }
}
