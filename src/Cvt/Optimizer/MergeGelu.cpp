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
    bool MergeGelu(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (!IsMulConst(src[index + 0], M_SQRT1_2))
            return false;
        if (src[index + 1].type() != LayerTypeUnaryOperation || src[index + 1].unaryOperation().type() != UnaryOperationTypeErf ||
            src[index + 1].src()[0] != src[index + 0].dst()[0])
            return false;
        if (!IsAddConst(src[index + 2], 1.0f) || src[index + 2].src()[0] != src[index + 1].dst()[0])
            return false;
        if (src[index + 3].type() != LayerTypeEltwise || src[index + 3].eltwise().operation() != Synet::EltwiseOperationTypeProduct ||
            src[index + 3].src()[0] != src[index + 0].src()[0] || src[index + 3].src()[1] != src[index + 2].dst()[0])
            return false;
        if (!IsMulConst(src[index + 4], 0.5f) || src[index + 4].src()[0] != src[index + 3].dst()[0])
            return false;
        if (InsideLink(src, index + 1, 4))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeGelu;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        dst.push_back(layer);
        index += 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeGeluV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (!IsMulConst(src[index + 0], M_SQRT1_2))
            return false;
        if (src[index + 1].type() != LayerTypeUnaryOperation || src[index + 1].unaryOperation().type() != UnaryOperationTypeErf ||
            src[index + 1].src()[0] != src[index + 0].dst()[0])
            return false;
        if (!IsAddConst(src[index + 2], 1.0f) || src[index + 2].src()[0] != src[index + 1].dst()[0])
            return false;
        if (!IsMulConst(src[index + 3], 0.5f) || src[index + 3].src()[0] != src[index + 2].dst()[0])
            return false;
        if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].eltwise().operation() != Synet::EltwiseOperationTypeProduct ||
            src[index + 4].src()[0] != src[index + 0].src()[0] || src[index + 4].src()[1] != src[index + 3].dst()[0])
            return false;
        if (InsideLink(src, index + 1, 4))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeGelu;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        dst.push_back(layer);
        index += 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeGeluV3(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        size_t is4 = index;
        if (src[is4].type() != LayerTypeEltwise || src[is4].eltwise().operation() != Synet::EltwiseOperationTypeProduct || src[is4].src().size() != 2)
            return false;
        size_t is3 = GetLayerIndex(src, src[is4].src()[1]);
        size_t id3 = GetLayerIndex(dst, src[is4].src()[1]);
        if (is3 >= src.size() || id3 >= dst.size() || UserCount(src, is3) > 1)
            return false;
        if (!IsMulConst(src[is3], 0.5f))
            return false;
        size_t is2 = GetLayerIndex(src, src[is3].src()[0]);
        size_t id2 = GetLayerIndex(dst, src[is3].src()[0]);
        if (is2 >= src.size() || id2 >= dst.size() || UserCount(src, is2) > 1)
            return false;
        if (!IsAddConst(src[is2], 1.0f))
            return false;
        size_t is1 = GetLayerIndex(src, src[is2].src()[0]);
        size_t id1 = GetLayerIndex(dst, src[is2].src()[0]);
        if (is1 >= src.size() || id1 >= dst.size() || UserCount(src, is1) > 1)
            return false;
        if (src[is1].type() != LayerTypeUnaryOperation || src[is1].unaryOperation().type() != UnaryOperationTypeErf)
            return false;
        size_t is0 = GetLayerIndex(src, src[is1].src()[0]);
        size_t id0 = GetLayerIndex(dst, src[is1].src()[0]);
        if (is0 >= src.size() || id0 >= dst.size() || UserCount(src, is0) > 1)
            return false;
        if (!IsMulConst(src[is0], M_SQRT1_2))
            return false;
        if (src[is4].src()[0] != src[is0].src()[0])
            return false;

        dst.erase(dst.begin() + id3, dst.begin() + id3 + 1);
        dst.erase(dst.begin() + id2, dst.begin() + id2 + 1);
        dst.erase(dst.begin() + id1, dst.begin() + id1 + 1);
        dst.erase(dst.begin() + id0, dst.begin() + id0 + 1);

        LayerParam layer;
        layer.type() = LayerTypeGelu;
        layer.name() = src[is4].name();
        layer.src() = src[is0].src();
        layer.dst() = src[is4].dst();
        dst.push_back(layer);
        return true;
    }
}
