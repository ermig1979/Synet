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

    //--------------------------------------------------------------------------------------------------

    bool MergeMish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypeUnaryOperation ||
            src[index + 0].unaryOperation().type() != UnaryOperationTypeExp)
            return false;
        if (src[index + 1].type() != LayerTypePower || src[index + 1].power().power() != 1.0f ||
            src[index + 1].power().scale() != 1.0f || src[index + 1].power().shift() != 1.0f ||
            src[index + 1].src()[0] != src[index + 0].name())
            return false;
        if (src[index + 2].type() != LayerTypeUnaryOperation ||
            src[index + 2].unaryOperation().type() != UnaryOperationTypeLog ||
            src[index + 2].src()[0] != src[index + 1].name())
            return false;
        if (src[index + 3].type() != LayerTypeUnaryOperation ||
            src[index + 3].unaryOperation().type() != UnaryOperationTypeTanh ||
            src[index + 3].src()[0] != src[index + 2].name())
            return false;
        if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].src().size() != 2 ||
            src[index + 4].src()[0] != src[index + 0].src()[0] || src[index + 4].src()[1] != src[index + 3].name() ||
            src[index + 4].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        if (InsideLink(src, index + 1, 4))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeMish;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        dst.push_back(layer);
        index += 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergePrelu0(const LayerParams& src, size_t& index, const Bytes& bin, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        if (src[index + 0].type() != LayerTypeScale)
            return false;
        if (src[index + 1].type() != LayerTypeEltwise || src[index + 1].src().size() != 2 ||
            src[index + 1].src()[1] != src[index + 0].src()[0] || src[index + 1].src()[0] != src[index + 0].name() ||
            src[index + 1].eltwise().operation() != EltwiseOperationTypeMax)
            return false;
        if (InsideLink(src, index + 1, 1))
            return false;
        const float* scale = GetWeight<float>(bin, src[index].weight()[0]);
        for (size_t i = 0, n = src[index].weight()[0].size() / 4; i < n; ++i)
            if (scale[i] < -1.0f || scale[i] > 1.0f)
                return false;
        if (src[index + 0].weight().size() > 1)
        {
            const float* shift = GetWeight<float>(bin, src[index].weight()[1]);
            for (size_t i = 0, n = src[index].weight()[1].size() / 4; i < n; ++i)
                if (shift[i] != 0.0f)
                    return false;
        }
        LayerParam layer;
        layer.type() = LayerTypePrelu;
        layer.name() = src[index + 1].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.prelu().axis() = src[index + 0].scale().axis();
        layer.weight().push_back(src[index + 0].weight()[0]);
        dst.push_back(layer);
        index += 1;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergePrelu1(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypePower || src[index + 0].power().scale() != -1.0f)
            return false;
        if (src[index + 1].type() != LayerTypeRelu)
            return false;
        if (src[index + 2].type() != LayerTypeEltwise || src[index + 2].src().size() != 2 ||
            src[index + 2].src()[0] != src[index + 1].name() ||
            src[index + 2].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        if (src[index + 3].type() != LayerTypeRelu)
            return false;
        if (!IsAdd(src[index + 4]) || src[index + 4].src()[0] != src[index + 2].name() || src[index + 4].src()[1] != src[index + 3].name())
            return false;
        if (InsideLink(src, index + 1, 5))
            return false;
        size_t tile = GetIndexByName(src, src[index + 2].src()[1]);
        if (tile == src.size() || tile < 2)
            return false;
        if (src[tile - 0].type() != LayerTypeTile || src[tile - 0].src()[0] != src[tile - 1].name())
            return false;
        if (src[tile - 1].type() != LayerTypeTile || src[tile - 1].src()[0] != src[tile - 2].name())
            return false;
        if (src[tile - 2].type() != LayerTypeConst)
            return false;
        LayerParam layer;
        layer.type() = LayerTypePrelu;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.weight().push_back(src[tile - 2].weight()[0]);
        dst.push_back(layer);
        if (buf.empty())
            buf = bin;
        const float* pSrc = GetWeight<float>(bin, layer.weight()[0]);
        float* pDst = GetWeight<float>(buf, layer.weight()[0]);
        for (size_t i = 0, n = layer.weight()[0].size() / 4; i < n; ++i)
            pDst[i] = -pSrc[i];
        //dst.erase(dst.begin() + tile - 2, dst.begin() + tile + 1);
        index += 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeSwish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (!IsMul(src[index]))
            return false;
        size_t dst0 = GetIndexByName(dst, src[index].src()[0]);
        size_t dst1 = GetIndexByName(dst, src[index].src()[1]);
        if (dst0 >= dst.size() || dst1 >= dst.size())
            return false;
        if (dst[dst0].type() != LayerTypeSigmoid && dst[dst1].type() != LayerTypeSigmoid)
            return false;
        LayerParam layer;
        layer.type() = LayerTypeSwish;
        layer.name() = src[index].name();
        layer.dst().push_back(layer.name());
        if (dst[dst0].type() == LayerTypeSigmoid)
        {
            size_t dst00 = GetIndexByName(dst, dst[dst0].src()[0]);
            if (dst00 >= dst.size() || dst00 != dst1)
                return false;
            size_t src0 = GetIndexByName(src, src[index].src()[0]);
            if (UserCount(src, src0) != 1)
                return false;
            layer.src().push_back(dst[dst0].src()[0]);
            dst.erase(dst.begin() + dst0, dst.begin() + dst0 + 1);
        }
        else
        {
            size_t dst10 = GetIndexByName(dst, dst[dst1].src()[0]);
            if (dst10 >= dst.size() || dst10 != dst0)
                return false;
            size_t src1 = GetIndexByName(src, src[index].src()[1]);
            if (UserCount(src, src1) != 1)
                return false;
            layer.src().push_back(dst[dst1].src()[0]);
            dst.erase(dst.begin() + dst1, dst.begin() + dst1 + 1);
        }
        dst.push_back(layer);
        return true;
    }
}