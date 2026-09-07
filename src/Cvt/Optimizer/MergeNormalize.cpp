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
    struct Utils : SynetUtils { using SynetUtils::PermutedToNchw; };

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

    bool MergeNormalize(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 4)
            return false;
        if (src[index + 0].type() != LayerTypeReduction || src[index + 0].reduction().type() != ReductionTypeL2)
            return false;
        if (src[index + 1].type() != LayerTypeRestrictRange || src[index + 1].restrictRange().lower() >= 0.0000001f)
            return false;
        if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeShape)
            return false;
        if (src[index + 3].type() != LayerTypeTile)
            return false;
        if (src[index + 4].type() != LayerTypeBinaryOperation || src[index + 4].binaryOperation().type() != BinaryOperationTypeDiv)
            return false;

        LayerParam layer;
        layer.type() = LayerTypeNormalize;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.normalize().acrossSpatial() = true;
        layer.normalize().channelShared() = true;
        layer.normalize().eps() = 0;
        dst.push_back(layer);
        index += 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeNormalizeV2(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 8)
            return false;
        if (src[index + 0].type() != LayerTypeReduction || src[index + 0].reduction().type() != ReductionTypeMean)
            return false;
        if (src[index + 1].type() != LayerTypeBinaryOperation || src[index + 1].binaryOperation().type() != BinaryOperationTypeSub)
            return false;
        if (src[index + 2].type() != LayerTypePower || src[index + 2].power().power() != 2.0f)
            return false;
        if (src[index + 3].type() != LayerTypeReduction || src[index + 3].reduction().type() != ReductionTypeMean)
            return false;
        if (src[index + 4].type() != LayerTypePower || src[index + 4].power().power() != 1.0f || src[index + 4].power().scale() != 1.0f)
            return false;
        if (src[index + 5].type() != LayerTypeUnaryOperation || src[index + 5].unaryOperation().type() != UnaryOperationTypeSqrt)
            return false;
        if (src[index + 6].type() != LayerTypeBinaryOperation || src[index + 6].binaryOperation().type() != BinaryOperationTypeDiv)
            return false;
        const WeightParam* scale = GetEltwiseWeight(index + 7, src);
        if (scale == NULL || src[index + 7].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        const WeightParam* shift = GetEltwiseWeight(index + 8, src);
        if (shift == NULL || !IsAdd(src[index + 8]))
            return false;
        if (InsideLink(src, index + 1, 7))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeNormalize;
        layer.name() = src[index + 8].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.normalize().eps() = src[index + 4].power().shift();
        layer.normalize().version() = 2;
        layer.weight().push_back(*scale);
        layer.weight().push_back(*shift);
        if (isNhwc && !Utils::PermutedToNchw(src, layer.src(), false, false, false))
            layer.normalize().axis() = 1;
        else
            layer.normalize().axis() = -1;
        dst.push_back(layer);
        index += 8;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeNormalizeV4(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 9)
            return false;
        if (src[index + 0].type() != LayerTypeUnaryOperation || src[index + 0].unaryOperation().type() != UnaryOperationTypeAbs)
            return false;
        if (src[index + 1].type() != LayerTypePower || src[index + 1].power().power() != 2.0f)
            return false;
        if (src[index + 2].type() != LayerTypeReduction || src[index + 2].reduction().type() != ReductionTypeSum)
            return false;
        if (src[index + 3].type() != LayerTypePower || src[index + 3].power().power() != 0.5f)
            return false;
        if (src[index + 4].type() != LayerTypeReduction || src[index + 4].reduction().type() != ReductionTypeMean)
            return false;
        if (src[index + 5].type() != LayerTypePower || src[index + 5].power().power() != 1.0f || src[index + 5].power().scale() != 1.0f)
            return false;
        if (src[index + 6].type() != LayerTypeBinaryOperation || src[index + 6].binaryOperation().type() != BinaryOperationTypeDiv)
            return false;
        if (src[index + 7].type() != LayerTypeEltwise || src[index + 7].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        if (src[index + 8].type() != LayerTypeScale || !src[index + 8].scale().biasTerm())
            return false;
        if (!IsAdd(src[index + 9]))
            return false;
        if (InsideLink(src, index + 1, 8))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeNormalize;
        layer.name() = src[index + 9].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.normalize().eps() = src[index + 5].power().shift();
        layer.normalize().version() = 4;
        layer.weight() = src[index + 8].weight();
        if (isNhwc && !Utils::PermutedToNchw(src, layer.src(), false, false, false))
            layer.normalize().axis() = -1;
        else
            layer.normalize().axis() = 1;
        dst.push_back(layer);
        index += 9;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeNormalizeV5(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 4)
            return false;
        if (src[index + 0].type() != LayerTypeReshape || src[index + 0].reshape().shape().size() != 3)
            return false;
        if (src[index + 1].type() != LayerTypeNormalize || src[index + 1].normalize().version() != 3 || src[index + 1].src()[0] != src[index + 0].dst()[0])
            return false;
        if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeShape || src[index + 2].src()[0] != src[index + 0].src()[0])
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].src()[0] != src[index + 1].dst()[0] || src[index + 3].src()[1] != src[index + 2].dst()[0])
            return false;

        LayerParam layer;
        layer.type() = LayerTypeNormalize;
        layer.name() = src[index + 3].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.normalize() = src[index + 1].normalize();
        layer.normalize().version() = 5;
        layer.weight() = src[index + 1].weight();
        dst.push_back(layer);
        index += 3;
        return true;
    }
}
