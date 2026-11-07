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
    bool MergeDynamicQuantizedInnerProduct(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 6)
            return false;
        if (src[index + 0].type() != LayerTypeDynamicQuantizeLinear || src[index + 0].dst().size() != 3)
            return false;
        if (src[index + 1].type() != LayerTypeEltwise || src[index + 1].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 1].src()[0] != src[index + 0].dst()[1] || src[index + 1].src().size() != 2)
            return false;
        size_t is11 = GetLayerIndex(src, src[index + 1].src()[1]);
        if (is11 >= src.size() || is11 >= dst.size() || src[is11].type() != LayerTypeConst)
            return false;
        if (src[index + 2].type() != LayerTypeMatMulInteger ||
            src[index + 2].src()[0] != src[index + 0].dst()[0] || src[index + 2].src()[1] != src[index + 0].dst()[2])
            return false;
        if (src[index + 3].type() != LayerTypeCast || src[index + 3].cast().type() != TensorType32f ||
            src[index + 3].src()[0] != src[index + 2].dst()[0])
            return false;
        if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[index + 4].src()[0] != src[index + 3].dst()[0] || src[index + 4].src()[1] != src[index + 1].dst()[0])
            return false;

        LayerParam layer;
        layer.type() = LayerTypeDynamicQuantizedInnerProduct;
        layer.name() = src[index + 2].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.weight() = src[index + 2].weight();
        layer.weight().push_back(src[is11].weight()[0]);
        layer.innerProduct().outputNum() = src[index + 2].weight()[0].dim()[1];
        layer.innerProduct().biasTerm() = false;
        if (src[index + 5].type() == LayerTypeEltwise && src[index + 5].eltwise().operation() == EltwiseOperationTypeSum &&
            src[index + 5].src().size() == 2 && src[index + 5].src()[0] == src[index + 4].dst()[0])
        {
            size_t is51 = GetLayerIndex(src, src[index + 5].src()[1]);
            if (is51 < src.size() && src[is51].type() == LayerTypeConst)
            {
                layer.weight().push_back(src[is51].weight()[0]);
                layer.innerProduct().biasTerm() = true;
                layer.dst() = src[index + 5].dst();
            }
        }
        dst.push_back(layer);
        index += layer.innerProduct().biasTerm() ? 5 : 4;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    static bool DynamicQuantizeLinearIsPrivate(const LayerParams& src, size_t dql, size_t matmul, size_t scaleMul)
    {
        const Strings& out = src[dql].dst();
        if (out.size() != 3)
            return false;
        size_t used0 = 0, used1 = 0, used2 = 0;
        for (size_t i = dql + 1; i < src.size(); ++i)
        {
            for (size_t s = 0; s < src[i].src().size(); ++s)
            {
                const String& name = src[i].src()[s];
                if (name == out[0])
                {
                    if (i != matmul)
                        return false;
                    used0++;
                }
                else if (name == out[1])
                {
                    if (i != scaleMul)
                        return false;
                    used1++;
                }
                else if (name == out[2])
                {
                    if (i != matmul)
                        return false;
                    used2++;
                }
            }
        }
        return used0 == 1 && used1 == 1 && used2 == 1;
    }

    static bool HasSingleConstSumUser(const LayerParams& src, size_t prod)
    {
        if (src[prod].dst().empty())
            return false;
        size_t users = 0;
        size_t user = src.size();
        for (size_t i = prod + 1; i < src.size(); ++i)
        {
            for (size_t s = 0; s < src[i].src().size(); ++s)
            {
                for (size_t d = 0; d < src[prod].dst().size(); ++d)
                {
                    if (src[i].src()[s] == src[prod].dst()[d])
                    {
                        users++;
                        user = i;
                    }
                }
            }
        }
        if (users != 1 || user >= src.size())
            return false;
        const LayerParam& add = src[user];
        if (add.type() != LayerTypeEltwise || add.eltwise().operation() != EltwiseOperationTypeSum || add.src().size() != 2)
            return false;
        if (add.src()[0] != src[prod].dst()[0])
            return false;
        size_t bias = GetLayerIndex(src, add.src()[1]);
        return bias < src.size() && src[bias].type() == LayerTypeConst && !src[bias].weight().empty();
    }

    bool MergeDynamicQuantizedInnerProductV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (index >= src.size())
            return false;
        const LayerParam& curr = src[index];
        if (curr.type() != LayerTypeEltwise || curr.src().size() != 2 || curr.dst().empty())
            return false;

        bool biasTerm = false;
        size_t prodS = src.size();
        size_t biasConstS = src.size();
        if (curr.eltwise().operation() == EltwiseOperationTypeSum)
        {
            biasConstS = GetLayerIndex(src, curr.src()[1]);
            if (biasConstS >= src.size() || src[biasConstS].type() != LayerTypeConst || src[biasConstS].weight().empty())
                return false;
            prodS = GetLayerIndex(src, curr.src()[0]);
            if (prodS >= src.size() || prodS >= index || src[prodS].dst().empty() || curr.src()[0] != src[prodS].dst()[0])
                return false;
            biasTerm = true;
        }
        else if (curr.eltwise().operation() == EltwiseOperationTypeProduct)
            prodS = index;
        else
            return false;

        const LayerParam& prod = src[prodS];
        if (prod.type() != LayerTypeEltwise || prod.eltwise().operation() != EltwiseOperationTypeProduct ||
            prod.src().size() != 2 || prod.dst().empty())
            return false;

        size_t castS = GetLayerIndex(src, prod.src()[0]);
        size_t castD = GetLayerIndex(dst, prod.src()[0]);
        if (castS >= src.size() || castD >= dst.size() || UserCount(src, castS) > 1)
            return false;
        if (src[castS].type() != LayerTypeCast || src[castS].cast().type() != TensorType32f ||
            src[castS].src().empty() || prod.src()[0] != src[castS].dst()[0])
            return false;

        size_t scaleMulS = GetLayerIndex(src, prod.src()[1]);
        size_t scaleMulD = GetLayerIndex(dst, prod.src()[1]);
        if (scaleMulS >= src.size() || scaleMulD >= dst.size() || UserCount(src, scaleMulS) > 1)
            return false;
        if (src[scaleMulS].type() != LayerTypeEltwise || src[scaleMulS].eltwise().operation() != EltwiseOperationTypeProduct ||
            src[scaleMulS].src().size() != 2 || src[scaleMulS].dst().empty() || prod.src()[1] != src[scaleMulS].dst()[0])
            return false;

        size_t matmulS = GetLayerIndex(src, src[castS].src()[0]);
        size_t matmulD = GetLayerIndex(dst, src[castS].src()[0]);
        if (matmulS >= src.size() || matmulD >= dst.size() || UserCount(src, matmulS) > 1)
            return false;
        if (src[matmulS].type() != LayerTypeMatMulInteger || src[matmulS].src().size() < 2 || src[matmulS].dst().empty() ||
            src[castS].src()[0] != src[matmulS].dst()[0] || src[matmulS].weight().empty() || src[matmulS].weight()[0].dim().size() < 2)
            return false;

        size_t scaleConstS = GetLayerIndex(src, src[scaleMulS].src()[1]);
        if (scaleConstS >= src.size() || src[scaleConstS].type() != LayerTypeConst || src[scaleConstS].weight().empty())
            return false;

        size_t dqlS = GetLayerIndex(src, src[scaleMulS].src()[0]);
        size_t dqlD = GetLayerIndex(dst, src[scaleMulS].src()[0]);
        if (dqlS >= src.size() || dqlD >= dst.size() || src[dqlS].src().empty())
            return false;
        if (src[dqlS].type() != LayerTypeDynamicQuantizeLinear || src[dqlS].dst().size() != 3)
            return false;
        if (src[scaleMulS].src()[0] != src[dqlS].dst()[1])
            return false;
        if (src[matmulS].src()[0] != src[dqlS].dst()[0] || src[matmulS].src()[1] != src[dqlS].dst()[2])
            return false;
        if (!DynamicQuantizeLinearIsPrivate(src, dqlS, matmulS, scaleMulS))
            return false;

        size_t prodD = dst.size();
        if (biasTerm)
        {
            if (UserCount(src, prodS) != 1)
                return false;
            prodD = GetLayerIndex(dst, prod.dst()[0]);
            if (prodD >= dst.size())
                return false;
        }
        else if (HasSingleConstSumUser(src, prodS))
            return false;

        Index removes;
        removes.push_back(dqlD);
        removes.push_back(scaleMulD);
        removes.push_back(matmulD);
        removes.push_back(castD);
        if (biasTerm)
            removes.push_back(prodD);
        std::sort(removes.begin(), removes.end(), [](const size_t& a, const size_t& b) { return a > b; });
        for (size_t i = 1; i < removes.size(); ++i)
        {
            if (removes[i] == removes[i - 1])
                return false;
        }
        for (size_t i = 0; i < removes.size(); ++i)
            dst.erase(dst.begin() + removes[i], dst.begin() + removes[i] + 1);

        LayerParam layer;
        layer.type() = LayerTypeDynamicQuantizedInnerProduct;
        layer.name() = src[matmulS].name();
        layer.src().push_back(src[dqlS].src()[0]);
        if (biasTerm)
            layer.dst() = curr.dst();
        else
            layer.dst().push_back(prod.dst()[0]);
        layer.weight() = src[matmulS].weight();
        layer.weight().push_back(src[scaleConstS].weight()[0]);
        layer.innerProduct().outputNum() = src[matmulS].weight()[0].dim()[1];
        layer.innerProduct().biasTerm() = false;
        if (biasTerm)
        {
            layer.weight().push_back(src[biasConstS].weight()[0]);
            layer.innerProduct().biasTerm() = true;
        }
        dst.push_back(layer);
        return true;
    }
}
