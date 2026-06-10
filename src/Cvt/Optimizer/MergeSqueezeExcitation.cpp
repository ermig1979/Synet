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
        const LayerParam& pa = src[index + 0];
        const LayerParam& c0 = src[index + 1];
        const LayerParam& c1 = src[index + 2];
        if (pa.type() != LayerTypePooling || pa.pooling().method() != PoolingMethodTypeAverage)
            return false;
        if (c0.type() != LayerTypeConvolution || c0.convolution().kernel() != Shp(1, 1) || c0.src()[0] != pa.name())
            return false;
        if (c1.type() != LayerTypeConvolution || c1.convolution().kernel() != Shp(1, 1) || c1.src()[0] != c0.name())
            return false;
        size_t emi;
        if (c1.convolution().activationType() == ActivationFunctionTypeIdentity)
        {
            const LayerParam& si = src[index + 3];
            const LayerParam& em = src[index + 4];
            if (si.type() != LayerTypeSigmoid || si.src()[0] != c1.name())
                return false;
            if (em.type() != LayerTypeEltwise || em.eltwise().operation() != EltwiseOperationTypeProduct || em.src()[0] != pa.src()[0] || em.src()[1] != si.dst()[0])
                return false;
            emi = 4;
        }
        else if(c1.convolution().activationType() == ActivationFunctionTypeHardSigmoid)
        {
            const LayerParam& em = src[index + 3];
            if (em.type() != LayerTypeEltwise || em.eltwise().operation() != EltwiseOperationTypeProduct || 
                ((em.src()[0] != pa.src()[0] || em.src()[1] != c1.dst()[0]) && (em.src()[0] != c1.dst()[0] || em.src()[1] != pa.src()[0])))
                return false;
            emi = 3;
        }
        else
            return false;
        if (InsideLink(src, index + 1, emi))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeSqueezeExcitation;
        layer.name() = src[index + emi].name();
        layer.src().push_back(pa.src()[0]);
        for(size_t i = 0; i < c0.weight().size(); ++i)
            layer.weight().push_back(c0.weight()[i]);
        for (size_t i = 0; i < c1.weight().size(); ++i)
            layer.weight().push_back(c1.weight()[i]);
        layer.dst().push_back(src[index + emi].dst()[0]);
        layer.squeezeExcitation().biasTerm0() = c0.convolution().biasTerm();
        layer.squeezeExcitation().activationType() = c0.convolution().activationType();
        layer.squeezeExcitation().activationParam0() = c0.convolution().activationParam0();
        layer.squeezeExcitation().activationParam1() = c0.convolution().activationParam1();
        layer.squeezeExcitation().biasTerm1() = c1.convolution().biasTerm();
        layer.squeezeExcitation().hardSigmoid() = c1.convolution().activationType() == ActivationFunctionTypeHardSigmoid;
        dst.push_back(layer);
        index += emi;
        return true;
    }
}