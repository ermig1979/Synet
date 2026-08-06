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
    bool MergeQuantizedSqueezeExcitation(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        const LayerParam& qpa = src[index + 0];
        const LayerParam& qc0 = src[index + 1];
        const LayerParam& qc1 = src[index + 2];
        const LayerParam& qhs = src[index + 3];
        const LayerParam& qm = src[index + 4];
        if (qpa.type() != LayerTypeQuantizedPooling || qpa.pooling().method() != PoolingMethodTypeAverage)
            return false;
        if (qc0.type() != LayerTypeQuantizedConvolution || qc0.convolution().kernel() != Shp(1, 1) || qc0.src()[0] != qpa.dst()[0])
            return false;
        if (qc1.type() != LayerTypeQuantizedConvolution || qc1.convolution().kernel() != Shp(1, 1) || qc1.src()[0] != qc0.dst()[0])
            return false;
        if (qhs.type() != LayerTypeQuantizedHardSigmoid || qhs.src()[0] != qc1.dst()[0])
            return false;
        if (qm.type() != LayerTypeQuantizedMul || qm.src()[0] != qhs.dst()[0] || qm.src()[1] != qpa.src()[0])
            return false;
        if (InsideLink(src, index + 1, 4))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedSqueezeExcitation;
        layer.name() = qm.name();
        layer.src().push_back(qpa.src()[0]);
        for (size_t i = 0; i < qc0.weight().size(); ++i)
            layer.weight().push_back(qc0.weight()[i]);
        for (size_t i = 0; i < qc1.weight().size(); ++i)
            layer.weight().push_back(qc1.weight()[i]);
        layer.dst().push_back(qm.dst()[0]);
        layer.squeezeExcitation().biasTerm0() = qc0.convolution().biasTerm();
        layer.squeezeExcitation().activationType() = qc0.convolution().activationType();
        layer.squeezeExcitation().activationParam0() = qc0.convolution().activationParam0();
        layer.squeezeExcitation().activationParam1() = qc0.convolution().activationParam1();
        layer.squeezeExcitation().biasTerm1() = qc1.convolution().biasTerm();
        layer.squeezeExcitation().hardSigmoid() = true;

        layer.qSrc().push_back(qpa.qSrc()[0]);
        for (size_t i = 0; i < qc0.qSrc().size(); ++i)
            layer.qSrc().push_back(qc0.qSrc()[i]);
        for (size_t i = 0; i < qc1.qSrc().size(); ++i)
            layer.qSrc().push_back(qc1.qSrc()[i]);
        layer.qSrc().push_back(qhs.qSrc()[0]);
        layer.qSrc().push_back(qm.qSrc()[0]);
        layer.qDst() = qm.qDst();
        dst.push_back(layer);
        index += 4;
        return true;
    }
}