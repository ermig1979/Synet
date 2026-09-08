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
    bool MergeParallelScaleAndDepthwiseConvolution(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 3)
            return false;
        const LayerParam& scale = src[index + 0];
        LayerParam conv = src[index + 1];
        const LayerParam& add = src[index + 2];
        if (scale.type() != LayerTypeScale)
            return false;
        if (conv.type() != LayerTypeConvolution || conv.convolution().group() != conv.convolution().outputNum() ||
            conv.convolution().activationType() != ActivationFunctionTypeIdentity || conv.src() != scale.src() ||
            conv.convolution().biasTerm() != scale.scale().biasTerm() || conv.weight().size() != scale.weight().size() ||
            conv.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsAdd(add) || ((add.src()[0] != scale.dst()[0] || add.src()[1] != conv.dst()[0]) &&
            (add.src()[1] != scale.dst()[0] || add.src()[0] != conv.dst()[0])))
            return false;
        if (InsideLink(src, index + 0, 3))
            return false;

        if (buf.empty())
            buf = bin;
        size_t C = conv.convolution().outputNum();
        const float* pScale = GetWeight<float>(bin, scale.weight()[0]);
        float* pWeight = GetWeight<float>(buf, conv.weight()[0]) +
            (conv.convolution().kernel()[1] * conv.convolution().pad()[0] + conv.convolution().pad()[1]) * C;
        for (size_t c = 0; c < C; ++c)
            pWeight[c] += pScale[c];
        if (conv.convolution().biasTerm())
        {
            const float* pShift = GetWeight<float>(bin, scale.weight()[1]);
            float* pBias = GetWeight<float>(buf, conv.weight()[1]);
            for (size_t c = 0; c < C; ++c)
                pBias[c] += pShift[c];
        }
        conv.name() = add.name();
        conv.dst() = add.dst();
        dst.push_back(conv);
        index += 2;
        return true;
    }
}
