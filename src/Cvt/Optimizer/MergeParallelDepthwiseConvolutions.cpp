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
    bool MergeParallelDepthwiseConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 4)
            return false;
        LayerParam conv0 = src[index + 0];
        const LayerParam conv1 = src[index + 1];
        const LayerParam& add2 = src[index + 2];
        const LayerParam& add3 = src[index + 3];
        if (!IsDeptwiseConvolution(conv0, Shp(3, 3), Shp(1, 1), true, ActivationFunctionTypeIdentity) ||
            conv0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsDeptwiseConvolution(conv1, Shp(1, 1), Shp(1, 1), true, ActivationFunctionTypeIdentity) ||
            conv1.src() != conv0.src() || conv1.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsAdd(add2) || ((add2.src()[0] != conv0.dst()[0] || add2.src()[1] != conv1.dst()[0]) &&
            (add2.src()[1] != conv1.dst()[0] || add2.src()[0] != conv0.dst()[0])))
            return false;
        if (!IsAdd(add3) || ((add3.src()[0] != conv0.src()[0] || add3.src()[1] != add2.dst()[0]) &&
            (add3.src()[1] != conv0.src()[0] || add3.src()[0] != add2.dst()[0])))
            return false;
        if (InsideLink(src, index + 0, 4))
            return false;

        if (buf.empty())
            buf = bin;
        size_t C = conv0.convolution().outputNum();
        float* pWeight = GetWeight<float>(buf, conv0.weight()[0]) + 4 * C;
        float* pBias = GetWeight<float>(buf, conv0.weight()[1]);
        const float* pScale = GetWeight<float>(bin, conv1.weight()[0]);
        const float* pShift = GetWeight<float>(bin, conv1.weight()[1]);
        for (size_t c = 0; c < C; ++c)
        {
            pWeight[c] += pScale[c] + 1.0f;
            pBias[c] += pShift[c];
        }
        conv0.name() = add3.name();
        conv0.dst() = add3.dst();
        dst.push_back(conv0);
        index += 3;
        return true;
    }
}
