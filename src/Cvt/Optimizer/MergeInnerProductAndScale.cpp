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
    bool MergeInnerProductAndScale(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (index == 0)
            return false;
        const LayerParam& ip = src[index - 1];
        const LayerParam& scale = src[index];
        if (ip.type() != LayerTypeInnerProduct || ip.innerProduct().biasTerm() || ip.innerProduct().transposeB())
            return false;
        if (scale.type() != LayerTypeScale || scale.src()[0] != ip.name())
            return false;
        if (InsideLink(src, index - 1, 2))
            return false;
        if (buf.empty())
            buf = bin;
        dst.back().name() = scale.name();
        dst.back().dst() = scale.dst();
        if (scale.scale().biasTerm())
        {
            dst.back().innerProduct().biasTerm() = true;
            dst.back().weight().push_back(scale.weight()[1]);
        }
        const float* pSrc = GetWeight<float>(bin, ip.weight()[0]);
        const float* pScale = GetWeight<float>(bin, scale.weight()[0]);
        float* pDst = GetWeight<float>(buf, ip.weight()[0]);
        const Shape& dim = ip.weight()[0].dim();
        for (size_t i = 0; i < dim[0]; ++i)
            for (size_t j = 0; j < dim[1]; ++j)
                pDst[i * dim[1] + j] = pSrc[i * dim[1] + j] * pScale[i];
        return true;
    }
}
