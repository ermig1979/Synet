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
    bool TransposeInnerProduct(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst)
    {
        const LayerParam& ip = src[index];
        if (ip.type() != LayerTypeInnerProduct || !ip.innerProduct().transposeB() || ip.weight().empty())
            return false;
        if (WeightUserCount(src, ip.weight()[0]) > 1)
            return false;
        const Shape & dim = ip.weight()[0].dim();
        if (buf.empty())
            buf = bin;
        dst.push_back(ip);
        dst.back().innerProduct().transposeB() = false;
        int axis = ip.innerProduct().axis(), dim0 = (int)dim[axis - 1], dim1 = (int)dim[axis];
        dst.back().weight()[0].dim() = Shp(dim1, dim0);
        const float* pSrc = GetWeight<float>(bin, ip.weight()[0]);
        float* pDst = GetWeight<float>(buf, ip.weight()[0]);
        for (int i = 0; i < dim0; ++i)
            for (int j = 0; j < dim1; ++j)
                pDst[j * dim0 + i] = pSrc[i * dim1 + j];
        return true;
    }
}
