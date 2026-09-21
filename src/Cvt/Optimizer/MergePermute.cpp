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
    bool MergePermute(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 3)
            return false;
        const LayerParam & s0 = src[index + 0];
        LayerParam s1 = src[index + 1];
        LayerParam s2 = src[index + 2];
        if (s0.type() != LayerTypePermute || s0.permute().order() != Shp(0, 3, 1, 2) && s0.permute().format() != TensorFormatNchw)
            return false;
        if (s1.type() != LayerTypeReshape || s1.reshape().shape().size() != 5)
            return false;
        if (s2.type() != LayerTypePermute || s2.permute().order() != Shp(0, 1, 3, 4, 2))
            return false;
        if (InsideLink(src, index + 1, 3))
            return false;
        const Shape & s = s1.reshape().shape();
        s1.src() = s0.src();
        s1.reshape().shape() = Shp(s[0], s[3], s[4], s[1], s[2]);
        dst.push_back(s1);
        s2.permute().format() = TensorFormatNchw;
        s2.permute().order() = Shp(0, 3, 1, 2, 4);
        dst.push_back(s2);
        index += 2;
        return true;
    }
}
