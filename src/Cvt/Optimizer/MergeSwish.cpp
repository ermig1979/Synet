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
    bool MergeSwish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (!IsMul(src[index]))
            return false;
        if (src[index].src().size() != 2)
            return false;
        size_t dst0 = GetLayerIndex(dst, src[index].src()[0]);
        size_t dst1 = GetLayerIndex(dst, src[index].src()[1]);
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
            size_t dst00 = GetLayerIndex(dst, dst[dst0].src()[0]);
            if (dst00 >= dst.size() || dst00 != dst1)
                return false;
            size_t src0 = GetLayerIndex(src, src[index].src()[0]);
            if (UserCount(src, src0) != 1)
                return false;
            layer.src().push_back(dst[dst0].src()[0]);
            dst.erase(dst.begin() + dst0, dst.begin() + dst0 + 1);
        }
        else
        {
            size_t dst10 = GetLayerIndex(dst, dst[dst1].src()[0]);
            if (dst10 >= dst.size() || dst10 != dst0)
                return false;
            size_t src1 = GetLayerIndex(src, src[index].src()[1]);
            if (UserCount(src, src1) != 1)
                return false;
            layer.src().push_back(dst[dst1].src()[0]);
            dst.erase(dst.begin() + dst1, dst.begin() + dst1 + 1);
        }
        dst.push_back(layer);
        return true;
    }
}
