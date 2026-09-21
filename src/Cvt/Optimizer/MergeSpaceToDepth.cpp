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
    bool MergeSpaceToDepth(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 9)
            return false;
        if (src[index + 0].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 1].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 2].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 3].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 4].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 5].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 6].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 7].type() != LayerTypeStridedSlice)
            return false;
        if (src[index + 8].type() != LayerTypeConcat)
            return false;
        if (InsideLink(src, index + 1, 8))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeSpaceToDepth;
        layer.name() = src[index + 8].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        dst.push_back(layer);
        index += 8;
        return true;
    }
}
