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
    bool MergeQuantizedShuffleV0(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 7)
            return false;
        if (src[index + 0].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 1].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 2].type() != LayerTypeConcat || src[index + 2].src().size() != 2)
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().shape().size() != 5)
            return false;
        if (src[index + 4].type() != LayerTypePermute)
            return false;
        if (src[index + 5].type() != LayerTypeReshape || src[index + 5].reshape().shape().size() != 4)
            return false;
        if (src[index + 6].type() != LayerTypeQuantizeLinear)
            return false;
        if (src[index + 7].type() != LayerTypeUnpack || src[index + 7].dst().size() != 2)
            return false;
        if (InsideLink(src, index, 6, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedShuffle;
        layer.name() = src[index + 0].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.src().push_back(src[index + 1].src()[0]);
        layer.qSrc().push_back(src[index + 0].quantize());
        layer.qSrc().push_back(src[index + 1].quantize());
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 7].dst()[0]);
        layer.dst().push_back(src[index + 7].dst()[1]);
        layer.qDst().push_back(src[index + 6].quantize());
        index += 7;
        dst.push_back(layer);
        return true;
    }
}
