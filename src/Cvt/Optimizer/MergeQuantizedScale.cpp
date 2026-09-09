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
    bool MergeQuantizedScale(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        size_t is2 = index;
        if (src[is2].type() != LayerTypeQuantizeLinear)
            return false;
        size_t is1 = GetLayerIndex(src, src[is2].src()[0]);
        size_t id1 = GetLayerIndex(dst, src[is2].src()[0]);
        if (is1 >= src.size() || id1 >= dst.size() || src[is1].type() != LayerTypeScale || UserCount(src, is1) > 1)
            return false;
        size_t is0 = GetLayerIndex(src, src[is1].src()[0]);
        size_t id0 = GetLayerIndex(dst, src[is1].src()[0]);
        if (is0 >= src.size() || id0 >= dst.size() || src[is0].type() != LayerTypeDequantizeLinear || UserCount(src, is0) > 1)
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedScale;
        layer.name() = src[is0].name();
        layer.src().push_back(src[is2].src()[0]);
        layer.qSrc().push_back(src[is0].quantize());
        layer.scale() = src[is1].scale();
        layer.weight() = src[is1].weight();
        layer.dst().push_back(src[is2].dst()[0]);
        layer.qDst().push_back(src[is2].quantize());
        dst.push_back(layer);
        dst[id0].type() = LayerTypeStub;
        dst[id1].type() = LayerTypeStub;
        return true;
    }
}
