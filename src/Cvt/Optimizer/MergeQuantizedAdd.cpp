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
    bool MergeQuantizedAdd(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        size_t is4 = index;
        if (src[is4].type() != LayerTypeQuantizeLinear)
            return false;
        size_t is3 = GetLayerIndex(src, src[is4].src()[0]);
        size_t id3 = GetLayerIndex(dst, src[is4].src()[0]);
        if (is3 >= src.size() || id3 >= dst.size() || UserCount(src, is3) > 1)
            return false;
        ActivationFunctionType actType = ActivationFunctionTypeIdentity;
        float actParam0 = 0.0f, actParam1 = 6.0f;
        size_t is2 = src.size(), id2 = dst.size();
        if (src[is3].type() == LayerTypeRelu)
        {
            is2 = GetLayerIndex(src, src[is3].src()[0]);
            id2 = GetLayerIndex(dst, src[is3].src()[0]);
            actType = ActivationFunctionTypeRelu;
        }
        else if(src[is3].type() == LayerTypeRestrictRange)
        {
            is2 = GetLayerIndex(src, src[is3].src()[0]);
            id2 = GetLayerIndex(dst, src[is3].src()[0]);
            actType = ActivationFunctionTypeRestrictRange;
            actParam0 = src[is3].restrictRange().lower();
            actParam1 = src[is3].restrictRange().upper();
        }
        else if (src[is3].type() == LayerTypeAdd)
        {
            is2 = is3;
            id2 = id3;
        }
        else
            return false;
        if (is2 >= src.size() || id2 >= dst.size() || src[is2].type() != LayerTypeAdd || UserCount(src, is2) > 1)
            return false;
        size_t is1 = GetLayerIndex(src, src[is2].src()[1]);
        size_t id1 = GetLayerIndex(dst, src[is2].src()[1]);
        if (is1 >= src.size() || id1 >= dst.size() || src[is1].type() != LayerTypeDequantizeLinear || UserCount(src, is1) > 1)
            return false;
        size_t is0 = GetLayerIndex(src, src[is2].src()[0]);
        size_t id0 = GetLayerIndex(dst, src[is2].src()[0]);
        if (is0 >= src.size() || id0 >= dst.size() || src[is0].type() != LayerTypeDequantizeLinear || UserCount(src, is0) > 1)
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedAdd;
        layer.name() = src[is2].name();
        layer.src().push_back(src[is2].src()[0]);
        layer.src().push_back(src[is2].src()[1]);
        layer.qSrc().push_back(src[is0].quantize());
        layer.qSrc().push_back(src[is1].quantize());
        layer.dst().push_back(src[is2].dst()[0]);
        layer.qDst().push_back(src[is4].quantize());
        dst[id0].type() = LayerTypeStub;
        dst[id1].type() = LayerTypeStub;
        if (actType == ActivationFunctionTypeRelu || (actType == ActivationFunctionTypeRestrictRange && actParam0 == 0.0f))
        {
            dst[id3].type() = LayerTypeStub;
            dst[id3].dst() = src[is4].dst();
        }
        else
            layer.dst() = src[is4].dst();
        dst[id2] = layer;
        return true;
    }
}
