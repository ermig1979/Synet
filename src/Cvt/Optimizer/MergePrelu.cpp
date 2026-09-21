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
    bool MergePrelu0(const LayerParams& src, size_t& index, const Bytes& bin, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        if (src[index + 0].type() != LayerTypeScale)
            return false;
        if (src[index + 1].type() != LayerTypeEltwise || src[index + 1].src().size() != 2 ||
            src[index + 1].src()[1] != src[index + 0].src()[0] || src[index + 1].src()[0] != src[index + 0].name() ||
            src[index + 1].eltwise().operation() != EltwiseOperationTypeMax)
            return false;
        if (InsideLink(src, index + 1, 1))
            return false;
        const float* scale = GetWeight<float>(bin, src[index].weight()[0]);
        for (size_t i = 0, n = src[index].weight()[0].size() / 4; i < n; ++i)
            if (scale[i] < -1.0f || scale[i] > 1.0f)
                return false;
        if (src[index + 0].weight().size() > 1)
        {
            const float* shift = GetWeight<float>(bin, src[index].weight()[1]);
            for (size_t i = 0, n = src[index].weight()[1].size() / 4; i < n; ++i)
                if (shift[i] != 0.0f)
                    return false;
        }
        LayerParam layer;
        layer.type() = LayerTypePrelu;
        layer.name() = src[index + 1].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.prelu().axis() = src[index + 0].scale().axis();
        layer.weight().push_back(src[index + 0].weight()[0]);
        dst.push_back(layer);
        index += 1;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergePrelu1(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypePower || src[index + 0].power().scale() != -1.0f)
            return false;
        if (src[index + 1].type() != LayerTypeRelu)
            return false;
        if (src[index + 2].type() != LayerTypeEltwise || src[index + 2].src().size() != 2 ||
            src[index + 2].src()[0] != src[index + 1].name() ||
            src[index + 2].eltwise().operation() != EltwiseOperationTypeProduct)
            return false;
        if (src[index + 3].type() != LayerTypeRelu)
            return false;
        if (!IsAdd(src[index + 4]) || src[index + 4].src()[0] != src[index + 2].name() || src[index + 4].src()[1] != src[index + 3].name())
            return false;
        if (InsideLink(src, index + 1, 5))
            return false;
        size_t tile = GetLayerIndex(src, src[index + 2].src()[1]);
        if (tile == src.size() || tile < 2)
            return false;
        if (src[tile - 0].type() != LayerTypeTile || src[tile - 0].src()[0] != src[tile - 1].name())
            return false;
        if (src[tile - 1].type() != LayerTypeTile || src[tile - 1].src()[0] != src[tile - 2].name())
            return false;
        if (src[tile - 2].type() != LayerTypeConst)
            return false;
        LayerParam layer;
        layer.type() = LayerTypePrelu;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.weight().push_back(src[tile - 2].weight()[0]);
        dst.push_back(layer);
        if (buf.empty())
            buf = bin;
        const float* pSrc = GetWeight<float>(bin, layer.weight()[0]);
        float* pDst = GetWeight<float>(buf, layer.weight()[0]);
        for (size_t i = 0, n = layer.weight()[0].size() / 4; i < n; ++i)
            pDst[i] = -pSrc[i];
        //dst.erase(dst.begin() + tile - 2, dst.begin() + tile + 1);
        index += 4;
        return true;
    }
}
