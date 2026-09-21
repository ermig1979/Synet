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
    bool SimplifyInterp(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (index + 7 >= src.size())
            return false;
        if (src[index + 0].type() != LayerTypeMeta || src[index + 0].meta().type() != MetaTypeShape)
            return false;
        if (src[index + 1].type() != LayerTypeMeta || src[index + 1].meta().type() != MetaTypeConst)
            return false;
        if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeConst)
            return false;
        if (src[index + 3].type() != LayerTypeMeta || src[index + 3].meta().type() != MetaTypeConst)
            return false;
        if (src[index + 4].type() != LayerTypeMeta || src[index + 4].meta().type() != MetaTypeSlice)
            return false;
        if (src[index + 5].type() != LayerTypeMeta || src[index + 5].meta().type() != MetaTypeConst || src[index + 5].meta().alpha().shape() != Shp(2))
            return false;
        if (src[index + 6].type() != LayerTypeMeta || src[index + 6].meta().type() != MetaTypePack)
            return false;
        if (src[index + 7].type() != LayerTypeInterp || src[index + 7].src().size() != 2)
            return false;

        LayerParam layer = src[index + 7];
        layer.src().resize(1);
        layer.interp().height() = (int)src[index + 5].meta().alpha().i64()[0];
        layer.interp().width() = (int)src[index + 5].meta().alpha().i64()[1];
        dst.push_back(layer);

        index += 7;
        return true;
    }
}
