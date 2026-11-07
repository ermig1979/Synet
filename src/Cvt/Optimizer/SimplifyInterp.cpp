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

    //--------------------------------------------------------------------------------------------------

    bool SimplifyInterpV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        Index removes;
        const LayerParam& interp = src[index];
        if (interp.type() != LayerTypeInterp || interp.src().size() != 2)
            return false;

        size_t pack0s = GetLayerIndex(src, interp.src()[1]);
        size_t pack0d = GetLayerIndex(dst, interp.src()[1]);
        if (pack0s >= src.size() || pack0d >= dst.size() || UserCount(src, pack0s) > 1)
            return false;
        if (src[pack0s].type() != LayerTypeMeta || src[pack0s].meta().type() != MetaTypePack || src[pack0s].src().size() != 2)
            return false;
        removes.push_back(pack0d);

        size_t slice0s = GetLayerIndex(src, src[pack0s].src()[0]);
        size_t slice0d = GetLayerIndex(dst, src[pack0s].src()[0]);
        if (slice0s >= src.size() || slice0d >= dst.size() || UserCount(src, slice0s) > 1)
            return false;
        if (src[slice0s].type() != LayerTypeMeta || src[slice0s].meta().type() != MetaTypeSlice || src[slice0s].src().size() != 4)
            return false;
        if (!IsMetaConst64i(src, src[slice0s].src()[1], Lng(0)))
            return false;
        if (!IsMetaConst64i(src, src[slice0s].src()[2], Lng(2)))
            return false;
        if (!IsMetaConst64i(src, src[slice0s].src()[3], Lng(0)))
            return false;
        removes.push_back(slice0d);

        size_t shape0s = GetLayerIndex(src, src[slice0s].src()[0]);
        size_t shape0d = GetLayerIndex(dst, src[slice0s].src()[0]);
        if (shape0s >= src.size() || shape0d >= dst.size() || UserCount(src, shape0s) > 1)
            return false;
        if (src[shape0s].type() != LayerTypeMeta || src[shape0s].meta().type() != MetaTypeShape || 
            src[shape0s].src().size() != 1 || src[shape0s].src()[0] != interp.src()[0])
            return false;
        removes.push_back(shape0d);

        size_t cast0s = GetLayerIndex(src, src[pack0s].src()[1]);
        size_t cast0d = GetLayerIndex(dst, src[pack0s].src()[1]);
        if (cast0s >= src.size() || cast0d >= dst.size() || UserCount(src, cast0s) > 1)
            return false;
        if (src[cast0s].type() != LayerTypeMeta || src[cast0s].meta().type() != MetaTypeCast ||
            src[cast0s].meta().alpha().type() != TensorType64i)
            return false;
        removes.push_back(cast0d);

        size_t pack1s = GetLayerIndex(src, src[cast0s].src()[0]);
        size_t pack1d = GetLayerIndex(dst, src[cast0s].src()[0]);
        if (pack1s >= src.size() || pack1d >= dst.size() || UserCount(src, pack1s) > 1)
            return false;
        if (src[pack1s].type() != LayerTypeMeta || src[pack1s].meta().type() != MetaTypePack || src[pack1s].src().size() != 2)
            return false;
        removes.push_back(pack1d);

        size_t expandDims0s = GetLayerIndex(src, src[pack1s].src()[0]);
        size_t expandDims0d = GetLayerIndex(dst, src[pack1s].src()[0]);
        if (expandDims0s >= src.size() || expandDims0d >= dst.size() || UserCount(src, expandDims0s) > 1)
            return false;
        if (src[expandDims0s].type() != LayerTypeMeta || src[expandDims0s].meta().type() != MetaTypeExpandDims)
            return false;
        removes.push_back(expandDims0d);

        size_t expandDims1s = GetLayerIndex(src, src[pack1s].src()[1]);
        size_t expandDims1d = GetLayerIndex(dst, src[pack1s].src()[1]);
        if (expandDims1s >= src.size() || expandDims1d >= dst.size() || UserCount(src, expandDims1s) > 1)
            return false;
        if (src[expandDims1s].type() != LayerTypeMeta || src[expandDims1s].meta().type() != MetaTypeExpandDims)
            return false;
        removes.push_back(expandDims1d);

        size_t gather0s = GetLayerIndex(src, src[expandDims0s].src()[0]);
        size_t gather0d = GetLayerIndex(dst, src[expandDims0s].src()[0]);
        if (gather0s >= src.size() || gather0d >= dst.size() || UserCount(src, gather0s) > 1)
            return false;
        if (src[gather0s].type() != LayerTypeMeta || src[gather0s].meta().type() != MetaTypeGather ||
            src[gather0s].src().size() != 2)
            return false;
        if (!IsMetaConst64i(src, src[gather0s].src()[1], Lng(2)))
            return false;
        removes.push_back(gather0d);

        size_t gather1s = GetLayerIndex(src, src[expandDims1s].src()[0]);
        size_t gather1d = GetLayerIndex(dst, src[expandDims1s].src()[0]);
        if (gather1s >= src.size() || gather1d >= dst.size() || UserCount(src, gather1s) > 1)
            return false;
        if (src[gather1s].type() != LayerTypeMeta || src[gather1s].meta().type() != MetaTypeGather ||
            src[gather1s].src().size() != 2)
            return false;
        if (!IsMetaConst64i(src, src[gather1s].src()[1], Lng(3)))
            return false;
        removes.push_back(gather1d);

        size_t shape1s = GetLayerIndex(src, src[gather0s].src()[0]);
        size_t shape1d = GetLayerIndex(dst, src[gather0s].src()[0]);
        if (shape1s >= src.size() || shape1d >= dst.size() || UserCount(src, shape1s) > 1)
            return false;
        if (src[shape1s].type() != LayerTypeMeta || src[shape1s].meta().type() != MetaTypeShape ||
            src[shape1s].src().size() != 1)
            return false;
        removes.push_back(shape1d);

        size_t shape2s = GetLayerIndex(src, src[gather1s].src()[0]);
        size_t shape2d = GetLayerIndex(dst, src[gather1s].src()[0]);
        if (shape2s >= src.size() || shape2d >= dst.size() || UserCount(src, shape2s) > 1)
            return false;
        if (src[shape2s].type() != LayerTypeMeta || src[shape1s].meta().type() != MetaTypeShape ||
            src[shape2s].src().size() != 1 || src[shape1s].src()[0] != src[shape2s].src()[0])
            return false;
        removes.push_back(shape2d);

        std::sort(removes.begin(), removes.end(), [](const size_t& a, const size_t& b) { return a > b; });

        for(size_t i = 0; i < removes.size(); ++i)
            dst.erase(dst.begin() + removes[i], dst.begin() + removes[i] + 1);

        LayerParam layer = interp;
        layer.src()[1] = src[shape1s].src()[0];
        layer.interp().useTensorSize() = true;
        dst.push_back(layer);

        return true;
    }
}
