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
    bool MergeShuffle0(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().shape().size() != 3)
            return false;
        if (src[index + 2].type() != LayerTypePermute)
            return false;
        if (src[index + 3].type() != LayerTypeUnpack || src[index + 3].dst().size() != 2)
            return false;
        if (src[index + 4].type() != LayerTypeReshape || 
            src[index + 4].reshape().shape().size() + src[index + 4].reshape().axis() != 4)
            return false;
        if (src[index + 5].type() != LayerTypeReshape || 
            src[index + 5].reshape().shape().size() + src[index + 5].reshape().axis() != 4)
            return false;
        if (InsideLink(src, index, 4, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeShuffle;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.shuffle().type() = 0;
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.dst().push_back(src[index + 5].dst()[0]);
        index += 5;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeShuffle1(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 4)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
            return false;
        if (src[index + 2].type() != LayerTypePermute)
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
            return false;
        if (src[index + 4].type() != LayerTypeUnpack || src[index + 4].dst().size() != 2)
            return false;
        if (InsideLink(src, index, 4, 0))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeShuffle;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.dst().push_back(src[index + 4].dst()[1]);
        index += 4;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeShuffle2(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (src.size() < index + 18)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        if (src[index + 1].type() != LayerTypeReshape)// || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
            return false;
        if (src[index + 2].type() != LayerTypePermute || src[index + 2].permute().order().size() != 5)
            return false;
        if (src[index + 3].type() != LayerTypeReshape)// || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
            return false;
        for (size_t i = 4; i < 14; ++i)
        {
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        }
        if (src[index + 14].type() != LayerTypeStridedSlice || src[index + 14].src().size() != 4)
            return false;
        for (size_t i = 15; i < 16; ++i)
        {
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        }
        if (src[index + 17].type() != LayerTypeStridedSlice || src[index + 17].src().size() != 4)
            return false;
        LayerParam layer;
        layer.type() = LayerTypeShuffle;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 14].dst()[0]);
        layer.dst().push_back(src[index + 17].dst()[0]);
        index += 17;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeShuffle3(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 43)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        for(size_t i = 1; i < 22; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 22].type() != LayerTypeReshape)
            return false;
        if (src[index + 23].type() != LayerTypePermute || src[index + 23].permute().order().size() != 5)
            return false;
        for (size_t i = 24; i < 28; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 28].type() != LayerTypeReshape)
            return false;
        for (size_t i = 29; i < 39; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 39].type() != LayerTypeStridedSlice || src[index + 39].src().size() != 4)
            return false;
        for (size_t i = 40; i < 42; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 42].type() != LayerTypeStridedSlice || src[index + 42].src().size() != 4)
            return false;
        LayerParam layer;
        layer.type() = LayerTypeShuffle;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 39].dst()[0]);
        layer.dst().push_back(src[index + 42].dst()[0]);
        index += 42;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeShuffle3cut(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 29)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        for (size_t i = 1; i < 22; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 22].type() != LayerTypeReshape)
            return false;
        if (src[index + 23].type() != LayerTypePermute || src[index + 23].permute().order().size() != 5)
            return false;
        for (size_t i = 24; i < 28; ++i)
            if (src[index + i].type() != LayerTypeMeta)
                return false;
        if (src[index + 28].type() != LayerTypeReshape)
            return false;
        LayerParam shuffle;
        shuffle.type() = LayerTypeShuffle;
        shuffle.name() = src[index + 0].name();
        shuffle.src() = src[index + 0].src();
        shuffle.shuffle().type() = 1;
        shuffle.dst().push_back(src[index + 28].dst()[0] + "_dst0");
        shuffle.dst().push_back(src[index + 28].dst()[0] + "_dst1");
        LayerParam concat;
        concat.type() = LayerTypeConcat;
        concat.name() = src[index + 28].name();
        concat.src() = shuffle.dst();
        concat.dst().push_back(src[index + 28].dst()[0]);
        concat.concat().axis() = isNhwc ? -1 : 1;
        index += 28;
        dst.push_back(shuffle);
        dst.push_back(concat);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeShuffle4(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
            return false;
        if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
            return false;
        if (src[index + 2].type() != LayerTypePermute)
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
            return false;
        if (src[index + 4].type() != LayerTypeStridedSlice || src[index + 4].src()[0] != src[index + 3].dst()[0])
            return false;
        if (src[index + 5].type() != LayerTypeStridedSlice || src[index + 5].src()[0] != src[index + 3].dst()[0])
            return false;
        if (InsideLink(src, index, 4, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeShuffle;
        layer.name() = src[index + 0].name();
        layer.src() = src[index + 0].src();
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 4].dst()[0]);
        layer.dst().push_back(src[index + 5].dst()[0]);
        index += 5;
        dst.push_back(layer);
        return true;
    }
}
