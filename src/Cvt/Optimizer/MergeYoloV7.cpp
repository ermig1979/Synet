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
    bool MergeYoloV7(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (index == 0 || index + 4 >= src.size())
            return false;

        const LayerParam &c0 = src[index - 1];
        if (c0.type() != LayerTypeConcat || c0.src().size() != 3)
            return false;

        const LayerParam &ss0 = src[index + 0];
        if (ss0.type() != LayerTypeStridedSlice || ss0.src().size() != 1 || ss0.src()[0] != c0.dst()[0] ||
            ss0.stridedSlice().beginDims() != Lng(0) || ss0.stridedSlice().endDims() != Lng(4) ||
            ss0.stridedSlice().strideDims() != Lng(1) || (ss0.stridedSlice().axes() != Shp(2) && ss0.stridedSlice().axes() != Shp(1)))
            return false;

        const LayerParam &ss1 = src[index + 1];
        if (ss1.type() != LayerTypeStridedSlice || ss1.src().size() != 1 || ss1.src()[0] != c0.dst()[0] ||
            ss1.stridedSlice().beginDims() != Lng(4) || ss1.stridedSlice().endDims() != Lng(5) ||
            ss1.stridedSlice().strideDims() != Lng(1) || (ss1.stridedSlice().axes() != Shp(2) && ss1.stridedSlice().axes() != Shp(1)))
            return false;

        size_t start = index + 2;
        const LayerParam & ss2 = src[index + 2];
        if (ss2.type() == LayerTypeStridedSlice)
        {
            if (ss2.src().size() != 1 || ss2.src()[0] != c0.dst()[0] ||
                ss2.stridedSlice().beginDims() != Lng(5) || ss1.stridedSlice().strideDims() != Lng(1) ||
                (ss2.stridedSlice().axes() != Shp(2) && ss2.stridedSlice().axes() != Shp(1)))
                return false;

            const LayerParam& e0 = src[index + 3];
            if (e0.type() != LayerTypeEltwise || e0.eltwise().operation() != EltwiseOperationTypeProduct || e0.src().size() != 2 || 
                (e0.src()[0] != ss1.dst()[0] && e0.src()[0] != ss2.dst()[0]) || (e0.src()[1] != ss1.dst()[0] && e0.src()[1] != ss2.dst()[0]))
                return false;

            start = index + 4;
        }

        const LayerParam& ip0 = src[start + 0];
        if (ip0.type() != LayerTypeInnerProduct || ip0.src().size() != 1 || ip0.src()[0] != ss0.dst()[0] ||
            ip0.innerProduct().outputNum() != 4 || ip0.innerProduct().biasTerm() != false ||
            ip0.weight()[0].dim() != Shp(4, 4))
            return false;

        const LayerParam& r0 = src[start + 1];
        if (r0.type() != LayerTypeReduction || r0.reduction().axis() != Ints({ 2 }) || r0.reduction().type() != ReductionTypeMax ||
            r0.src().size() != 1 || r0.src()[0] != src[start - 1].dst()[0])
            return false;

        const LayerParam& am0 = src[start + 2];
        if (am0.type() != LayerTypeArgMax || am0.argMax().axis() != 2 || 
            am0.src().size() != 1 || am0.src()[0] != src[start - 1].dst()[0])
            return false;

        const LayerParam& c1 = src[start + 3];
        if (c1.type() != LayerTypeCast || c1.cast().type() != TensorType32f || 
            c1.src().size() != 1 || c1.src()[0] != am0.dst()[0])
            return false;

        const LayerParam& p0 = src[start + 4];
        if (p0.type() != LayerTypePower || p0.power().power() != 1.0f || p0.power().shift() != 0.0f ||
            p0.src().size() != 1 || p0.src()[0] != c1.dst()[0])
            return false;

        const LayerParam& e1 = src[start + 5];
        if (!IsAdd(e1) || (e1.src()[0] != ip0.dst()[0] && e1.src()[0] != p0.dst()[0]) || (e1.src()[1] != ip0.dst()[0] && e1.src()[1] != p0.dst()[0]))
            return false;

        const LayerParam& p1 = src[start + 6];
        if (p1.type() != LayerTypePermute || p1.permute().order() != Shp(0, 2, 1) ||
            p1.src().size() != 1 || p1.src()[0] != r0.dst()[0])
            return false;

        const LayerParam& nms0 = src[start + 7];
        if (nms0.type() != LayerTypeNonMaxSuppression || nms0.src().size() != 2 ||
            (nms0.src()[0] != e1.dst()[0] && nms0.src()[0] != p1.dst()[0]) || (nms0.src()[1] != e1.dst()[0] && nms0.src()[1] != p1.dst()[0]))
            return false;

        LayerParam yoloV7;
        yoloV7.type() = LayerTypeYoloV7;
        yoloV7.name() = src.back().dst()[0];
        yoloV7.src().push_back(c0.dst()[0]);
        yoloV7.dst().push_back(src.back().dst()[0]);
        yoloV7.yoloV7().maxOutputBoxesPerClass() = nms0.nonMaxSuppression().maxOutputBoxesPerClass();
        yoloV7.yoloV7().iouThreshold() = nms0.nonMaxSuppression().iouThreshold();
        yoloV7.yoloV7().scoreThreshold() = nms0.nonMaxSuppression().scoreThreshold();
        yoloV7.yoloV7().oneClass() = (start == index + 2);
        index += src.size() - 1 - index;
        dst.push_back(yoloV7);

        return true;
    }
}
