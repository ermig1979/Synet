/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layers/GridSampleLayer.h"

namespace Synet
{
    GridSampleLayer::GridSampleLayer(const LayerParam& param, Context* context)
        : Base(param, context)
    {
    }

    bool GridSampleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 && dst.size() != 1)
            SYNET_ERROR("GridSampleLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("GridSampleLayer don't support src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " !");
        if (src[1]->GetType() != TensorType32f)
            SYNET_ERROR("GridSampleLayer don't support src[1] type: " << Cpl::ToStr(src[1]->GetType()) << " !");

        Shape srcShape = src[0]->Shape();
        Shape gridShape = src[1]->Shape();
        int rank = (int)srcShape.size() - 2;
        if(srcShape.size() != gridShape.size() || rank < 1 || srcShape[0] != gridShape[0] || gridShape[rank + 1] != rank)
            SYNET_ERROR("GridSampleLayer has incompatible input shapes: src[0] " << ToStr(srcShape) << " and src[1] " << ToStr(gridShape) << " !");

        Shape dstShape = Shp(srcShape[0], srcShape[1]);
        for(int r = 1; r <= rank; ++r)
            dstShape.push_back(gridShape[r]);

        dst[0]->Reshape(src[0]->GetType(), dstShape, src[0]->Format());

        //SYNET_ERROR("GridSampleLayer is not implemented!");
        return true;
    }

    void GridSampleLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        assert(0);
    }
}