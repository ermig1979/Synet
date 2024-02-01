/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/FlattenLayer.h"

namespace Synet
{
    FlattenLayer::FlattenLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool FlattenLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("FlattenLayer supports only 1 input and 1 output!");
        if(src[0] == dst[0])
            SYNET_ERROR("FlattenLayer input and output must be different!");

        const FlattenParam & flatten = this->Param().flatten();
        size_t begAxis = src[0]->Index(flatten.axis());
        if (begAxis >= src[0]->Count())
            SYNET_ERROR("FlattenLayer parameter axis: " << begAxis << " has wrong value for input " << ToStr(src[0]->Shape()) << " !");

        size_t endAxis = src[0]->Index(flatten.endAxis());
        if (endAxis >= src[0]->Count())
            SYNET_ERROR("FlattenLayer parameter endAxis: " << endAxis << " has wrong value for input " << ToStr(src[0]->Shape()) << " !");

        Shape dstShape;
        dstShape.push_back(src[0]->Size(0, begAxis));
        dstShape.push_back(src[0]->Size(begAxis, endAxis + 1));
        if(endAxis < src[0]->Count() - 1)
            dstShape.push_back(src[0]->Size(endAxis + 1));
        dst[0]->ShareAs(*src[0], dstShape, src[0]->Format());
        _const = true;
        return true;
    }

    void FlattenLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}