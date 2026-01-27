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

#include "Synet/Layers/Const/ConstLayer.h"

namespace Synet
{
    ConstLayer::ConstLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool ConstLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 0 || dst.size() != 1)
            SYNET_ERROR("ConstLayer supports only 1 output!");
        if (this->Weight().size() != 1)
            SYNET_ERROR("ConstLayer mast have 1 weight!");

        dst[0]->Share(this->Weight()[0]);
        dst[0]->SetConst(true);
        _const = true;
        return true;
    }

    void ConstLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}