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

#include "Synet/Layers/Legacy/SwitchLayer.h"

namespace Synet
{
    SwitchLayer::SwitchLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool SwitchLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 2)
            SYNET_ERROR("SwitchLayer supports only 2 inputs and 2 outputs!");
        if (src[1]->GetType() == TensorType32i)
            SYNET_ERROR("SwitchLayer src[1] must have INT32 type!");
        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        dst[1]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        this->UsePerfStat();
        _const = false;
        return true;
    }

    void SwitchLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        int pred = src[1]->Data<int32_t>()[0];
        if (pred)
            memcpy(dst[1]->RawData(), src[0]->RawData(), src[0]->RawSize());
        else
            memcpy(dst[0]->RawData(), src[0]->RawData(), src[0]->RawSize());
    }
}