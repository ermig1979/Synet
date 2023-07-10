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

#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    EluLayer::EluLayer(const LayerParam& param, Context* context)
        : Base(param, context)
    {
    }

    void EluLayer::Reshape(const TensorPtrs & src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _alpha = this->Param().elu().alpha();
        if (src.size() != 1 && dst.size() != 1)
        {
            CPL_LOG_SS(Error, "EluLayer supports only 1 input and 1 output!");
            assert(0);
        }
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
        {
            CPL_LOG_SS(Error, "EluLayer supports only FP32 input and output!");
            assert(0);
        }
        this->UsePerfStat();
    }

    void EluLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Elu32f(src[0]->Data<float>(), _size, _alpha, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------
}