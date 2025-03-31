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
#include "Synet/Layers/BinaryOperationLayer.h"

namespace Synet
{
    BinaryOperationLayer::BinaryOperationLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool BinaryOperationLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("BinaryOperationLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("BinaryOperationLayer inpus have different types: " << Cpl::ToStr(src[0]->GetType()) << " != " << Cpl::ToStr(src[1]->GetType()) << " !");
        if (!IsCompatible(src[0]->Shape(), src[1]->Shape()))
            SYNET_ERROR("BinaryOperationLayer incompatible input shapes!");

        _dstShape = OutputShape(src[0]->Shape(), src[1]->Shape());
        if(_dstShape.size() > 4)
            SYNET_ERROR("BinaryOperationLayer too complicated shape!");
        _steps0 = SourceSteps(src[0]->Shape(), _dstShape);
        _steps1 = SourceSteps(src[1]->Shape(), _dstShape);
        _universalBinary = GetUniversalBinary(this->Param().binaryOperation().type(), src[0]->GetType(), _dstShape.size());
        if(_universalBinary == NULL)
            SYNET_ERROR("BinaryOperationLayer can't get universal handler!");

        dst[0]->Reshape(src[0]->GetType(), _dstShape, src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    int64_t BinaryOperationLayer::Flop() const
    {
        if (_const)
            return 0;
        return TensorSize(_dstShape);
    }

    void BinaryOperationLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _universalBinary(src[0]->RawData(), _steps0, src[1]->RawData(), _steps1, dst[0]->RawData(), _dstShape);
    }
}