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

#include "Synet/Layers/Reshape/ReshapeLayer.h"

namespace Synet
{
    ReshapeLayer::ReshapeLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool ReshapeLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("ReshapeLayer supports 1 or 2 inputs and 1 output!");
        if (src[0] == dst[0])
            SYNET_ERROR("ReshapeLayer input and output can't be the same!");

        const ReshapeParam & param = this->Param().reshape();
        const Shape & shape = param.shape();
        Shape copyAxes;
        ptrdiff_t inferredAxis = -1;
        ptrdiff_t constantCount = 1;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
                copyAxes.push_back(i);
            else if (shape[i] == -1)
            {
                if(inferredAxis != -1)
                    SYNET_ERROR("ReshapeLayer: check reshape().shape() parameter!")
                inferredAxis = i;
            }
            else
                constantCount *= shape[i];
        }

        if (src.size() == 2)
        {
            if (src[1]->Count() != 1)
                SYNET_ERROR("ReshapeLayer src[1] must be 1D tensor!");
            bool trans = src[0]->Format() == TensorFormatNhwc;
            Shape shape;
            if (src[1]->GetType() == TensorType32i)
            {
                for (size_t i = 0; i < src[1]->Size(); ++i)
                    shape.push_back((size_t)src[1]->Data<int32_t>()[i]);
                if (!trans && shape.size() == 2 && shape[0] != -1)
                    shape = Shape({ shape[1], shape[0] });
                if (!trans && shape.size() == 4)
                    shape = Shape({ shape[0], shape[3], shape[1], shape[2] });
            }
            else if (src[1]->GetType() == TensorType64i)
            {
                for (size_t i = 0; i < src[1]->Size(); ++i)
                    shape.push_back((size_t)src[1]->Data<int64_t>()[i]);
            }
            else
                SYNET_ERROR("ReshapeLayer src[1] must be INT32 or INT64 type tensor!");
            size_t unknown = 0;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] == -1)
                    unknown++;
            }
            if(unknown > 1)
                SYNET_ERROR("ReshapeLayer src[1] has more then one negative elements!");
            if (unknown)
            {
                if (src[0]->Count() > 1 && shape.size() > 1 && shape[0] == 1 && Context().batchSize == src[0]->Axis(0))
                    shape[0] = src[0]->Axis(0);
                size_t known = 1, index = shape.size();
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (shape[i] != -1)
                        known *= shape[i];
                    else
                        index = i;
                }
                shape[index] = src[0]->Size()/known;
            }
            dst[0]->ShareAs(*src[0], shape, src[0]->Format());
        }
        else
        {
            const ReshapeParam & param = this->Param().reshape();
            const ptrdiff_t inputStartAxis = param.axis();
            const ptrdiff_t startAxis = (inputStartAxis >= 0) ? inputStartAxis : src[0]->Count() + inputStartAxis + 1;
            if (startAxis < 0 || startAxis > (ptrdiff_t)src[0]->Count())
                SYNET_ERROR("ReshapeLayer: check reshape().axis() parameter!");
            const ptrdiff_t numAxes = param.numAxes();
            if (numAxes < -1)
                SYNET_ERROR("ReshapeLayer: check reshape().numAxes() parameter!");
            const ptrdiff_t endAxis = (numAxes == -1) ? src[0]->Count() : (startAxis + numAxes);
            if(endAxis > (ptrdiff_t)src[0]->Count())
                SYNET_ERROR("ReshapeLayer: check reshape().axis() and reshape().numAxes() parameters!");
            const ptrdiff_t numAxesReplaced = endAxis - startAxis;
            const ptrdiff_t numAxesRetained = src[0]->Count() - numAxesReplaced;
            Shape shape(numAxesRetained + param.shape().size());
            int topShapeIndex = 0;
            for (ptrdiff_t i = 0; i < startAxis; ++i)
                shape[topShapeIndex++] = src[0]->Axis(i);
            for (size_t i = 0; i < param.shape().size(); ++i)
                shape[topShapeIndex++] = param.shape()[i];
            for (size_t i = endAxis; i < src[0]->Count(); ++i)
                shape[topShapeIndex++] = src[0]->Axis(i);
            if (topShapeIndex != shape.size())
                SYNET_ERROR("ReshapeLayer: check parameters!");
            for (size_t i = 0; i < copyAxes.size(); ++i)
            {
                if (src[0]->Count() < startAxis + copyAxes[i])
                    SYNET_ERROR("ReshapeLayer: check reshape().shape() parameter!");
                shape[startAxis + copyAxes[i]] = src[0]->Axis(startAxis + copyAxes[i]);
            }
            if (inferredAxis >= 0)
            {
                size_t explicitCount = constantCount;
                explicitCount *= src[0]->Size(0, startAxis);
                explicitCount *= src[0]->Size(endAxis);
                for (size_t i = 0; i < copyAxes.size(); ++i)
                    explicitCount *= shape[startAxis + copyAxes[i]];
                if(src[0]->Size() % explicitCount != 0)
                    SYNET_ERROR("ReshapeLayer: check parameters!");
                shape[startAxis + inferredAxis] = src[0]->Size() / explicitCount;
            }
            dst[0]->ShareAs(*src[0], shape, src[0]->Format());
        }
        _const = true;
        return true;
    }

    void ReshapeLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}