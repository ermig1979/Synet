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

#include "Synet/Layers/ExpandDimsLayer.h"

namespace Synet
{
    ExpandDimsLayer::ExpandDimsLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType ExpandDimsLayer::LowPrecision(TensorType type) const
    {
        return LowPrecisionTypePassive;
    }

    bool ExpandDimsLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() > 2 || dst.size() != 1)
            SYNET_ERROR("ExpandDimsLayer supports 1 or 2 inputs and 1 output!");
        if (src[0] == dst[0])
            SYNET_ERROR("ExpandDimsLayer input and output can't be the same!");
        const ExpandDimsParam & param = this->Param().expandDims();
        const Ints & axes = param.axes();
        Shape shape;
        if (axes.empty())
        {
            ptrdiff_t axis = param.axis();
            if (axis < 0)
                axis += src[0]->Count();
            for (ptrdiff_t i = 0; i < axis; ++i)
                shape.push_back(src[0]->Axis(i));
            shape.push_back(1);
            for (size_t i = axis; i < src[0]->Count(); ++i)
                shape.push_back(src[0]->Axis(i));
        }
        else
        {
            shape.resize(src[0]->Count() + axes.size(), 1);
            for (size_t i = 0, s = 0; i < shape.size(); ++i)
            {
                bool insert = true;
                for (size_t a = 0; a < axes.size() && insert; ++a)
                {
                    if (axes[a] >= 0 ? axes[a] == i : axes[a] + shape.size() == i)
                        insert = false;
                }
                if (insert)
                    shape[i] = src[0]->Axis(s++);
            }
        }
        dst[0]->ShareAs(*src[0], shape, src[0]->Format());
        _const = true;
        return true;
    }

    void ExpandDimsLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}