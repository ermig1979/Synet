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
    bool MergeTwoDynamicConvolutions(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 4 || !param.mergeTwoConvolutions())
            return false;
        const LayerParam& c0 = src[index + 0];
        const Shape& k0 = c0.convolution().kernel();
        const Shape& d0 = c0.convolution().dilation();
        const LayerParam& i1 = src[index + 1];
        const LayerParam& r2 = src[index + 2];
        const LayerParam& c3 = src[index + 3];
        const Shape& k3 = c3.convolution().kernel();

        if (c0.type() != LayerTypeConvolution || c0.src().size() != 2 || c0.convolution().format() != TensorFormatNhwc)
            return false;
        if (c0.convolution().outputNum() != c0.convolution().group() || c0.convolution().group() == 1)
            return false;
        if (k0.size() < 2 || (k0[0] != k0[1] || (k0[0] != 3 && k0[0] != 5 && k0[0] != 7)))
            return false;
        if (d0.size() < 2 || d0[0] != 1 || d0[1] != 1)
            return false;
        if (c0.lowPrecision().bf16Type() != LowPrecisionTypeNone)
            return false;

        if (i1.type() != LayerTypeInnerProduct)
            return false;
        if (r2.type() != LayerTypeReshape || r2.src()[0] != i1.dst()[0])
            return false;

        if (c3.type() != LayerTypeConvolution || c3.src().size() != 2 || c3.convolution().format() != TensorFormatNhwc)
            return false;
        if (c3.src()[0] != c0.dst()[0] || c3.src()[1] != r2.dst()[0])
            return false;
        if (c3.lowPrecision().bf16Type() != LowPrecisionTypeNone)
            return false;

        if (c0.convolution().outputNum() > param.mergeTwoConvolutionsOutputNumMax() &&
            c3.convolution().outputNum() > param.mergeTwoConvolutionsOutputNumMax())
            return false;

        LayerParam mc;
        mc.type() = LayerTypeMergedConvolution;
        mc.name() = c3.name();
        mc.src() = c0.src();
        mc.src().push_back(c3.src()[1]);
        mc.dst().push_back(c3.name());
        for (size_t i = 0; i < c0.weight().size(); ++i)
            mc.weight().push_back(c0.weight()[i]);
        for (size_t i = 0; i < c3.weight().size(); ++i)
            mc.weight().push_back(c3.weight()[i]);
        mc.mergedConvolution().conv().push_back(c0.convolution());
        mc.mergedConvolution().conv().push_back(c3.convolution());

        index += 3;
        dst.push_back(i1);
        dst.push_back(r2);
        dst.push_back(mc);
        return true;
    }
}
