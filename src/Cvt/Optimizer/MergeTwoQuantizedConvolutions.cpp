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
    bool MergeTwoQuantizedConvolutions(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (!param.mergeQuantizedConvolutions())
            return false;
        if (src.size() < index + 2 || !param.mergeTwoConvolutions())
            return false;
        const LayerParam& l0 = src[index + 0];
        const Shape& k0 = l0.convolution().kernel();
        const LayerParam& l1 = src[index + 1];
        const Shape& k1 = l1.convolution().kernel();
        if (l0.type() != LayerTypeQuantizedConvolution || l1.type() != LayerTypeQuantizedConvolution || l1.src()[0] != l0.dst()[0])
            return false;
        if (l0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (InsideLink(src, index, 2))
            return false;
        if (l0.convolution().outputNum() > param.mergeTwoConvolutionsOutputNumMax() &&
            l1.convolution().outputNum() > param.mergeTwoConvolutionsOutputNumMax())
            return false;
        if (l0.convolution().group() != 1)
        {
            if (l0.convolution().outputNum() != l0.convolution().group() || l0.convolution().group() == 1)
                return false;
            if (k0.size() < 2 || (k0[0] != k0[1] || (k0[0] != 3 && k0[0] != 5 && k0[0] != 7)))
                return false;
            if (k1.size() < 2 || (k1[0] != k1[1] || (k1[0] != 1)) || l1.convolution().group() != 1)
                return false;
        }
        else
        {
            if (k0.size() < 2 || k0[0] != k0[1] || k0[0] != 1 || l0.convolution().group() != 1)
                return false;
            if (l1.convolution().outputNum() != l1.convolution().group() || l1.convolution().group() == 1)
                return false;
            if (k1.size() < 2 || (k1[0] != k1[1] || (k1[0] != 3 && k1[0] != 5 && k1[0] != 7)))
                return false;
            if (k0[0] != 1)
                return false;
        }
        LayerParam layer;
        layer.type() = LayerTypeQuantizedMergedConvolution;
        layer.name() = l1.name();
        layer.src() = l0.src();
        layer.dst().push_back(layer.name());
        for (size_t l = 0; l < 2; ++l)
        {
            for (size_t i = 0; i < src[index + l].weight().size(); ++i)
                layer.weight().push_back(src[index + l].weight()[i]);
            for (size_t i = 0; i < src[index + l].qSrc().size(); ++i)
                layer.qSrc().push_back(src[index + l].qSrc()[i]);
        }
        layer.qDst() = l1.qDst();
        layer.mergedConvolution().conv().push_back(l0.convolution());
        layer.mergedConvolution().conv().push_back(l1.convolution());
        index += 1;
        dst.push_back(layer);
        return true;
    }
}
