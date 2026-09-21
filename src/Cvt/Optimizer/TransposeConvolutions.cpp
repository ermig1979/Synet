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
    bool TransposeConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        struct Utils : SynetUtils { using SynetUtils::PermutedToNchw; };
        size_t end = index;
        if (!Utils::PermutedToNchw(src, src[index].src(), true, false, false))
            return false;
        if (src[index].type() != LayerTypeConvolution || 
            src[index].weight()[0].format() != TensorFormatNchw || 
            UserCount(src, index) != 1)
            return false;
        for (size_t i = index + 1; i < src.size(); ++i)
        {
            if (src[i].type() != LayerTypeConvolution || 
                src[i].weight()[0].format() != TensorFormatNchw ||
                UserCount(src, i) != 1)
                break;
            end = i;
        }
        size_t count = end + 1 - index;
        if (!(count >= param.convToNhwc() || (count == 1 && src[index].convolution().group() != 1)))
            return false;

        LayerParam toNhwc;
        toNhwc.type() = LayerTypePermute;
        toNhwc.src().push_back(src[index].src()[0]);
        toNhwc.name() = src[index].src()[0] + "_permute_to_nhwc";
        toNhwc.dst().push_back(toNhwc.name());
        toNhwc.permute().order() = Shape({ 0, 2, 3, 1 });
        toNhwc.permute().format() = TensorFormatNhwc;
        dst.push_back(toNhwc);

        if (buf.empty())
            buf = bin;
        for (size_t i = index; i <= end; ++i)
        {
            dst.push_back(src[i]);
            if (i == index)
                dst.back().src()[0] = toNhwc.name();
            ReorderWeight(bin, Shape(), dst.back(), buf);
        }
        dst.back().name() = dst.back().name() + "_tmp";
        dst.back().dst()[0] = dst.back().name();

        LayerParam toNchw;
        toNchw.type() = LayerTypePermute;
        toNchw.src().push_back(dst.back().dst()[0]);
        toNchw.name() = dst.back().dst()[0] + "_permute_to_nchw";
        toNchw.dst().push_back(toNchw.name());
        toNchw.permute().order() = Shape({ 0, 3, 1, 2 });
        toNchw.permute().format() = TensorFormatNchw;
        dst.push_back(toNchw);

        index += end - index;
        changes.push_back(Change(src[end].dst()[0], toNchw.dst()[0]));
        return true;
    }
}
