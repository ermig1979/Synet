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
    bool SkipTwoPermutes(const LayerParams& src, size_t& index, LayerParams& dst)
    {
        if (src.size() <= index + 1)
            return false;
        if (src[index].type() != LayerTypePermute)
            return false;
        size_t second = index + 1;
        for (; second < src.size(); ++second)
        {
            if (src[second].type() == LayerTypeMeta)
                continue;
            else if (src[second].type() == LayerTypeReshape)
                continue;
            else if (src[second].type() == LayerTypePermute)
                break;
            else
                return false;
        }

        bool skip = false;
        if ((src[index].permute().order() == Shp(0, 3, 1, 2)) && src[second].permute().order() == Shp(0, 2, 3, 1) && 
            src[index].permute().format() == TensorFormatNchw)
            skip = true;
        if ((src[index].permute().order() == Shp(0, 2, 1) || src[index].permute().order() == Shp(0, 3, 1, 2)) &&
            src[second].permute().order() == Shp(0, 2, 3, 1) && src[second].permute().format() == TensorFormatNhwc)
            skip = true;
        if (src[index].permute().order() == Shp(0, 3, 1, 2) && src[index].permute().format() == TensorFormatNchw && 
            (src[second].permute().order() == Shp(0, 2, 1) || src[second].permute().order() == Shp(0, 2, 3, 1)))
            skip = true;
        if (!skip)
            return false;

        dst.push_back(src[index]);
        dst.back().permute().skip() = true;
        for (size_t i = index + 1; i < second; ++i)
            dst.push_back(src[i]);
        dst.push_back(src[second]);
        dst.back().permute().skip() = true;
        index = second;
        return true;
    }
}
