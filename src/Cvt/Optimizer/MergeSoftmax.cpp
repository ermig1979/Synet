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
    bool MergeSoftmax(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
    {
        if (index == 0 || src.size() < index + 5)
            return false;
        if (src[index + 0].type() != LayerTypeReduction || src[index + 0].reduction().type() != ReductionTypeMax ||
            src[index + 0].reduction().axis().size() != 1)
            return false;
        if (src[index + 1].type() != LayerTypeBinaryOperation || src[index + 1].binaryOperation().type() != BinaryOperationTypeSub ||
            src[index + 1].src()[0] != src[index + 0].src()[0] || src[index + 1].src()[1] != src[index + 0].name())
            return false;
        if (src[index + 2].type() != LayerTypeUnaryOperation || src[index + 2].unaryOperation().type() != UnaryOperationTypeExp ||
            src[index + 2].src()[0] != src[index + 1].name())
            return false;
        if (src[index + 3].type() != LayerTypeReduction || src[index + 3].reduction().type() != ReductionTypeSum ||
            src[index + 3].reduction().axis() != src[index + 0].reduction().axis() || src[index + 3].src()[0] != src[index + 2].name())
            return false;
        if (src[index + 4].type() != LayerTypeBinaryOperation || src[index + 4].binaryOperation().type() != BinaryOperationTypeDiv ||
            src[index + 4].src()[0] != src[index + 2].name() || src[index + 4].src()[1] != src[index + 3].name())
            return false;
        for (size_t i = index + 5; i < src.size(); ++i)
        {
            for (size_t j = 0; j < src[i].src().size(); ++j)
            {
                for (ptrdiff_t k = 0; k < 4; ++k)
                {
                    if (src[i].src()[j] == src[index + k].name())
                        return false;
                }
            }
        }
        LayerParam layer;
        layer.type() = LayerTypeSoftmax;
        layer.name() = src[index + 4].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.dst().push_back(layer.name());
        layer.softmax().axis() = src[index + 0].reduction().axis()[0];
        dst.push_back(layer);
        index += 4;
        return true;
    }
}
