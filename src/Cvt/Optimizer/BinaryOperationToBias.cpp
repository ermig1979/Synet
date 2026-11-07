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
    bool BinaryOperationToBias(const LayerParams & src, size_t & index, const Bytes& bin, Bytes& buf, LayerParams & dst)
    {
        if (src[index].type() != Synet::LayerTypeBinaryOperation ||
            src[index].binaryOperation().type() != BinaryOperationTypeSub)
            return false;
        const LayerParam* src1 = GetLayer(src, src[index].src()[1]);
        if (src1 == NULL || src1->type() != LayerTypeConst ||
            src1->weight()[0].type() != TensorType32f ||
            SignificantDimsCount(src1->weight()[0].dim()) != 1)
            return false;
        LayerParam layer;
        layer.type() = Synet::LayerTypeBias;
        layer.name() = src[index].name();
        layer.src() = src[index].src();
        layer.dst() = src[index].dst();
        layer.weight() = src1->weight();
        if (!CompactShape(layer.weight()[0].dim()))
            return false;
        const float* pSrc = GetWeight<float>(bin, layer.weight()[0]);
        if (buf.empty())
            buf = bin;
        if (WeightUserCount(src, layer.weight()[0]) > 1)
        {
            layer.weight()[0].offset() = buf.size();
            buf.resize(buf.size() + layer.weight()[0].size());
        }
        float* pDst = GetWeight<float>(buf, layer.weight()[0]);
        size_t size = TensorSize(layer.weight()[0].dim());
        for (size_t i = 0; i < size; ++i)
            pDst[i] = -pSrc[i];
        layer.src().resize(1);
        dst.push_back(layer);
        return true;
    }
}
