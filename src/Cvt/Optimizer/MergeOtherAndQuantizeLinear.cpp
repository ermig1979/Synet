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
    bool MergeOtherAndQuantizeLinear(const LayerParams& src, size_t index, QuantizationMethod method, LayerParams& dst, Changes& changes)
    {
        const LayerParam& ql = src[index];
        if (ql.type() != LayerTypeQuantizeLinear)
            return false;
        if (ql.quantize().weights())
            return false;
        size_t dst0 = GetLayerIndex(dst, ql.src()[0]);
        size_t src0 = GetLayerIndex(src, ql.src()[0]);
        if (dst0 >= dst.size() || src0 >= src.size())
            return false;
        LayerParam& other = dst[dst0];
        if (other.type() != LayerTypeQuantizedAdd &&
            other.type() != LayerTypeQuantizedConvolution &&
            other.type() != LayerTypeQuantizedInnerProduct &&
            other.type() != LayerTypeQuantizedPooling)
            return false;
        if (other.qDst().size())
            return false;
        if (UserCount(src, src0) != 1)
            return false;
        other.dst() = ql.dst();
        if (other.type() == LayerTypeQuantizedConvolution && (other.convolution().activationType() == ActivationFunctionTypeRelu ||
            (other.convolution().activationType() == ActivationFunctionTypeRestrictRange && other.convolution().activationParam0() == 0.0f)))
        {
            other.convolution().activationType() = ActivationFunctionTypeIdentity;
            other.convolution().activationParam1() = other.convolution().activationParam1.Default();
        }
        if (other.type() == LayerTypeQuantizedAdd && (other.activation().type() == ActivationFunctionTypeRelu ||
            (other.activation().type() == ActivationFunctionTypeRestrictRange && other.activation().param0() == 0.0f)))
        {
            other.activation().type() = ActivationFunctionTypeIdentity;
            other.activation().param1() = other.activation().param1.Default();
        }
        other.qDst().push_back(ql.quantize());
        return true;
    }
}
