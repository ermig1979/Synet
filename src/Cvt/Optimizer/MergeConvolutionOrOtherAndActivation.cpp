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
    bool MergeConvolutionOrOtherAndActivation(const LayerParams& src, size_t index, QuantizationMethod method, LayerParams& dst, Changes& changes)
    {
        const LayerParam& act = src[index];
        ActivationFunctionType type = ActivationFunctionTypeIdentity;
        float param0 = ConvolutionParam().activationParam0(), param1 = ConvolutionParam().activationParam1();
        if (act.type() == LayerTypeRestrictRange)
        {
            type = ActivationFunctionTypeRestrictRange;
            param0 = act.restrictRange().lower();
            param1 = act.restrictRange().upper();
        }
        if (act.type() == LayerTypeRelu)
        {
            type = act.relu().negativeSlope() == 0.0f ? ActivationFunctionTypeRelu : ActivationFunctionTypeLeakyRelu;
            param0 = act.relu().negativeSlope();
        }
        if (act.type() == LayerTypePrelu && method != QuantizationMethodIECompatible)
        {
            type = ActivationFunctionTypePrelu;
        }
        if (act.type() == LayerTypeElu)
        {
            type = ActivationFunctionTypeElu;
            param0 = act.elu().alpha();
        }
        if (act.type() == LayerTypeHswish)
        {
            type = ActivationFunctionTypeHswish;
            param0 = act.hswish().shift();
            param1 = act.hswish().scale();
        }
        if (act.type() == LayerTypeMish)
        {
            type = ActivationFunctionTypeMish;
            param0 = act.softplus().threshold();
        }
        if (act.type() == LayerTypeHardSigmoid)
        {
            type = ActivationFunctionTypeHardSigmoid;
            param0 = act.hardSigmoid().scale();
            param1 = act.hardSigmoid().shift();
        }
        if (act.type() == LayerTypeSwish)
        {
            type = ActivationFunctionTypeSwish;
            param0 = 1.0f;
        }
        if (act.type() == LayerTypeGelu)
        {
            type = ActivationFunctionTypeGelu;
        }
        if (type == ActivationFunctionTypeIdentity)
            return false;
        size_t dst0 = GetLayerIndex(dst, act.src()[0]);
        size_t src0 = GetLayerIndex(src, act.src()[0]);
        if (dst0 >= dst.size() || src0 >= src.size())
            return false;
        if (UserCount(src, src0) != 1)
            return false;
        const LayerParam& prev = dst[dst0];
        if (prev.type() == LayerTypeConvolution || prev.type() == LayerTypeDeconvolution || prev.type() == LayerTypeQuantizedConvolution)
        {
            LayerParam& conv = dst[dst0];
            if (conv.convolution().activationType() != ActivationFunctionTypeIdentity)
                return false;
            if (conv.convolution().quantizationLevel() == TensorType8i)
            {
                conv.origin().push_back(conv.name());
                conv.name() = act.name();
                conv.dst()[0] = act.name();
            }
            else
            {
                if (index == src.size() - 1)
                    conv.dst() = act.dst();
                else
                    changes.push_back(Change(act.name(), conv.name()));
            }
            conv.convolution().activationType() = type;
            conv.convolution().activationParam0() = param0;
            conv.convolution().activationParam1() = param1;
            if (act.weight().size())
                conv.weight().push_back(act.weight()[0]);
        }
        else if (prev.type() == LayerTypeInnerProduct)
        {
            LayerParam& ip = dst[dst0];
            if (ip.innerProduct().activationType() != ActivationFunctionTypeIdentity)
                return false;
            if (index == src.size() - 1)
                ip.dst() = act.dst();
            else
                changes.push_back(Change(act.name(), ip.name()));
            ip.innerProduct().activationType() = type;
            ip.innerProduct().activationParam0() = param0;
            ip.innerProduct().activationParam1() = param1;
            if (act.weight().size())
                ip.weight().push_back(act.weight()[0]);
        }
        else if (prev.type() == LayerTypeQuantizedAdd && (type == ActivationFunctionTypeRelu))
        {
            LayerParam& quant = dst[dst0];
            if (quant.activation().type() != ActivationFunctionTypeIdentity)
                return false;
            changes.push_back(Change(act.name(), quant.name()));
            quant.activation().type() = type;
            quant.activation().param0() = param0;
            quant.activation().param1() = param1;
        }
        else
            return false;
        return true;
    }
}
