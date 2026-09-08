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
    bool MergeThreeConvolutions(const LayerParams& src, size_t& index, QuantizationMethod method, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 3 || (method != QuantizationMethodUnknown && !param.mergeInt8Convolutions()))
            return false;
        const LayerParam& l0 = src[index + 0];
        const Shape& k0 = l0.convolution().kernel();
        const LayerParam& l1 = src[index + 1];
        const Shape& k1 = l1.convolution().kernel();
        const Shape& d1 = l1.convolution().dilation();
        const LayerParam& l2 = src[index + 2];
        const Shape& k2 = l2.convolution().kernel();
        if (l0.type() != LayerTypeConvolution || l1.type() != LayerTypeConvolution ||
            l2.type() != LayerTypeConvolution || l1.src()[0] != l0.dst()[0] || l2.src()[0] != l1.dst()[0])
            return false;
        if (l0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (l0.weight()[0].dim()[2] > param.mergeConvolutionsInputNumMax())
            return false;
        if (k0.size() < 2 || (k0[0] != k0[1] || (k0[0] != 1 && k0[0] != 3)) || l0.convolution().group() != 1)
            return false;
        if (l1.convolution().outputNum() != l1.convolution().group() || l1.convolution().group() == 1)
            return false;
        if (k1.size() < 2 || (k1[0] != k1[1] || (k1[0] != 3 && k1[0] != 5 && k1[0] != 7)))
            return false;
        if (d1.size() < 2 || d1[0] != 1 || d1[1] != 1)
            return false;
        if (k2.size() < 2 || k2[0] != 1 || k2[1] != 1 || l2.convolution().group() != 1)
            return false;
        if (InsideLink(src, index, 3))
            return false;
        if (l1.convolution().outputNum() < l2.convolution().outputNum() * 0.75 && l2.convolution().outputNum() > 256)
            return false;
        if (index && param.mergeTwoConvolutions())
        {
            const LayerParam& ln = src[index - 1];
            if (ln.type() == LayerTypeConvolution && l0.src()[0] == ln.dst()[0] &&
                ln.convolution().outputNum() == ln.convolution().group() && !InsideLink(src, index - 1, 4) &&
                l2.convolution().outputNum() >= l1.convolution().outputNum())
                return false;
        }
        if (src.size() > index + 3 && param.mergeTwoConvolutions())
        {
            const LayerParam& l3 = src[index + 3];
            if (l3.type() == LayerTypeConvolution && l3.src()[0] == l2.dst()[0] &&
                l3.convolution().outputNum() == l3.convolution().group() && !InsideLink(src, index, 4) &&
                l2.convolution().outputNum() >= l1.convolution().outputNum())
                return false;
        }
        if (l0.convolution().quantizationLevel() != l2.convolution().quantizationLevel())// || l0.lowPrecision().bf16Type() != l2.lowPrecision().bf16Type())
        {
            return false;
        }
        LayerParam layer;
        layer.type() = LayerTypeMergedConvolution;
        layer.name() = l2.name();
        layer.src() = l0.src();
        layer.dst().push_back(layer.name());
        for (size_t l = 0; l < 3; ++l)
            for (size_t i = 0; i < src[index + l].weight().size(); ++i)
                layer.weight().push_back(src[index + l].weight()[i]);
        layer.mergedConvolution().conv().push_back(l0.convolution());
        layer.mergedConvolution().conv().push_back(l1.convolution());
        layer.mergedConvolution().conv().push_back(l2.convolution());
        if (layer.mergedConvolution().conv()[0].quantizationLevel() == TensorType8i ||
            layer.mergedConvolution().conv()[2].quantizationLevel() == TensorType8i)
        {
            layer.origin().push_back(l0.name());
            layer.origin().push_back(l1.name());
        }
        if (l0.lowPrecision().bf16Type() != LowPrecisionTypeNone && AtLeast2D(l0.convolution().kernel()) == Shp(1, 1))
            layer.lowPrecision().bf16Type() = l0.lowPrecision().bf16Type();
        if (l2.lowPrecision().bf16Type() != LowPrecisionTypeNone && AtLeast2D(l0.convolution().kernel()) == Shp(1, 1))
            layer.lowPrecision().bf16Type() = l2.lowPrecision().bf16Type();
        index += 2;
        dst.push_back(layer);
        if (src.size() > index + 1 && method == QuantizationMethodUnknown)// && l0.lowPrecision().bf16Type() == LowPrecisionTypeNone && l2.lowPrecision().bf16Type() == LowPrecisionTypeNone)
        {
            const LayerParam& l3 = src[index + 1];
            if (l2.convolution().activationType() == ActivationFunctionTypeIdentity && IsAdd(l3) && ((l3.src()[0] == l0.src()[0] && l3.src()[1] == l2.dst()[0]) ||
                ((l3.src()[1] == l0.src()[0] && l3.src()[0] == l2.dst()[0]))) && !InsideLink(src, index - 2, 4))
            {
                dst.back().mergedConvolution().add() = 1;
                dst.back().name() = l3.name();
                dst.back().dst()[0] = dst.back().name();
                index += 1;
                if (src.size() > index + 1)
                {
                    const LayerParam& l4 = src[index + 1];
                    if (l4.src().size() == 1 && l4.src()[0] == l3.name() && !InsideLink(src, index - 3, 5))
                    {
                        bool result = false;
                        if (l4.type() == LayerTypeRestrictRange)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypeRestrictRange;
                            dst.back().mergedConvolution().conv()[2].activationParam0() = l4.restrictRange().lower();
                            dst.back().mergedConvolution().conv()[2].activationParam1() = l4.restrictRange().upper();
                            result = true;
                        }
                        if (l4.type() == LayerTypeRelu)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = l4.relu().negativeSlope() == 0.0f ? ActivationFunctionTypeRelu : ActivationFunctionTypeLeakyRelu;
                            dst.back().mergedConvolution().conv()[2].activationParam0() = l4.relu().negativeSlope();
                            result = true;
                        }
                        if (l4.type() == LayerTypePrelu)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypePrelu;
                            dst.back().weight().push_back(l4.weight()[0]);
                            result = true;
                        }
                        if (l4.type() == LayerTypeElu)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypeElu;
                            dst.back().mergedConvolution().conv()[2].activationParam0() = l4.elu().alpha();
                            result = true;
                        }
                        if (l4.type() == LayerTypeHswish)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypeHswish;
                            dst.back().mergedConvolution().conv()[2].activationParam0() = l4.hswish().shift();
                            dst.back().mergedConvolution().conv()[2].activationParam1() = l4.hswish().scale();
                            result = true;
                        }
                        if (l4.type() == LayerTypeMish)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypeMish;
                            dst.back().mergedConvolution().conv()[2].activationParam0() = l4.softplus().threshold();
                            result = true;
                        }
                        if (l4.type() == LayerTypeGelu)
                        {
                            dst.back().mergedConvolution().conv()[2].activationType() = ActivationFunctionTypeGelu;
                            result = true;
                        }
                        if (result)
                        {
                            dst.back().name() = l4.name();
                            dst.back().dst()[0] = dst.back().name();
                            index += 1;
                        }
                    }
                }
            }
        }
        return true;
    }
}
