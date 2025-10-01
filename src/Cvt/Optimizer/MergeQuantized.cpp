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
    bool MergeQuantizedPrelu(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        if (src[index + 0].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 1].type() != LayerTypePrelu)
            return false;
        if (src[index + 2].type() != LayerTypeQuantizeLinear)
            return false;
        if (InsideLink(src, index, 2, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedPrelu;
        layer.name() = src[index + 0].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.qSrc().push_back(src[index + 0].quantize());
        layer.prelu() = src[index + 1].prelu();
        layer.weight() = src[index + 1].weight();
        layer.dst().push_back(src[index + 2].dst()[0]);
        layer.qDst().push_back(src[index + 2].quantize());
        index += 2;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeQuantizedScale(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        if (src[index + 0].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 1].type() != LayerTypeScale)
            return false;
        if (src[index + 2].type() != LayerTypeQuantizeLinear)
            return false;
        if (InsideLink(src, index, 2, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedScale;
        layer.name() = src[index + 0].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.qSrc().push_back(src[index + 0].quantize());
        layer.scale() = src[index + 1].scale();
        layer.weight() = src[index + 1].weight();
        layer.dst().push_back(src[index + 2].dst()[0]);
        layer.qDst().push_back(src[index + 2].quantize());
        index += 2;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeQuantizedShuffleV0(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 7)
            return false;
        if (src[index + 0].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 1].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 2].type() != LayerTypeConcat || src[index + 2].src().size() != 2)
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().shape().size() != 5)
            return false;
        if (src[index + 4].type() != LayerTypePermute)
            return false;
        if (src[index + 5].type() != LayerTypeReshape || src[index + 5].reshape().shape().size() != 4)
            return false;
        if (src[index + 6].type() != LayerTypeQuantizeLinear)
            return false;
        if (src[index + 7].type() != LayerTypeUnpack || src[index + 7].dst().size() != 2)
            return false;
        if (InsideLink(src, index, 6, 1))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeQuantizedShuffle;
        layer.name() = src[index + 0].name();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.src().push_back(src[index + 1].src()[0]);
        layer.qSrc().push_back(src[index + 0].quantize());
        layer.qSrc().push_back(src[index + 1].quantize());
        layer.shuffle().type() = 1;
        layer.dst().push_back(src[index + 7].dst()[0]);
        layer.dst().push_back(src[index + 7].dst()[1]);
        layer.qDst().push_back(src[index + 6].quantize());
        index += 7;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeThreeQuantizedConvolutions(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (!param.mergeQuantizedConvolutions())
            return false;
        if (src.size() < index + 3)
            return false;
        const LayerParam& l0 = src[index + 0];
        const Shape& k0 = l0.convolution().kernel();
        const LayerParam& l1 = src[index + 1];
        const Shape& k1 = l1.convolution().kernel();
        const LayerParam& l2 = src[index + 2];
        const Shape& k2 = l2.convolution().kernel();
        if (l0.type() != LayerTypeQuantizedConvolution || l1.type() != LayerTypeQuantizedConvolution ||
            l2.type() != LayerTypeQuantizedConvolution || l1.src()[0] != l0.dst()[0] || l2.src()[0] != l1.dst()[0])
            return false;
        if (l0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (k0.size() < 2 || k0[0] != k0[1] || k0[0] != 1 || l0.convolution().group() != 1)
            return false;
        if (l1.convolution().outputNum() != l1.convolution().group() || l1.convolution().group() == 1)
            return false;
        if (k1.size() < 2 || k1[0] != k1[1] || (k1[0] != 3 && k1[0] != 5 && k1[0] != 7))
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
            if (ln.type() == LayerTypeQuantizedConvolution && l0.src()[0] == ln.dst()[0] &&
                ln.convolution().outputNum() == ln.convolution().group() && !InsideLink(src, index - 1, 4) &&
                l2.convolution().outputNum() >= l1.convolution().outputNum())
                return false;
        }
        if (src.size() > index + 3 && param.mergeTwoConvolutions())
        {
            const LayerParam& l3 = src[index + 3];
            if (l3.type() == LayerTypeQuantizedConvolution && l3.src()[0] == l2.dst()[0] &&
                l3.convolution().outputNum() == l3.convolution().group() && !InsideLink(src, index, 4) &&
                l2.convolution().outputNum() >= l1.convolution().outputNum())
                return false;
        }
        LayerParam layer;
        layer.type() = LayerTypeQuantizedMergedConvolution;
        layer.name() = l2.name();
        layer.src() = l0.src();
        layer.dst().push_back(layer.name());
        for (size_t l = 0; l < 3; ++l)
        {
            for (size_t i = 0; i < src[index + l].weight().size(); ++i)
                layer.weight().push_back(src[index + l].weight()[i]);
            for (size_t i = 0; i < src[index + l].qSrc().size(); ++i)
                layer.qSrc().push_back(src[index + l].qSrc()[i]);
        }
        layer.qDst() = l2.qDst();
        layer.mergedConvolution().conv().push_back(l0.convolution());
        layer.mergedConvolution().conv().push_back(l1.convolution());
        layer.mergedConvolution().conv().push_back(l2.convolution());

        index += 2;
        dst.push_back(layer);
        if (src.size() > index + 1)
        {
            const LayerParam& l3 = src[index + 1];
            if (l2.convolution().activationType() == ActivationFunctionTypeIdentity && l3.type() == LayerTypeQuantizedAdd &&
                ((l3.src()[0] == l0.src()[0] && l3.src()[1] == l2.dst()[0]) ||
                ((l3.src()[1] == l0.src()[0] && l3.src()[0] == l2.dst()[0]))) && !InsideLink(src, index - 2, 4))
            {
                dst.back().mergedConvolution().add() = l3.src()[0] == l0.src()[0] ? 2 : 1;
                dst.back().name() = l3.name();
                dst.back().dst()[0] = dst.back().name();
                dst.back().qSrc().push_back(l2.qDst()[0]);
                dst.back().qDst() = l3.qDst();
                index += 1;
            }
        }
        return true;
    }

    //--------------------------------------------------------------------------------------------------

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

    //--------------------------------------------------------------------------------------------------

    bool SkipUnnecessaryDequantizeQuantizeV0(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 3)
            return false;
        const LayerParam& dl = src[index + 0];
        LayerParam layer = src[index + 1];
        const LayerParam& ql = src[index + 2];
        if (dl.type() != LayerTypeDequantizeLinear)
            return false;
        if (layer.type() != LayerTypeFlatten &&
            (layer.type() != LayerTypePooling || layer.pooling().method() != PoolingMethodTypeMax))
            return false;
        if (ql.type() != LayerTypeQuantizeLinear)
            return false;
        if (dl.quantize().scale() != ql.quantize().scale() ||
            dl.quantize().zero() != ql.quantize().zero())
            return false;
        if (InsideLink(src, index, 3))
            return false;
        layer.src() = dl.src();
        layer.dst() = ql.dst();
        dst.push_back(layer);
        index += 2;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool SkipUnnecessaryDequantizeQuantizeV1(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 6)
            return false;
        if (src[index + 0].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 1].type() != LayerTypeDequantizeLinear)
            return false;
        if (src[index + 2].type() != LayerTypeConcat || src[index + 2].src().size() != 2)
            return false;
        if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().shape().size() != 5)
            return false;
        if (src[index + 4].type() != LayerTypePermute)
            return false;
        if (src[index + 5].type() != LayerTypeReshape || src[index + 5].reshape().shape().size() != 4)
            return false;
        if (src[index + 6].type() != LayerTypeQuantizeLinear)
            return false;
        if (InsideLink(src, index, 5, 1))
            return false;

        LayerParam layer;
        layer.type() = LayerTypeQuantizedConcat;
        layer.name() = src[index + 2].name();
        layer.concat() = src[index + 2].concat();
        layer.src().push_back(src[index + 0].src()[0]);
        layer.src().push_back(src[index + 1].src()[0]);
        layer.qSrc().push_back(src[index + 0].quantize());
        layer.qSrc().push_back(src[index + 1].quantize());
        layer.dst() = src[index + 2].dst();
        layer.qDst().push_back(src[index + 6].quantize());
        dst.push_back(layer);
        dst.push_back(src[index + 3]);
        dst.push_back(src[index + 4]);
        dst.push_back(src[index + 5]);
        dst.back().dst() = src[index + 6].dst();
        index += 6;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool SkipUnnecessaryDequantize(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2)
            return false;
        const LayerParam& dl = src[index + 0];
        LayerParam layer = src[index + 1];
        if (dl.type() != LayerTypeDequantizeLinear)
            return false;
        if (layer.type() != LayerTypeMeta && layer.meta().type() != MetaTypeShape)
            return false;
        if (InsideLink(src, index, 2))
            return false;
        layer.src() = dl.src();
        dst.push_back(layer);
        index += 1;
        return true;
    }
}