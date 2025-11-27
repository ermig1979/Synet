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
        const LayerParam& l2 = src[index + 2];
        const Shape& k2 = l2.convolution().kernel();
        if (l0.type() != LayerTypeConvolution || l1.type() != LayerTypeConvolution ||
            l2.type() != LayerTypeConvolution || l1.src()[0] != l0.dst()[0] || l2.src()[0] != l1.dst()[0])
            return false;
        if (l0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (k0.size() < 2 || (k0[0] != k0[1] || (k0[0] != 1 && k0[0] != 3)) || l0.convolution().group() != 1)
            return false;
        if (l1.convolution().outputNum() != l1.convolution().group() || l1.convolution().group() == 1)
            return false;
        if (k1.size() < 2 || (k1[0] != k1[1] || (k1[0] != 3 && k1[0] != 5 && k1[0] != 7)))
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

    //--------------------------------------------------------------------------------------------------

    bool MergeTwoConvolutions(const LayerParams& src, size_t& index, QuantizationMethod method, const OptimizerParam& param, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 2 || !param.mergeTwoConvolutions() || (method != QuantizationMethodUnknown && !param.mergeInt8Convolutions()))
            return false;
        const LayerParam& l0 = src[index + 0];
        const Shape& k0 = l0.convolution().kernel();
        const LayerParam& l1 = src[index + 1];
        const Shape& k1 = l1.convolution().kernel();
        if (l0.type() != LayerTypeConvolution || l1.type() != LayerTypeConvolution || l1.src()[0] != l0.dst()[0])
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
            if (k0.size() < 2 || (k0[0] != k0[1] || (k0[0] != 1 && k0[0] != 3)) || l0.convolution().group() != 1)
                return false;
            if (l1.convolution().outputNum() != l1.convolution().group() || l1.convolution().group() == 1)
                return false;
            if (k1.size() < 2 || (k1[0] != k1[1] || (k1[0] != 3 && k1[0] != 5 && k1[0] != 7)))
                return false;
            if (l0.lowPrecision().bf16Type() == LowPrecisionTypeActive && k0[0] != 1)
                return false;
        }
        LayerParam layer;
        layer.type() = LayerTypeMergedConvolution;
        layer.name() = l1.name();
        layer.src() = l0.src();
        layer.dst().push_back(layer.name());
        for (size_t l = 0; l < 2; ++l)
            for (size_t i = 0; i < src[index + l].weight().size(); ++i)
                layer.weight().push_back(src[index + l].weight()[i]);
        layer.mergedConvolution().conv().push_back(l0.convolution());
        layer.mergedConvolution().conv().push_back(l1.convolution());
        if (layer.mergedConvolution().conv()[0].quantizationLevel() == TensorType8i ||
            layer.mergedConvolution().conv()[1].quantizationLevel() == TensorType8i)
            layer.origin().push_back(l0.name());
        if (l0.lowPrecision().bf16Type() == LowPrecisionTypeActive || l1.lowPrecision().bf16Type() == LowPrecisionTypeActive)
            layer.lowPrecision().bf16Type() = LowPrecisionTypeActive;
        index += 1;
        dst.push_back(layer);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeParallelConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        const LayerParam& l0 = src[index];
        size_t MinPart = 32, minPart = MinPart;
        Shape parts;
        for (; index + parts.size() < src.size();)
        {
            const LayerParam& l = src[index + parts.size()];
            if (l.type() != LayerTypeConvolution)
                break;
            const ConvolutionParam& c = l.convolution();
            if (c.group() != 1)
                break;
            if (parts.size())
            {
                if (l.src() != l0.src())
                    break;
                if (l.weight().size() != l0.weight().size())
                    break;
                if (l.weight()[0].format() != l0.weight()[0].format())
                    break;
                const ConvolutionParam& c0 = l0.convolution();
                if (c.kernel() != c0.kernel())
                    break;
                if (c.pad() != c0.pad())
                    break;
                if (c.stride() != c0.stride())
                    break;
                if (c.dilation() != c0.dilation())
                    break;
                if (c.biasTerm() != c0.biasTerm())
                    break;
                if (c.activationType() != c0.activationType())
                    break;
                if (c.activationParam0() != c0.activationParam0())
                    break;
                if (c.activationParam1() != c0.activationParam1())
                    break;
            }
            parts.push_back(c.outputNum());
            minPart = std::min(parts.back(), minPart);
        }
        if (parts.size() < 2 || minPart == MinPart)
            return false;

        LayerParam conv;
        conv.type() = LayerTypeConvolution;
        conv.name() = l0.name();
        conv.src() = l0.src();
        conv.convolution() = l0.convolution();
        conv.weight() = l0.weight();
        for (size_t p = 1; p < parts.size(); ++p)
        {
            const LayerParam& l = src[index + p];
            conv.convolution().outputNum() += l.convolution().outputNum();
            conv.name() = conv.name() + "_" + l.name();
            for (size_t w = 0; w < l0.weight().size(); ++w)
            {
                if (l.weight()[0].format() == TensorFormatNhwc)
                    conv.weight()[w].dim().back() += l.weight()[w].dim().back();
                else
                    conv.weight()[w].dim().front() += l.weight()[w].dim().front();
                conv.weight()[w].size() += l.weight()[w].size();
            }
        }
        conv.dst().push_back(conv.name());

        if (buf.empty())
            buf = bin;
        size_t newSize = buf.size();
        for (size_t w = 0; w < conv.weight().size(); ++w)
            newSize += conv.weight()[w].size();
        conv.weight()[0].offset() = buf.size();
        buf.resize(newSize);
        for (size_t w = 0; w < conv.weight().size(); ++w)
        {
            if (w)
                conv.weight()[w].offset() = conv.weight()[w - 1].offset() + conv.weight()[w - 1].size();
            const Shape& dim = conv.weight()[w].dim();
            std::vector<const float*> pSrc(parts.size());
            for (size_t p = 0; p < parts.size(); ++p)
                pSrc[p] = GetWeight<float>(bin, src[index + p].weight()[w]);
            float* pDst = GetWeight<float>(buf, conv.weight()[w]);
            if (l0.weight()[0].format() == TensorFormatNhwc && w == 0)
            {
                for (size_t o = 0, outer = dim[0] * dim[1] * dim[2]; o < outer; ++o)
                {
                    for (size_t p = 0; p < parts.size(); ++p)
                    {
                        for (size_t i = 0; i < parts[p]; ++i)
                            *pDst++ = *pSrc[p]++;
                    }
                }
            }
            else
            {
                for (size_t p = 0; p < parts.size(); ++p)
                {
                    memcpy(pDst, pSrc[p], src[index + p].weight()[w].size());
                    pDst += src[index + p].weight()[w].size() / sizeof(float);
                }
            }
        }

        LayerParam unpack;
        unpack.type() = LayerTypeUnpack;
        unpack.name() = conv.name() + "_unpack";
        unpack.src().push_back(conv.name());
        unpack.unpack().axis() = l0.weight()[0].format() == TensorFormatNhwc ? 3 : 1;
        unpack.unpack().parts() = parts;
        for (size_t i = 0; i < parts.size(); ++i)
            unpack.dst().push_back(src[index + i].dst()[0] + "_unpacked");

        dst.push_back(conv);
        dst.push_back(unpack);
        for (size_t i = 0; i < parts.size(); ++i)
        {
            LayerParam stub;
            stub.type() = LayerTypeStub;
            stub.name() = src[index + i].dst()[0];
            stub.dst().push_back(stub.name());
            stub.src().push_back(unpack.dst()[i]);
            dst.push_back(stub);
        }
        index += parts.size() - 1;

        return true;
    }

    bool MergeConcatedParallelConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        const LayerParam& l0 = src[index];
        Shape parts;
        for (; index + parts.size() < src.size();)
        {
            const LayerParam& l = src[index + parts.size()];
            if (l.type() != LayerTypeConvolution)
                break;
            const ConvolutionParam& c = l.convolution();
            if (c.group() != 1)
                break;
            if (parts.size())
            {
                if (l.src() != l0.src())
                    break;
                if (l.weight().size() != l0.weight().size())
                    break;
                if (l.weight()[0].format() != l0.weight()[0].format())
                    break;
                const ConvolutionParam& c0 = l0.convolution();
                if (c.kernel() != c0.kernel())
                    break;
                if (c.pad() != c0.pad())
                    break;
                if (c.stride() != c0.stride())
                    break;
                if (c.dilation() != c0.dilation())
                    break;
                if (c.biasTerm() != c0.biasTerm())
                    break;
                if (c.activationType() != c0.activationType())
                    break;
                if (c.activationParam0() != c0.activationParam0())
                    break;
                if (c.activationParam1() != c0.activationParam1())
                    break;
            }
            parts.push_back(c.outputNum());
        }
        if (parts.size() < 2)
            return false;
        const LayerParam& conc = src[index + parts.size()];
        if (conc.type() != LayerTypeConcat && conc.src().size() != parts.size())
            return false;
        for (size_t s = 0; s < conc.src().size(); ++s)
            if (conc.src()[s] != src[index + s].dst()[0])
                return false;

        LayerParam conv;
        conv.type() = LayerTypeConvolution;
        conv.name() = conc.name();
        conv.src() = l0.src();
        conv.convolution() = l0.convolution();
        conv.weight() = l0.weight();
        for (size_t p = 1; p < parts.size(); ++p)
        {
            const LayerParam& l = src[index + p];
            conv.convolution().outputNum() += l.convolution().outputNum();
            conv.name() = conv.name() + "_" + l.name();
            for (size_t w = 0; w < l0.weight().size(); ++w)
            {
                if (l.weight()[0].format() == TensorFormatNhwc)
                    conv.weight()[w].dim().back() += l.weight()[w].dim().back();
                else
                    conv.weight()[w].dim().front() += l.weight()[w].dim().front();
                conv.weight()[w].size() += l.weight()[w].size();
            }
        }
        conv.dst() = conc.dst();

        if (buf.empty())
            buf = bin;
        size_t newSize = buf.size();
        for (size_t w = 0; w < conv.weight().size(); ++w)
            newSize += conv.weight()[w].size();
        conv.weight()[0].offset() = buf.size();
        buf.resize(newSize);
        for (size_t w = 0; w < conv.weight().size(); ++w)
        {
            if (w)
                conv.weight()[w].offset() = conv.weight()[w - 1].offset() + conv.weight()[w - 1].size();
            const Shape& dim = conv.weight()[w].dim();
            std::vector<const float*> pSrc(parts.size());
            for (size_t p = 0; p < parts.size(); ++p)
                pSrc[p] = GetWeight<float>(bin, src[index + p].weight()[w]);
            float* pDst = GetWeight<float>(buf, conv.weight()[w]);
            if (l0.weight()[0].format() == TensorFormatNhwc && w == 0)
            {
                for (size_t o = 0, outer = dim[0] * dim[1] * dim[2]; o < outer; ++o)
                {
                    for (size_t p = 0; p < parts.size(); ++p)
                    {
                        for (size_t i = 0; i < parts[p]; ++i)
                            *pDst++ = *pSrc[p]++;
                    }
                }
            }
            else
            {
                for (size_t p = 0; p < parts.size(); ++p)
                {
                    memcpy(pDst, pSrc[p], src[index + p].weight()[w].size());
                    pDst += src[index + p].weight()[w].size() / sizeof(float);
                }
            }
        }

        dst.push_back(conv);

        index += parts.size();

        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeParallelDepthwiseConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 4)
            return false;
        LayerParam conv0 = src[index + 0];
        const LayerParam conv1 = src[index + 1];
        const LayerParam& add2 = src[index + 2];
        const LayerParam& add3 = src[index + 3];
        if (!IsDeptwiseConvolution(conv0, Shp(3, 3), Shp(1, 1), true, ActivationFunctionTypeIdentity) ||
            conv0.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsDeptwiseConvolution(conv1, Shp(1, 1), Shp(1, 1), true, ActivationFunctionTypeIdentity) ||
            conv1.src() != conv0.src() || conv1.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsAdd(add2) || ((add2.src()[0] != conv0.dst()[0] || add2.src()[1] != conv1.dst()[0]) &&
            (add2.src()[1] != conv1.dst()[0] || add2.src()[0] != conv0.dst()[0])))
            return false;
        if (!IsAdd(add3) || ((add3.src()[0] != conv0.src()[0] || add3.src()[1] != add2.dst()[0]) &&
            (add3.src()[1] != conv0.src()[0] || add3.src()[0] != add2.dst()[0])))
            return false;
        if (InsideLink(src, index + 0, 4))
            return false;

        if (buf.empty())
            buf = bin;
        size_t C = conv0.convolution().outputNum();
        float* pWeight = GetWeight<float>(buf, conv0.weight()[0]) + 4 * C;
        float* pBias = GetWeight<float>(buf, conv0.weight()[1]);
        const float* pScale = GetWeight<float>(bin, conv1.weight()[0]);
        const float* pShift = GetWeight<float>(bin, conv1.weight()[1]);
        for (size_t c = 0; c < C; ++c)
        {
            pWeight[c] += pScale[c] + 1.0f;
            pBias[c] += pShift[c];
        }
        conv0.name() = add3.name();
        conv0.dst() = add3.dst();
        dst.push_back(conv0);
        index += 3;
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeParallelScaleAndDepthwiseConvolution(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        if (src.size() < index + 3)
            return false;
        const LayerParam& scale = src[index + 0];
        LayerParam conv = src[index + 1];
        const LayerParam& add = src[index + 2];
        if (scale.type() != LayerTypeScale)
            return false;
        if (conv.type() != LayerTypeConvolution || conv.convolution().group() != conv.convolution().outputNum() ||
            conv.convolution().activationType() != ActivationFunctionTypeIdentity || conv.src() != scale.src() ||
            conv.convolution().biasTerm() != scale.scale().biasTerm() || conv.weight().size() != scale.weight().size() ||
            conv.weight()[0].format() != TensorFormatNhwc)
            return false;
        if (!IsAdd(add) || ((add.src()[0] != scale.dst()[0] || add.src()[1] != conv.dst()[0]) &&
            (add.src()[1] != scale.dst()[0] || add.src()[0] != conv.dst()[0])))
            return false;
        if (InsideLink(src, index + 0, 3))
            return false;

        if (buf.empty())
            buf = bin;
        size_t C = conv.convolution().outputNum();
        const float* pScale = GetWeight<float>(bin, scale.weight()[0]);
        float* pWeight = GetWeight<float>(buf, conv.weight()[0]) +
            (conv.convolution().kernel()[1] * conv.convolution().pad()[0] + conv.convolution().pad()[1]) * C;
        for (size_t c = 0; c < C; ++c)
            pWeight[c] += pScale[c];
        if (conv.convolution().biasTerm())
        {
            const float* pShift = GetWeight<float>(bin, scale.weight()[1]);
            float* pBias = GetWeight<float>(buf, conv.weight()[1]);
            for (size_t c = 0; c < C; ++c)
                pBias[c] += pShift[c];
        }
        conv.name() = add.name();
        conv.dst() = add.dst();
        dst.push_back(conv);
        index += 2;
        return true;
    }
}