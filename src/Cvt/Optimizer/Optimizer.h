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

#pragma once

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"
#include "Cvt/Optimizer/Common.h"

namespace Synet
{
    class Optimizer : public SynetUtils
    {
    public:
        Optimizer(const OptimizerParam& param);

        bool Run(Synet::NetworkParam& network, Bytes& bin);

    private:
        const OptimizerParam & _param;

        bool OptimizeLayers(Synet::NetworkParam& network, Bytes& bin, int stage);

        bool ReduceTensorIteratorIO(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst)
        {
            const LayerParam & stt = src[index];
            if (stt.type() != LayerTypeTensorIterator || stt.src().size() < 3 || stt.tensorIterator().back().size() < 1)
                return false;
            size_t srcDupls = 0;
            for (size_t i = 2; i < stt.src().size(); ++i)
            {
                if (stt.src()[1] == stt.src()[i])
                    srcDupls++;
            }
            size_t backDupls = 0;
            for (size_t i = 1; i < stt.tensorIterator().back().size(); ++i)
            {
                if (stt.tensorIterator().back()[0].src() == stt.tensorIterator().back()[i].src())
                    backDupls++;
            }
            if (srcDupls == 0 || srcDupls != backDupls || srcDupls < stt.src().size() - 2)
                return false;
            dst.push_back(stt);
            LayerParam& dtt = dst.back();
            dtt.src().resize(2);
            String rem, iter;
            for (size_t i = 0; i < dtt.tensorIterator().input().size() && iter.empty(); ++i)
                if (dtt.tensorIterator().input()[i].axis() != -1)
                    iter = dtt.tensorIterator().input()[i].dst();
            for (size_t i = index + 1; i < src.size() && rem.empty(); ++i)
            {
                if (src[i].parent() != stt.name())
                    break;
                if (src[i].type() == LayerTypeInput && src[i].name() != iter)
                    rem = src[i].name();
            }
            StringSet del;
            std::vector<ConnectionParam> back, input;
            for (size_t i = 0; i < dtt.tensorIterator().input().size(); ++i)
            {
                ConnectionParam & p = dtt.tensorIterator().input()[i];
                if (p.dst() == rem || p.dst() == iter)
                {
                    p.port() = Synet::Min<int>(1, p.port());
                    input.push_back(p);
                }
                else
                    del.insert(p.dst());
            }
            dtt.tensorIterator().input().swap(input);
            for (size_t i = 0; i < stt.tensorIterator().back().size(); ++i)
            {
                if (del.find(dtt.tensorIterator().back()[i].dst()) == del.end())
                    back.push_back(dtt.tensorIterator().back()[i]);
            }
            dtt.tensorIterator().back().swap(back);
            for (size_t i = index + 1; i < src.size(); ++i)
            {
                if (src[i].parent() != stt.name())
                    break;
                if (src[i].type() != LayerTypeInput || del.find(src[i].name()) == del.end())
                    dst.push_back(src[i]);
                for (size_t j = 0; j < dst.back().src().size(); ++j)
                {
                    if (del.find(dst.back().src()[j]) != del.end())
                        dst.back().src()[j] = rem;
                }
                index++;
            }
            return true;
        }

        bool TransposeInnerProduct(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst)
        {
            const LayerParam& ip = src[index];
            if (ip.type() != LayerTypeInnerProduct || !ip.innerProduct().transposeB() || ip.weight().empty())
                return false;
            if (WeightUserCount(src, ip.weight()[0]) > 1)
                return false;
            const Shape & dim = ip.weight()[0].dim();
            if (buf.empty())
                buf = bin;
            dst.push_back(ip);
            dst.back().innerProduct().transposeB() = false;
            int axis = ip.innerProduct().axis(), dim0 = (int)dim[axis - 1], dim1 = (int)dim[axis];
            dst.back().weight()[0].dim() = Shp(dim1, dim0);
            const float* pSrc = GetWeight<float>(bin, ip.weight()[0]);
            float* pDst = GetWeight<float>(buf, ip.weight()[0]);
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j)
                    pDst[j * dim0 + i] = pSrc[i * dim1 + j];
            return true;
        }

        bool MergeCurrentAndBias(const LayerParams& src, size_t& index, Bytes& bin, LayerParams& dst, Changes& changes)
        {
            if (index == 0)
                return false;
            const LayerParam & current = src[index - 1];
            const LayerParam & bias = src[index];
            const WeightParam* weight = GetEltwiseWeight(index, src);
            if (bias.type() == LayerTypeBias)
                weight = &bias.weight()[0];
            if(weight == NULL)
                return false;
            if (!(bias.src()[0] == current.name() || (bias.src().size() == 2 && bias.src()[1] == current.name())))
                return false;
            if (InsideLink(src, index - 1, 2))
                return false;
            switch (current.type())
            {
            case LayerTypeConvolution:
                if (current.convolution().biasTerm() || current.convolution().outputNum() != weight->dim()[0])
                    return false;
                dst.back().convolution().biasTerm() = true;
                break;
            case LayerTypeInnerProduct:
                if (current.innerProduct().biasTerm() || current.src().size() != 1 || current.innerProduct().outputNum() != weight->dim()[0])
                    return false;
                dst.back().innerProduct().biasTerm() = true;
                break;
            case LayerTypePower:
                if (current.power().power() != 1.0f || current.power().shift() != 0.0f || bias.type() != LayerTypeBias)
                    return false;
                dst.back().type() = LayerTypeScale;
                dst.back().scale().biasTerm() = true;
                dst.back().weight().push_back(*weight);
                dst.back().weight()[0].offset() = bin.size();
                for (size_t i = 0; i < dst.back().weight()[0].dim()[0]; ++i)
                    PushBack<float>(bin, current.power().scale());
                dst.back().power().scale() = 1.0f;
                break;
            case LayerTypeScale:
                if (current.scale().biasTerm())
                    return false;
                dst.back().scale().biasTerm() = true;
                break;
                default:
                    return false;
            }
            dst.back().name() = bias.name();
            dst.back().dst() = bias.dst();
            dst.back().weight().push_back(*weight);
            return true;
        }

        bool MergePowerAndScaleAndPower(const LayerParams& src, size_t& index, Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
        {
            bool pre = false, scale = false, post = false;
            if (src.size() > index + 0 && src[index + 0].type() == LayerTypePower && src[index + 0].power().power() == 1.0f)
                pre = true;
            if (src.size() > index + 1 && src[index + 1].type() == LayerTypeScale && (pre ? src[index + 1].src()[0] == src[index + 0].name() : true) && src[index + 1].scale().biasTerm())
                scale = true;
            if (src.size() > index + 2 && src[index + 2].type() == LayerTypePower && src[index + 2].power().power() == 1.0f && src[index + 2].src()[0] == src[index + 1].name())
                post = true;
            if (!(scale && (pre || post)))
                return false;
            if (InsideLink(src, index + (pre ? 0 : 1), 1 + (pre ? 1 : 0) + (post ? 1 : 0), 0, LayerTypes({ LayerTypePriorBox, LayerTypePriorBoxClustered, LayerTypeMeta })))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeScale;
            layer.name() = src[index + 1].name();
            layer.src().push_back(pre ? src[index + 0].src()[0] : src[index + 1].src()[0]);
            layer.dst().push_back(post ? src[index + 2].dst()[0] : src[index + 1].dst()[0]);
            layer.scale() = src[index + 1].scale();
            layer.weight() = src[index + 1].weight();
            float preScale = pre ? src[index + 0].power().scale() : 1.0f;
            float preBias = pre ? src[index + 0].power().shift() : 0.0f;
            float postScale = post ? src[index + 2].power().scale() : 1.0f;
            float postBias = post ? src[index + 2].power().shift() : 0.0f;
            if (buf.empty())
                buf = bin;
            float* pScale = GetWeight<float>(buf, layer.weight()[0]);
            float* pShift = GetWeight<float>(buf, layer.weight()[1]);
            size_t size = TensorSize(layer.weight()[0].dim());
            for (size_t i = 0; i < size; ++i)
            {
                pShift[i] = (preBias * pScale[i] + pShift[i]) * postScale + postBias;
                pScale[i] = preScale * pScale[i] * postScale;
            }
            if (pre)
                changes.push_back(Change(src[index + 0].dst()[0], layer.dst()[0]));
            else
                dst.push_back(src[index + 0]);
            index += 1 + (post ? 1 : 0);
            dst.push_back(layer);
            return true;
        }

        bool MergeBiasAndScale(const LayerParams& src, size_t& index, Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
        {
            if (index == 0)
                return false;
            const LayerParam & bias = src[index - 1];
            const LayerParam & scale = src[index];
            if (bias.type() != LayerTypeBias)
                return false;
            if (scale.type() != LayerTypeScale || scale.scale().biasTerm() || scale.src()[0] != bias.dst()[0])
                return false;
            if (InsideLink(src, index - 1, 2))
                return false;
            dst.back().type() = LayerTypeScale;
            dst.back().name() = scale.name();
            dst.back().dst() = scale.dst();
            dst.back().scale().biasTerm() = true;
            dst.back().weight().push_back(scale.weight()[0]);
            if (buf.empty())
                buf = bin;
            const float* pSrcScale = GetWeight<float>(bin, dst.back().weight()[1]);
            const float* pSrcBias = GetWeight<float>(bin, dst.back().weight()[0]);
            float* pDstScale = GetWeight<float>(buf, dst.back().weight()[0]);
            float* pDstBias = GetWeight<float>(buf, dst.back().weight()[1]);
            size_t size = TensorSize(dst.back().weight()[0].dim());
            for (size_t i = 0; i < size; ++i)
            {
                pDstScale[i] = pSrcScale[i];
                pDstBias[i] = pSrcScale[i] * pSrcBias[i];
            }
            return true;
        }

        bool MergeConvolutionAndPower(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
        {
            if (index == 0)
                return false;
            const LayerParam& conv = src[index - 1];
            const LayerParam& power = src[index];
            if (conv.type() != LayerTypeConvolution || 
                conv.convolution().activationType() != ActivationFunctionTypeIdentity)
                return false;
            if (power.type() != LayerTypePower || power.src()[0] != conv.name() ||
                power.power().power() != 1.0f || power.power().shift() != 0.0f)
                return false;
            if (InsideLink(src, index - 1, 2))
                return false;
            if (conv.weight()[0].format() != TensorFormatNhwc)
                return false;
            if (buf.empty())
                buf = bin;
            dst.back().name() = power.name();
            dst.back().dst() = power.dst();
            float scale = power.power().scale();
            for (size_t w = 0; w < conv.weight().size(); ++w)
            {
                const float* pSrc = GetWeight<float>(bin, conv.weight()[w]);
                float* pDst = GetWeight<float>(buf, conv.weight()[w]);
                for (size_t i = 0, n = conv.weight()[w].size() / 4; i < n; ++i)
                    pDst[i] = pSrc[i] * scale;
            }
            return true;
        }

        bool MergeInnerProductAndPower(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
        {
            if (index == 0)
                return false;
            const LayerParam& ip = src[index - 1];
            const LayerParam& power = src[index];
            if (ip.type() != LayerTypeInnerProduct || ip.src().size() != 1)
                return false;
            if (power.type() != LayerTypePower || power.src()[0] != ip.name() ||
                power.power().power() != 1.0f || power.power().shift() != 0.0f)
                return false;
            if (InsideLink(src, index - 1, 2))
                return false;
            if (buf.empty())
                buf = bin;
            dst.back().name() = power.name();
            dst.back().dst() = power.dst();
            float scale = power.power().scale();
            for (size_t w = 0; w < ip.weight().size(); ++w)
            {
                const float* pSrc = GetWeight<float>(bin, ip.weight()[w]);
                float* pDst = GetWeight<float>(buf, ip.weight()[w]);
                for (size_t i = 0, n = ip.weight()[w].size() / 4; i < n; ++i)
                    pDst[i] = pSrc[i] * scale;
            }
            return true;
        }

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

        bool MergeShuffle0(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 5)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().shape().size() != 3)
                return false;
            if (src[index + 2].type() != LayerTypePermute)
                return false;
            if (src[index + 3].type() != LayerTypeUnpack || src[index + 3].dst().size() != 2)
                return false;
            if (src[index + 4].type() != LayerTypeReshape || 
                src[index + 4].reshape().shape().size() + src[index + 4].reshape().axis() != 4)
                return false;
            if (src[index + 5].type() != LayerTypeReshape || 
                src[index + 5].reshape().shape().size() + src[index + 5].reshape().axis() != 4)
                return false;
            if (InsideLink(src, index, 4, 1))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeShuffle;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.shuffle().type() = 0;
            layer.dst().push_back(src[index + 4].dst()[0]);
            layer.dst().push_back(src[index + 5].dst()[0]);
            index += 5;
            dst.push_back(layer);
            return true;
        }

        bool MergeShuffle1(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 4)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
                return false;
            if (src[index + 2].type() != LayerTypePermute)
                return false;
            if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
                return false;
            if (src[index + 4].type() != LayerTypeUnpack || src[index + 4].dst().size() != 2)
                return false;
            if (InsideLink(src, index, 4, 0))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeShuffle;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.shuffle().type() = 1;
            layer.dst().push_back(src[index + 4].dst()[0]);
            layer.dst().push_back(src[index + 4].dst()[1]);
            index += 4;
            dst.push_back(layer);
            return true;
        }

        bool MergeShuffle2(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 18)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            if (src[index + 1].type() != LayerTypeReshape)// || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
                return false;
            if (src[index + 2].type() != LayerTypePermute || src[index + 2].permute().order().size() != 5)
                return false;
            if (src[index + 3].type() != LayerTypeReshape)// || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
                return false;
            for (size_t i = 4; i < 14; ++i)
            {
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            }
            if (src[index + 14].type() != LayerTypeStridedSlice || src[index + 14].src().size() != 4)
                return false;
            for (size_t i = 15; i < 16; ++i)
            {
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            }
            if (src[index + 17].type() != LayerTypeStridedSlice || src[index + 17].src().size() != 4)
                return false;
            LayerParam layer;
            layer.type() = LayerTypeShuffle;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.shuffle().type() = 1;
            layer.dst().push_back(src[index + 14].dst()[0]);
            layer.dst().push_back(src[index + 17].dst()[0]);
            index += 17;
            dst.push_back(layer);
            return true;
        }

        bool MergeShuffle3(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 43)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            for(size_t i = 1; i < 22; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 22].type() != LayerTypeReshape)
                return false;
            if (src[index + 23].type() != LayerTypePermute || src[index + 23].permute().order().size() != 5)
                return false;
            for (size_t i = 24; i < 28; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 28].type() != LayerTypeReshape)
                return false;
            for (size_t i = 29; i < 39; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 39].type() != LayerTypeStridedSlice || src[index + 39].src().size() != 4)
                return false;
            for (size_t i = 40; i < 42; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 42].type() != LayerTypeStridedSlice || src[index + 42].src().size() != 4)
                return false;
            LayerParam layer;
            layer.type() = LayerTypeShuffle;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.shuffle().type() = 1;
            layer.dst().push_back(src[index + 39].dst()[0]);
            layer.dst().push_back(src[index + 42].dst()[0]);
            index += 42;
            dst.push_back(layer);
            return true;
        }

        bool MergeShuffle3cut(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 29)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            for (size_t i = 1; i < 22; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 22].type() != LayerTypeReshape)
                return false;
            if (src[index + 23].type() != LayerTypePermute || src[index + 23].permute().order().size() != 5)
                return false;
            for (size_t i = 24; i < 28; ++i)
                if (src[index + i].type() != LayerTypeMeta)
                    return false;
            if (src[index + 28].type() != LayerTypeReshape)
                return false;
            LayerParam shuffle;
            shuffle.type() = LayerTypeShuffle;
            shuffle.name() = src[index + 0].name();
            shuffle.src() = src[index + 0].src();
            shuffle.shuffle().type() = 1;
            shuffle.dst().push_back(src[index + 28].dst()[0] + "_dst0");
            shuffle.dst().push_back(src[index + 28].dst()[0] + "_dst1");
            LayerParam concat;
            concat.type() = LayerTypeConcat;
            concat.name() = src[index + 28].name();
            concat.src() = shuffle.dst();
            concat.dst().push_back(src[index + 28].dst()[0]);
            concat.concat().axis() = isNhwc ? -1 : 1;
            index += 28;
            dst.push_back(shuffle);
            dst.push_back(concat);
            return true;
        }

        bool MergeShuffle4(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 5)
                return false;
            if (src[index + 0].type() != LayerTypeConcat || src[index + 0].src().size() != 2)
                return false;
            if (src[index + 1].type() != LayerTypeReshape || src[index + 1].reshape().axis() + src[index + 1].reshape().shape().size() != 5)
                return false;
            if (src[index + 2].type() != LayerTypePermute)
                return false;
            if (src[index + 3].type() != LayerTypeReshape || src[index + 3].reshape().axis() + src[index + 3].reshape().shape().size() != 4)
                return false;
            if (src[index + 4].type() != LayerTypeStridedSlice || src[index + 4].src()[0] != src[index + 3].dst()[0])
                return false;
            if (src[index + 5].type() != LayerTypeStridedSlice || src[index + 5].src()[0] != src[index + 3].dst()[0])
                return false;
            if (InsideLink(src, index, 4, 1))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeShuffle;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.shuffle().type() = 1;
            layer.dst().push_back(src[index + 4].dst()[0]);
            layer.dst().push_back(src[index + 5].dst()[0]);
            index += 5;
            dst.push_back(layer);
            return true;
        }

        bool MergeSqueezeExcitation(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() <= index + 4)
                return false;
            if (src[index + 0].type() != LayerTypePooling || src[index + 0].pooling().method() != PoolingMethodTypeAverage)
                return false;
            if (src[index + 1].type() != LayerTypeConvolution || src[index + 1].convolution().kernel() != Shp(1, 1) || 
                src[index + 1].convolution().biasTerm() || src[index + 1].src()[0] != src[index + 0].name() || 
                src[index + 1].convolution().activationType() != ActivationFunctionTypeRelu)
                return false;
            if (src[index + 2].type() != LayerTypeConvolution || src[index + 2].convolution().kernel() != Shp(1, 1) ||
                src[index + 2].convolution().biasTerm() || src[index + 2].src()[0] != src[index + 1].name())
                return false;
            if (src[index + 3].type() != LayerTypeSigmoid || src[index + 3].src()[0] != src[index + 2].name())
                return false;
            if (src[index + 4].type() != LayerTypeEltwise || src[index + 4].eltwise().operation() != EltwiseOperationTypeProduct ||
                src[index + 4].src()[0] != src[index + 0].src()[0] || src[index + 4].src()[1] != src[index + 3].dst()[0])
                return false;
            if (InsideLink(src, index + 1, 4))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeSqueezeExcitation;
            layer.name() = src[index + 4].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.weight().push_back(src[index + 1].weight()[0]);
            layer.weight().push_back(src[index + 2].weight()[0]);
            layer.dst().push_back(src[index + 4].dst()[0]);
            dst.push_back(layer);
            index += 4;
            return true;
        }

        bool MergePermute(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 3)
                return false;
            const LayerParam & s0 = src[index + 0];
            LayerParam s1 = src[index + 1];
            LayerParam s2 = src[index + 2];
            if (s0.type() != LayerTypePermute || s0.permute().order() != Shp(0, 3, 1, 2) && s0.permute().format() != TensorFormatNchw)
                return false;
            if (s1.type() != LayerTypeReshape || s1.reshape().shape().size() != 5)
                return false;
            if (s2.type() != LayerTypePermute || s2.permute().order() != Shp(0, 1, 3, 4, 2))
                return false;
            if (InsideLink(src, index + 1, 3))
                return false;
            const Shape & s = s1.reshape().shape();
            s1.src() = s0.src();
            s1.reshape().shape() = Shp(s[0], s[3], s[4], s[1], s[2]);
            dst.push_back(s1);
            s2.permute().format() = TensorFormatNchw;
            s2.permute().order() = Shp(0, 3, 1, 2, 4);
            dst.push_back(s2);
            index += 2;
            return true;
        }

        bool MergePooling(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 5)
                return false;
            if (src[index + 0].type() != LayerTypeReshape)
                return false;
            if (src[index + 1].type() != LayerTypePooling || src[index + 1].src()[0] != src[index + 0].name() || src[index + 1].pooling().kernel()[1] != 1)
                return false;
            if (src[index + 2].type() != LayerTypeReshape || src[index + 2].src()[0] != src[index + 1].name())
                return false;
            if (src[index + 3].type() != LayerTypeReshape || src[index + 3].src()[0] != src[index + 2].name())
                return false;
            if (src[index + 4].type() != LayerTypePooling || src[index + 4].src()[0] != src[index + 3].name() || src[index + 4].pooling().kernel()[1] != 1)
                return false;
            if (InsideLink(src, index + 1, 4))
                return false;

            LayerParam layer;
            layer.type() = LayerTypePooling;
            layer.name() = src[index + 4].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.pooling().method() = src[index + 4].pooling().method();
            layer.pooling().kernel() = Shape({ src[index + 1].pooling().kernel()[0], src[index + 4].pooling().kernel()[0] });
            layer.pooling().pad() = src[index + 4].pooling().pad();
            layer.pooling().stride() = src[index + 4].pooling().stride();
            layer.pooling().excludePad() = src[index + 4].pooling().excludePad();
            dst.push_back(layer);
            index += 4;
            return true;
        }

        bool MergeSpaceToDepth(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 9)
                return false;
            if (src[index + 0].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 1].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 2].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 3].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 4].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 5].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 6].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 7].type() != LayerTypeStridedSlice)
                return false;
            if (src[index + 8].type() != LayerTypeConcat)
                return false;
            if (InsideLink(src, index + 1, 8))
                return false;

            LayerParam layer;
            layer.type() = LayerTypeSpaceToDepth;
            layer.name() = src[index + 8].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            dst.push_back(layer);
            index += 8;
            return true;
        }

        bool MergeNormalize(const LayerParams & src, size_t & index, LayerParams & dst, Changes & changes)
        {
            if (src.size() < index + 4)
                return false;
            if (src[index + 0].type() != LayerTypeReduction || src[index + 0].reduction().type() != ReductionTypeL2)
                return false;
            if (src[index + 1].type() != LayerTypeRestrictRange || src[index + 1].restrictRange().lower() >= 0.0000001f)
                return false;
            if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeShape)
                return false;
            if (src[index + 3].type() != LayerTypeTile)
                return false;
            if (src[index + 4].type() != LayerTypeBinaryOperation || src[index + 4].binaryOperation().type() != BinaryOperationTypeDiv)
                return false;

            LayerParam layer;
            layer.type() = LayerTypeNormalize;
            layer.name() = src[index + 4].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.normalize().acrossSpatial() = true;
            layer.normalize().channelShared() = true;
            layer.normalize().eps() = 0;
            dst.push_back(layer);
            index += 4;
            return true;
        }

        bool MergeNormalizeV2(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 8)
                return false;
            if (src[index + 0].type() != LayerTypeReduction || src[index + 0].reduction().type() != ReductionTypeMean)
                return false;
            if (src[index + 1].type() != LayerTypeBinaryOperation || src[index + 1].binaryOperation().type() != BinaryOperationTypeSub)
                return false;
            if (src[index + 2].type() != LayerTypePower || src[index + 2].power().power() != 2.0f)
                return false;
            if (src[index + 3].type() != LayerTypeReduction || src[index + 3].reduction().type() != ReductionTypeMean)
                return false;
            if (src[index + 4].type() != LayerTypePower || src[index + 4].power().power() != 1.0f || src[index + 4].power().scale() != 1.0f)
                return false;
            if (src[index + 5].type() != LayerTypeUnaryOperation || src[index + 5].unaryOperation().type() != UnaryOperationTypeSqrt)
                return false;
            if (src[index + 6].type() != LayerTypeBinaryOperation || src[index + 6].binaryOperation().type() != BinaryOperationTypeDiv)
                return false;
            const WeightParam* scale = GetEltwiseWeight(index + 7, src);
            if (scale == NULL || src[index + 7].eltwise().operation() != EltwiseOperationTypeProduct)
                return false;
            const WeightParam* shift = GetEltwiseWeight(index + 8, src);
            if (shift == NULL || !IsAdd(src[index + 8]))
                return false;
            if (InsideLink(src, index + 1, 7))
                return false;

            LayerParam layer;
            layer.type() = LayerTypeNormalize;
            layer.name() = src[index + 8].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.normalize().eps() = src[index + 4].power().shift();
            layer.normalize().version() = 2;
            layer.weight().push_back(*scale);
            layer.weight().push_back(*shift);
            if (isNhwc && !PermutedToNchw(src, layer.src(), false, false, false))
                layer.normalize().axis() = 1;
            else
                layer.normalize().axis() = -1;
            dst.push_back(layer);
            index += 8;
            return true;
        }

        bool MergeNormalizeV4(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 9)
                return false;
            if (src[index + 0].type() != LayerTypeUnaryOperation || src[index + 0].unaryOperation().type() != UnaryOperationTypeAbs)
                return false;
            if (src[index + 1].type() != LayerTypePower || src[index + 1].power().power() != 2.0f)
                return false;
            if (src[index + 2].type() != LayerTypeReduction || src[index + 2].reduction().type() != ReductionTypeSum)
                return false;
            if (src[index + 3].type() != LayerTypePower || src[index + 3].power().power() != 0.5f)
                return false;
            if (src[index + 4].type() != LayerTypeReduction || src[index + 4].reduction().type() != ReductionTypeMean)
                return false;
            if (src[index + 5].type() != LayerTypePower || src[index + 5].power().power() != 1.0f || src[index + 5].power().scale() != 1.0f)
                return false;
            if (src[index + 6].type() != LayerTypeBinaryOperation || src[index + 6].binaryOperation().type() != BinaryOperationTypeDiv)
                return false;
            if (src[index + 7].type() != LayerTypeEltwise || src[index + 7].eltwise().operation() != EltwiseOperationTypeProduct)
                return false;
            if (src[index + 8].type() != LayerTypeScale || !src[index + 8].scale().biasTerm())
                return false;
            if (!IsAdd(src[index + 9]))
                return false;
            if (InsideLink(src, index + 1, 8))
                return false;
            LayerParam layer;
            layer.type() = LayerTypeNormalize;
            layer.name() = src[index + 9].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.normalize().eps() = src[index + 5].power().shift();
            layer.normalize().version() = 4;
            layer.weight() = src[index + 8].weight();
            if (isNhwc && !PermutedToNchw(src, layer.src(), false, false, false))
                layer.normalize().axis() = -1;
            else
                layer.normalize().axis() = 1;
            dst.push_back(layer);
            index += 9;
            return true;
        }

        bool MergeNormalizeV5(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 4)
                return false;
            if (src[index + 0].type() != LayerTypeReshape || src[index + 0].reshape().shape().size() != 3)
                return false;
            if (src[index + 1].type() != LayerTypeNormalize || src[index + 1].normalize().version() != 3 || src[index + 1].src()[0] != src[index + 0].dst()[0])
                return false;
            if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeShape || src[index + 2].src()[0] != src[index + 0].src()[0])
                return false;
            if (src[index + 3].type() != LayerTypeReshape || src[index + 3].src()[0] != src[index + 1].dst()[0] || src[index + 3].src()[1] != src[index + 2].dst()[0])
                return false;

            LayerParam layer;
            layer.type() = LayerTypeNormalize;
            layer.name() = src[index + 3].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.normalize() = src[index + 1].normalize();
            layer.normalize().version() = 5;
            layer.weight() = src[index + 1].weight();
            dst.push_back(layer);
            index += 3;
            return true;
        }

        bool MergeScale(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 2)
                return false;
            const WeightParam* scale = GetEltwiseWeight(index + 0, src);
            if (scale == NULL || src[index + 0].eltwise().operation() != EltwiseOperationTypeProduct)
                return false;
            const WeightParam* shift = GetEltwiseWeight(index + 1, src);
            if (shift == NULL || !IsAdd(src[index + 1]))
                return false;
            if(src[index + 1].src()[0] != src[index + 0].dst()[0])
                return false;
            if (scale->dim() != shift->dim())
                return false;

            LayerParam layer;
            layer.type() = LayerTypeScale;
            layer.name() = src[index + 1].name();
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            layer.scale().biasTerm() = true;
            layer.weight().push_back(*scale);
            layer.weight().push_back(*shift);
            dst.push_back(layer);
            index += 1;
            return true;
        }

        bool MergeTiledScale2D(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 4)
                return false;
            if (src[index + 0].type() != LayerTypeTile || src[index + 0].src().size() != 2)
                return false;
            if (src[index + 1].type() != LayerTypeTile || src[index + 1].src().size() != 2)
                return false;
            if (!IsMul(src[index + 2]) || src[index + 2].src().size() != 2)
                return false;
            if (!IsMul(src[index + 3]) || src[index + 3].src().size() != 2)
                return false;
            if (src[index + 0].src()[1] != src[index + 1].src()[1] || 
                src[index + 2].src()[1] != src[index + 1].dst()[0] ||
                src[index + 3].src()[1] != src[index + 0].dst()[0])
                return false;

            LayerParam layer;
            layer.type() = LayerTypeTiledScale2D;
            layer.name() = src[index + 3].name();
            layer.src().push_back(src[index + 2].src()[0]);
            layer.src().push_back(src[index + 1].src()[0]);
            layer.src().push_back(src[index + 0].src()[0]);
            layer.dst().push_back(layer.name());
            dst.push_back(layer);
            index += 3;
            return true;
        }

        bool MergeUnpack4(const LayerParams& src, size_t& index, bool isNhwc, LayerParams& dst, Changes& changes)
        {
            if (src.size() < index + 19)
                return false;
            if (src[index + 0].type() != LayerTypeMeta || src[index + 0].meta().type() != MetaTypeShape)
                return false;
            if (!IsMetaConst64i(src[index + 1]))
                return false;
            if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeGather)
                return false;
            if (!IsMetaConst64i(src[index + 3]))
                return false;
            if (!IsMetaConst64i(src[index + 4]))
                return false;
            if (src[index + 5].type() != LayerTypeMeta || src[index + 5].meta().type() != MetaTypeAdd)
                return false;
            if (!IsMetaConst64i(src[index + 6], Lng(4)))
                return false;
            if (src[index + 7].type() != LayerTypeMeta || src[index + 7].meta().type() != MetaTypeDiv)
                return false;
            if (!IsMetaConst64i(src[index + 8], Lng(1)))
                return false;
            if (src[index + 9].type() != LayerTypeMeta || src[index + 9].meta().type() != MetaTypeMul)
                return false;
            if (src[index + 10].type() != LayerTypeStridedSlice || src[index + 10].src()[0] != src[index + 0].src()[0])
                return false;
            if (!IsMetaConst64i(src[index + 11], Lng(2)))
                return false;
            if (src[index + 12].type() != LayerTypeMeta || src[index + 12].meta().type() != MetaTypeMul)
                return false;
            if (src[index + 13].type() != LayerTypeStridedSlice || src[index + 13].src()[0] != src[index + 0].src()[0])
                return false;
            if (!IsMetaConst64i(src[index + 14], Lng(3)))
                return false;
            if (src[index + 15].type() != LayerTypeMeta || src[index + 15].meta().type() != MetaTypeMul)
                return false;
            if (src[index + 16].type() != LayerTypeStridedSlice || src[index + 16].src()[0] != src[index + 0].src()[0])
                return false;
            if (!IsMetaConst64i(src[index + 17], Lng(4)))
                return false;
            if (src[index + 18].type() != LayerTypeMeta || src[index + 18].meta().type() != MetaTypeMul)
                return false;
            if (src[index + 19].type() != LayerTypeStridedSlice || src[index + 19].src()[0] != src[index + 0].src()[0])
                return false;

            LayerParam layer;
            layer.type() = LayerTypeUnpack;
            layer.name() = src[index + 0].name();
            layer.src() = src[index + 0].src();
            layer.unpack().axis() = isNhwc ? 3 : 1;
            layer.dst().push_back(src[index + 10].dst()[0]);
            layer.dst().push_back(src[index + 13].dst()[0]);
            layer.dst().push_back(src[index + 16].dst()[0]);
            layer.dst().push_back(src[index + 19].dst()[0]);
            dst.push_back(layer);
            index += 19;
            return true;
        }

        bool MergeRnnGruBd(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)       
        {
            const size_t RNN_GRU_BD_SIZE = 19;
            if (index == 0 || index + RNN_GRU_BD_SIZE >= src.size())
                return false;
            const LayerParam& parent = src[index - 1];
            if (parent.type() != LayerTypeTensorIterator || parent.src().size() != 2 || 
                parent.dst().size() != 1 || parent.tensorIterator().back().size() != 1)
                return false;
            for (size_t i = 0; i < RNN_GRU_BD_SIZE; ++i)
            {
                if (src[index + i].parent() != parent.name())
                    return false;
            }
            if (src[index + 0].type() != LayerTypeInput || src[index + 1].type() != LayerTypeMeta)
                return false;
            if (src[index + 2].type() != LayerTypeSqueeze || src[index + 3].type() != LayerTypeInput)
                return false;
            if (src[index + 4].type() != LayerTypeConcat || src[index + 5].type() != LayerTypeInnerProduct || src[index + 5].weight().size() != 2)
                return false;
            if (src[index + 6].type() != LayerTypeSigmoid || src[index + 7].type() != LayerTypeUnpack)
                return false;
            if (src[index + 8].type() != LayerTypeEltwise || src[index + 9].type() != LayerTypePower)
                return false;
            if (src[index + 10].type() != LayerTypeEltwise || src[index + 11].type() != LayerTypeConcat)
                return false;
            if (src[index + 12].type() != LayerTypeInnerProduct || src[index + 12].weight().size() != 2 || src[index + 13].type() != LayerTypeUnaryOperation)
                return false;
            if (src[index + 14].type() != LayerTypeEltwise || src[index + 15].type() != LayerTypeEltwise)
                return false;
            if (src[index + 16].type() != LayerTypeStub || src[index + 17].type() != LayerTypeExpandDims || src[index + 18].type() != LayerTypeStub)
                return false;
            if (!src[index + RNN_GRU_BD_SIZE].parent().empty())
                return false;

            dst.push_back(src[index + 0]);
            dst.push_back(src[index + 3]);

            LayerParam layer;
            layer.type() = LayerTypeRnnGruBd;
            layer.parent() = parent.name();
            layer.name() = parent.name() + "_RnnGruBd";
            layer.src().push_back(src[index + 0].dst()[0]);
            layer.src().push_back(src[index + 3].dst()[0]);
            layer.dst().push_back(src[index + 18].dst()[0]);
            layer.dst().push_back(src[index + 16].dst()[0]);
            layer.weight().push_back(src[index + 5].weight()[0]);
            layer.weight().push_back(src[index + 5].weight()[1]);
            layer.weight().push_back(src[index + 12].weight()[0]);
            layer.weight().push_back(src[index + 12].weight()[1]);
            dst.push_back(layer);

            index += RNN_GRU_BD_SIZE - 1;
            return true;
        }

        bool MergeYoloV7(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (index == 0 || index + 4 >= src.size())
                return false;

            const LayerParam &c0 = src[index - 1];
            if (c0.type() != LayerTypeConcat || c0.src().size() != 3)
                return false;

            const LayerParam &ss0 = src[index + 0];
            if (ss0.type() != LayerTypeStridedSlice || ss0.src().size() != 1 || ss0.src()[0] != c0.dst()[0] ||
                ss0.stridedSlice().beginDims() != Lng(0) || ss0.stridedSlice().endDims() != Lng(4) ||
                ss0.stridedSlice().strideDims() != Lng(1) || (ss0.stridedSlice().axes() != Shp(2) && ss0.stridedSlice().axes() != Shp(1)))
                return false;

            const LayerParam &ss1 = src[index + 1];
            if (ss1.type() != LayerTypeStridedSlice || ss1.src().size() != 1 || ss1.src()[0] != c0.dst()[0] ||
                ss1.stridedSlice().beginDims() != Lng(4) || ss1.stridedSlice().endDims() != Lng(5) ||
                ss1.stridedSlice().strideDims() != Lng(1) || (ss1.stridedSlice().axes() != Shp(2) && ss1.stridedSlice().axes() != Shp(1)))
                return false;

            size_t start = index + 2;
            const LayerParam & ss2 = src[index + 2];
            if (ss2.type() == LayerTypeStridedSlice)
            {
                if (ss2.src().size() != 1 || ss2.src()[0] != c0.dst()[0] ||
                    ss2.stridedSlice().beginDims() != Lng(5) || ss1.stridedSlice().strideDims() != Lng(1) ||
                    (ss2.stridedSlice().axes() != Shp(2) && ss2.stridedSlice().axes() != Shp(1)))
                    return false;

                const LayerParam& e0 = src[index + 3];
                if (e0.type() != LayerTypeEltwise || e0.eltwise().operation() != EltwiseOperationTypeProduct || e0.src().size() != 2 || 
                    (e0.src()[0] != ss1.dst()[0] && e0.src()[0] != ss2.dst()[0]) || (e0.src()[1] != ss1.dst()[0] && e0.src()[1] != ss2.dst()[0]))
                    return false;

                start = index + 4;
            }

            const LayerParam& ip0 = src[start + 0];
            if (ip0.type() != LayerTypeInnerProduct || ip0.src().size() != 1 || ip0.src()[0] != ss0.dst()[0] ||
                ip0.innerProduct().outputNum() != 4 || ip0.innerProduct().biasTerm() != false ||
                ip0.weight()[0].dim() != Shp(4, 4))
                return false;

            const LayerParam& r0 = src[start + 1];
            if (r0.type() != LayerTypeReduction || r0.reduction().axis() != Ints({ 2 }) || r0.reduction().type() != ReductionTypeMax ||
                r0.src().size() != 1 || r0.src()[0] != src[start - 1].dst()[0])
                return false;

            const LayerParam& am0 = src[start + 2];
            if (am0.type() != LayerTypeArgMax || am0.argMax().axis() != 2 || 
                am0.src().size() != 1 || am0.src()[0] != src[start - 1].dst()[0])
                return false;

            const LayerParam& c1 = src[start + 3];
            if (c1.type() != LayerTypeCast || c1.cast().type() != TensorType32f || 
                c1.src().size() != 1 || c1.src()[0] != am0.dst()[0])
                return false;

            const LayerParam& p0 = src[start + 4];
            if (p0.type() != LayerTypePower || p0.power().power() != 1.0f || p0.power().shift() != 0.0f ||
                p0.src().size() != 1 || p0.src()[0] != c1.dst()[0])
                return false;

            const LayerParam& e1 = src[start + 5];
            if (!IsAdd(e1) || (e1.src()[0] != ip0.dst()[0] && e1.src()[0] != p0.dst()[0]) || (e1.src()[1] != ip0.dst()[0] && e1.src()[1] != p0.dst()[0]))
                return false;

            const LayerParam& p1 = src[start + 6];
            if (p1.type() != LayerTypePermute || p1.permute().order() != Shp(0, 2, 1) ||
                p1.src().size() != 1 || p1.src()[0] != r0.dst()[0])
                return false;

            const LayerParam& nms0 = src[start + 7];
            if (nms0.type() != LayerTypeNonMaxSuppression || nms0.src().size() != 2 ||
                (nms0.src()[0] != e1.dst()[0] && nms0.src()[0] != p1.dst()[0]) || (nms0.src()[1] != e1.dst()[0] && nms0.src()[1] != p1.dst()[0]))
                return false;

            LayerParam yoloV7;
            yoloV7.type() = LayerTypeYoloV7;
            yoloV7.name() = src.back().dst()[0];
            yoloV7.src().push_back(c0.dst()[0]);
            yoloV7.dst().push_back(src.back().dst()[0]);
            yoloV7.yoloV7().maxOutputBoxesPerClass() = nms0.nonMaxSuppression().maxOutputBoxesPerClass();
            yoloV7.yoloV7().iouThreshold() = nms0.nonMaxSuppression().iouThreshold();
            yoloV7.yoloV7().scoreThreshold() = nms0.nonMaxSuppression().scoreThreshold();
            yoloV7.yoloV7().oneClass() = (start == index + 2);
            index += src.size() - 1 - index;
            dst.push_back(yoloV7);

            return true;
        }

        bool TransposeConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
        {
            size_t end = index;
            if (!PermutedToNchw(src, src[index].src(), true, false, false))
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
            if (!(count >= _param.convToNhwc() || (count == 1 && src[index].convolution().group() != 1)))
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

        bool SimplifyInterp(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes)
        {
            if (index + 7 >= src.size())
                return false;
            if (src[index + 0].type() != LayerTypeMeta || src[index + 0].meta().type() != MetaTypeShape)
                return false;
            if (src[index + 1].type() != LayerTypeMeta || src[index + 1].meta().type() != MetaTypeConst)
                return false;
            if (src[index + 2].type() != LayerTypeMeta || src[index + 2].meta().type() != MetaTypeConst)
                return false;
            if (src[index + 3].type() != LayerTypeMeta || src[index + 3].meta().type() != MetaTypeConst)
                return false;
            if (src[index + 4].type() != LayerTypeMeta || src[index + 4].meta().type() != MetaTypeSlice)
                return false;
            if (src[index + 5].type() != LayerTypeMeta || src[index + 5].meta().type() != MetaTypeConst || src[index + 5].meta().alpha().shape() != Shp(2))
                return false;
            if (src[index + 6].type() != LayerTypeMeta || src[index + 6].meta().type() != MetaTypePack)
                return false;
            if (src[index + 7].type() != LayerTypeInterp || src[index + 7].src().size() != 2)
                return false;

            LayerParam layer = src[index + 7];
            layer.src().resize(1);
            layer.interp().height() = (int)src[index + 5].meta().alpha().i64()[0];
            layer.interp().width() = (int)src[index + 5].meta().alpha().i64()[1];
            dst.push_back(layer);

            index += 7;
            return true;
        }

        //-------------------------------------------------------------------------------------------------

        const WeightParam* GetEltwiseWeight(size_t index, const LayerParams& layers) const
        {
            if (index < layers.size() && (layers[index].type() == LayerTypeEltwise && layers[index].src().size() == 2) || layers[index].type() == LayerTypeAdd)
            {
                const LayerParam* src0 = GetLayerByName(layers, layers[index].src()[0]);
                if (src0 && src0->type() == LayerTypeConst)
                    return src0->weight().data() + 0;
                const LayerParam* src1 = GetLayerByName(layers, layers[index].src()[1]);
                if (src1 && src1->type() == LayerTypeConst)
                    return src1->weight().data() + 0;
            }
            return NULL;
        }

        bool Rename(const Change & change, LayerParams & layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                for (size_t j = 0; j < layers[i].src().size(); ++j)
                {
                    if (layers[i].src()[j] == change.first)
                    {
                        if (layers[i].src()[0] == layers[i].dst()[0] && layers[i].src().size() == 1)
                            layers[i].dst()[0] = change.second;
                        layers[i].src()[j] = change.second;
                    }
                }
            }
            return true;
        }

        bool Rename(const Changes & changes, LayerParams & layers)
        {
            for (size_t k = 0; k < changes.size(); ++k)
            {
                if (!Rename(changes[k], layers))
                    return false;
            }
            return true;
        }

        size_t Users(const String& name, const LayerParams& layers, size_t start, const String & parent) const
        {
            size_t users = 0;
            for (size_t i = start; i < layers.size(); ++i)
            {
                if (layers[i].parent() != parent)
                    continue;
                for (size_t j = 0; j < layers[i].src().size(); ++j)
                {
                    if (layers[i].src()[j] == name)
                        users++;
                }
            }
            return users;
        }

        bool CanReuse(const LayerParam & layer)
        {
            if (layer.type() == LayerTypeSigmoid)
                return true;
            if (layer.type() == LayerTypeSwish)
                return true;
            //if (layer.type() == LayerTypeScale)
            //    return true;
            //if (layer.type() == LayerTypePower)
            //    return true;
            if (_param.reuseEltwise() && layer.type() == LayerTypeEltwise)
                return true;
            if (layer.type() == LayerTypeRelu)
                return true;
            if (layer.type() == LayerTypeGelu)
                return true;
            if (layer.type() == LayerTypeSqueezeExcitation)
                return true;
            if (layer.type() == LayerTypeSoftmax && layer.softmax().log() == 0)
                return true;
            if (layer.type() == LayerTypePooling && layer.pooling().method() == PoolingMethodTypeMax && 
                layer.pooling().kernel() == Shp(1, 1) && layer.pooling().stride() == Shp(1, 1))
                return true;
            if (layer.type() == LayerTypeTiledScale2D)
                return true;
            return false;
        }

        bool HasOutput(const Synet::NetworkParam& network, const LayerParam & layer)
        {
            for (size_t l = 0; l < layer.dst().size(); ++l)
                for (size_t d = 0; d < network.dst().size(); ++d)
                    if (layer.dst()[l] == network.dst()[d])
                        return true;
            return false;
        }

        bool ReuseLayers(Synet::NetworkParam& network)
        {
            if (network.quantization().method() != QuantizationMethodUnknown)
                return true;
            LayerParams & layers = network.layers();
            for (size_t i = 0; i < layers.size(); ++i)
            {
                LayerParam & layer = layers[i];
                if (layer.src().empty())
                    continue;
                if (Users(layer.src()[0], layers, i, "") > 1)
                    continue;
                if (i && layer.src()[0] == layers[i - 1].name() && layers[i - 1].type() == LayerTypeConst)
                    continue;
                if (Users(layer.dst()[0], layers, i + 1, "") == 0)
                    continue;
                if (HasOutput(network, layer))
                    continue;
                size_t srcIndex = GetLayerIndex(layers, layer.src()[0]);
                if (layers[srcIndex].type() == LayerTypeReshape)
                {
                    if (Users(layers[srcIndex].src()[0], layers, srcIndex, "") > 1)
                        continue;
                }
                if (!CanReuse(layer))
                    continue;
                if (!Rename(Change(layer.dst()[0], layer.src()[0]), layers))
                    return false;
                layer.dst()[0] = layer.src()[0];
            }
            return true;
        }

        bool IsStub(const LayerParam& layer, const Synet::NetworkParam& network)
        {
            if (layer.type() == LayerTypeStub)
            {
                if (Users(layer.dst()[0], network.layers(), 0, layer.parent()) > 0)// && !HasOutput(network, layer))
                    return true;
                const LayerParam* prev = GetLayerByName(network.layers(), layer.src()[0]);
                if (prev && prev->type() == LayerTypeDetectionOutput)
                    return true;
            }
            if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeStub)
                return true;
            if (layer.type() == LayerTypePooling && layer.pooling().method() == PoolingMethodTypeMax &&
                layer.pooling().kernel() == Shp(1, 1) && layer.pooling().stride() == Shp(1, 1))
                return true;
            return false;
        }

        bool RemoveStub(Synet::NetworkParam& network)
        {
            LayerParams& layers = network.layers();
            for (size_t i = 1; i < layers.size(); ++i)
            {
                LayerParam & layer = layers[i];
                if (!IsStub(layer, network))
                    continue;
                if (layer.src().size() != 1 || layer.dst().size() != 1)
                    continue;
                if (!Rename(Change(layer.dst()[0], layer.src()[0]), layers))
                    return false;
                layers.erase(layers.begin() + i);
                if (i)
                    i--;
            }
            return true;
        }

        bool IsNnwc(const NetworkParam& network)
        {
            for (size_t i = 0; i < network.layers().size(); ++i)
            {
                if (network.layers()[i].weight().size() && network.layers()[i].weight()[0].format() == TensorFormatNhwc)
                    return true;
            }
            return false;
        }
    };

    //--------------------------------------------------------------------------------------------------

    bool OptimizeSynetModel(const String& srcXml, const String& srcBin, const String& dstXml, const String& dstBin, const OptimizerParam& param = OptimizerParam());
}