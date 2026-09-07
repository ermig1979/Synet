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
                const LayerParam* prev = GetLayer(network.layers(), layer.src()[0]);
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