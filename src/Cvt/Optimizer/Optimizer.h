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