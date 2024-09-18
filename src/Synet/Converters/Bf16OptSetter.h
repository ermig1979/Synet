/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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
#include "Synet/Converters/SynetUtils.h"

namespace Synet
{
    struct Bf16OptParam
    {
        CPL_PARAM_VALUE(bool, enable, false);
        CPL_PARAM_VALUE(uint32_t, minSrcC, 32);
        CPL_PARAM_VALUE(uint32_t, minDstC, 16);
        CPL_PARAM_VALUE(LowPrecisionType, addType, LowPrecisionTypeActive);
        CPL_PARAM_VALUE(LowPrecisionType, reluType, LowPrecisionTypePassive);
        CPL_PARAM_VALUE(Strings, exclude, Strings());
        CPL_PARAM_VALUE(Strings, manualActive, Strings());
    };

    class Bf16OptSetter : public SynetUtils
    {
    public:
        Bf16OptSetter(const Bf16OptParam& param)
            : _param(param)
        {
        }

        bool Run(Synet::NetworkParam & network, Floats & bin)
        {
            if (!_param.enable())
                return true;            
            if (!SetSimpleCase(network.layers()))
                return false;
            if (!SetManualActive(network.layers()))
                return false;
            //if (!SetConvAddReluCase(network.layers()))
            //    return false;
            return true;
        }

    private:
        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef std::pair<String, String> Change;
        typedef std::vector<Change> Changes;
        typedef std::vector<LayerType> LayerTypes;
        typedef std::set<String> StringSet;

        const Bf16OptParam& _param;

        //-------------------------------------------------------------------------------------------------

        bool SetSimpleCase(LayerParams& layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                LayerParam &layer = layers[i];
                if (IsExclude(layers, layer.name()))
                    continue;
                if (layer.type() == LayerTypeConvolution && (layer.weight()[0].format() == TensorFormatNhwc || (layer.convolution().kernel()[0] == 1 && layer.convolution().kernel()[1] == 1)))
                {
                    if(layer.convolution().group() == 1 && EffectiveSrcC(layer) >= _param.minSrcC() && layer.convolution().outputNum() >= _param.minDstC())
                        layer.lowPrecision().bf16Type() = LowPrecisionTypeActive;
                }
                else if (layer.type() == LayerTypeInnerProduct)
                {
                    if ((EffectiveSrcC(layer) >= _param.minSrcC() && layer.innerProduct().outputNum() >= _param.minDstC()) || layer.src().size() > 1)
                        layer.lowPrecision().bf16Type() = LowPrecisionTypeActive;
                }
                else if (layer.type() == Synet::LayerTypeEltwise && 
                    layer.eltwise().operation() == Synet::EltwiseOperationTypeSum && layer.src().size() <= 2)
                {
                    layer.type() = LayerTypeAdd;
                    layer.eltwise() = EltwiseParam();
                    layer.lowPrecision().bf16Type() = _param.addType();
                }
                else if (layer.type() == Synet::LayerTypeRelu)
                {
                    layer.lowPrecision().bf16Type() = _param.reluType();
                }
                else if (layer.type() == LayerTypeDeconvolution && layer.weight()[0].format() == TensorFormatNhwc)
                {
                    if (layer.convolution().group() == 1 && EffectiveSrcC(layer) >= _param.minSrcC() && layer.convolution().outputNum() >= _param.minDstC())
                        layer.lowPrecision().bf16Type() = LowPrecisionTypeActive;
                }
            }
            return true;
        }

        bool SetManualActive(LayerParams& layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const LayerParam & layer = layers[i];
                bool set = false;
                for (size_t m = 0; m < _param.manualActive().size() && !set; ++m)
                    if (layer.name() == _param.manualActive()[m])
                        set = true;
                if(set)
                    ((LayerParam&)layer).lowPrecision().bf16Type() = LowPrecisionTypeActive;
            }
            return true;
        }

        bool SetConvAddReluCase(LayerParams& layers)
        {
            if (_param.addType() != LowPrecisionTypePassive || _param.reluType() != LowPrecisionTypePassive)
                return true;
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const LayerParam * conv = &layers[i];
                if (conv->type() != Synet::LayerTypeConvolution || conv->lowPrecision().bf16Type() != LowPrecisionTypeActive)
                    continue;
                const LayerParam* relu = GetLayerByName(layers, conv->src()[0]);
                if (relu == NULL || relu->type() != Synet::LayerTypeRelu || relu->lowPrecision().bf16Type() != LowPrecisionTypePassive)
                    continue;
                const LayerParam* add = GetLayerByName(layers, relu->src()[0]);
                if (add == NULL || add->type() != Synet::LayerTypeAdd || add->lowPrecision().bf16Type() != LowPrecisionTypePassive)
                    continue;
                if (IsExclude(layers, add->name()))
                    continue;
                ((LayerParam*)add)->lowPrecision().bf16Type() = LowPrecisionTypeActive;
            }
            return true;
        }

        bool IsExclude(const LayerParams& layers, const String& name) const
        {
            bool exclude = false;
            for (size_t e = 0; e < _param.exclude().size() && !exclude; ++e)
                if (name == _param.exclude()[e])
                    exclude = true;
            return exclude;
        }

        size_t EffectiveSrcC(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeConvolution)
            {
                const WeightParam& weight = layer.weight()[0];
                if (weight.format() == TensorFormatNhwc)
                    return weight.dim()[2] * layer.convolution().kernel()[0] * layer.convolution().kernel()[1];
                else
                    return weight.dim()[1];
            }
            if (layer.type() == LayerTypeInnerProduct && layer.src().size() == 1)
            {
                const WeightParam& weight = layer.weight()[0];
                if (layer.innerProduct().transposeB())
                    return weight.dim()[0];
                else
                    return weight.dim()[1];
            }
            if (layer.type() == LayerTypeDeconvolution)
            {
                const WeightParam& weight = layer.weight()[0];
                if (weight.format() == TensorFormatNhwc)
                    return weight.dim()[0];
                else
                    return weight.dim()[0];
            }
            return 0;
        }

        size_t EffectiveDstC(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeDeconvolution)
            {
                const WeightParam& weight = layer.weight()[0];
                if (weight.format() == TensorFormatNhwc)
                    return weight.dim()[3] * layer.convolution().kernel()[0] * layer.convolution().kernel()[1];
                else
                    return weight.dim()[1] * layer.convolution().kernel()[0] * layer.convolution().kernel()[1];
            }
            return 0;
        }
    };
}