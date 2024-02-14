/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
#include "Synet/Tensor.h"

namespace Synet
{
    struct OnnxParam
    {
        CPL_PARAM_VALUE(Strings, toNchwHints, Strings());
        CPL_PARAM_VALUE(Strings, toNhwcHints, Strings());
        CPL_PARAM_VALUE(bool, transpose0312PermuteToNhwc, false);
        CPL_PARAM_VALUE(bool, globalPoolingPermuteToNchw, true);
    };

    class SynetUtils
    {
    protected:
        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<float> Vector;

        struct Pin
        {
            String name;
            int index;
            Pin(const String& n = String(), int i = 0) : name(n), index(i) {}
        };

        static bool CheckDims(const Shape& shape, size_t dims, const String& desc)
        {
            if (shape.size() != dims)
                SYNET_ERROR("Wrong " << desc << " shape " << ToStr(shape) << " !");
            return true;
        }

        static bool CheckSourceNumber(const LayerParam& layer, size_t size)
        {
            if (layer.src().size() != size)
                SYNET_ERROR("Wrong number of sources (" << layer.src().size() << " instead of " << size << ") !");
            return true;
        }

        static bool CheckSourceNumber(const LayerParam& layer, size_t min, size_t max)
        {
            if (layer.src().size() < min || layer.src().size() > max)
                SYNET_ERROR("Wrong number of sources (" << layer.src().size() << ". It must be in range [" << min << ", " << max << "] !");
            return true;
        }

        static bool CheckDestinationNumber(const LayerParam& layer, size_t size)
        {
            if (layer.dst().size() != size)
                SYNET_ERROR("Wrong number of destinations (" << layer.dst().size() << " instead of " << size << ") !");
            return true;
        }

        static bool CompactShape(Shape& shape)
        {
            size_t count = 0, value = 1;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] != 1)
                {
                    value = shape[i];
                    count++;
                }
            }
            if (count > 1)
                SYNET_ERROR("Can't compact shape " << ToStr(shape) << " !");
            shape = Shp(value);
            return true;
        }

        static size_t GetLayerIndex(const LayerParams& layers, const String& name)
        {
            Pin pin = ParsePin(name);
            for (size_t i = 0; i < layers.size(); ++i)
            {
                if (pin.name == layers[i].name())
                    return i;
                for (size_t d = 0; d < layers[i].dst().size(); ++d)
                    if (pin.name == layers[i].dst()[d])
                        return i;
            }
            return layers.size();
        }

        static const LayerParam* GetLayer(const LayerParams& layers, const String& name)
        {
            Pin pin = ParsePin(name);
            for (size_t i = 0; i < layers.size(); ++i)
            {
                if (pin.name == layers[i].name())
                    return &layers[i];
                for(size_t d = 0; d < layers[i].dst().size(); ++d)
                    if (pin.name == layers[i].dst()[d])
                        return &layers[i];
            }
            SYNET_ERROR("Can't found layer " << pin.name << " !");
        }

        template<class T> static T* GetWeight(Vector& bin, size_t offset)
        {
            if (offset >= bin.size() * sizeof(T))
                SYNET_ERROR("Vector access overflow: " << offset << " >= " << bin.size() * sizeof(T) << " !");
            return (T*)((uint8_t*)bin.data() + offset);
        }

        template<class T> static const T* GetWeight(const Vector& bin, size_t offset)
        {
            if (offset >= bin.size() * sizeof(T))
                SYNET_ERROR("Vector access overflow: " << offset << " >= " << bin.size() * sizeof(T) << " !");
            return (const T*)((const uint8_t*)bin.data() + offset);
        }

        template<class T> static T* GetWeight(Vector& bin, const WeightParam& param)
        {
            return GetWeight<T>(bin, param.offset());
        }

        template<class T> static const T* GetWeight(const Vector& bin, const WeightParam& param)
        {
            return GetWeight<T>(bin, param.offset());
        }

        static String NotImplementedMarker()
        {
            return "~~~NOT_IMPLEMENTED~~~";
        }

        static Pin ParsePin(const String& name)
        {
            Pin pin(name);
            size_t delimiter = name.find_first_of(":");
            if (delimiter != std::string::npos)
            {
                pin.name = name.substr(0, delimiter);
                std::istringstream(name.substr(delimiter + 1)) >> pin.index;
            }
            return pin;
        }

        static bool PermutedToNchw(const LayerParams& layers, size_t current, bool checkInnerProduct, bool checkPriorBox, bool globalPooling)
        {
            const LayerParam& layer = layers[current];
            if (layer.type() == LayerTypeConvolution && layer.weight()[0].format() == TensorFormatNhwc)
                return false;
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNhwc)
                return false;
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNchw)
                return true;
            if (checkInnerProduct && layer.type() == LayerTypeInnerProduct)
                return true;
            if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
                return true;
            if (globalPooling && layer.type() == LayerTypePooling && layer.pooling().globalPooling())
                return true;
            if (layer.type() == LayerTypeInput && layer.input().shape().size() > 0 &&
                layer.input().shape()[0].format() == TensorFormatNchw)
                return true;
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                const String & src = layer.src()[s];
                for (size_t l = 0; l < current; ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        if (layers[l].type() == LayerTypeMeta || layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        const String & dst = layers[l].dst()[d];
                        if (src == dst && PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling))
                            return true;
                    }
                }
            }
            return false;
        }

        static bool PermutedToNchw(const LayerParams& layers, bool checkInnerProduct, bool checkPriorBox, bool globalPooling)
        {
            size_t start = layers.size() - 1;
            if (layers[start].type() == LayerTypeConst && start)
                start--;
            return PermutedToNchw(layers, start, checkInnerProduct, checkPriorBox, globalPooling);
        }

        static bool PermutedToNchw(const LayerParams& layers, const Strings & names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling)
        {
            for (size_t s = 0; s < names.size(); ++s)
            {
                for (size_t l = 0; l < layers.size(); ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        const String & dst = layers[l].dst()[d];
                        if (layers[l].type() == LayerTypeMeta || layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        if (names[s] == dst && PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling))
                            return true;
                    }
                }
            }
            return false;
        }

        static int PermutedToNchw(const LayerParams& layers, const Strings& names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, Ints & stat)
        {
            stat.resize(names.size(), 0);
            int count = 0;
            for (size_t s = 0; s < names.size(); ++s)
            {
                for (size_t l = 0; l < layers.size(); ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        if(names[s] == layers[l].dst()[d] && layers[l].type() != LayerTypeMeta && layers[l].type() &&
                            PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling))
                        {
                            stat[s] = 1;
                            count++;
                            break;
                        }
                    }
                }
            }
            return count;
        }

        static TensorFormat CurrentTensorFormat(const LayerParams& layers, size_t current, bool checkInnerProduct, bool checkPriorBox, bool globalPooling)
        {
            const LayerParam& layer = layers[current];
            if (layer.type() == LayerTypeConvolution || layer.type() == LayerTypeDeconvolution)
                return layer.weight()[0].format();
            if (layer.type() == LayerTypePermute && layer.permute().format() != TensorFormatUnknown)
                return layer.permute().format();
            if (layer.type() == LayerTypeInnerProduct)
            {
                if (layer.weight().size())
                    return layer.weight()[0].format();
                if(checkInnerProduct)
                    return TensorFormatNchw;
            }
            if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
                return TensorFormatNchw;
            if (globalPooling && layer.type() == LayerTypePooling && layer.pooling().globalPooling())
                return TensorFormatNchw;
            if (layer.type() == LayerTypeInput && layer.input().shape().size())
                return layer.input().shape()[0].format();
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                const String& src = layer.src()[s];
                for (size_t l = 0; l < current; ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        if (layers[l].type() == LayerTypeMeta || layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        const String& dst = layers[l].dst()[d];
                        if (src == dst)
                        {
                            TensorFormat format = CurrentTensorFormat(layers, l, checkInnerProduct, checkPriorBox, globalPooling);
                            if (format != TensorFormatUnknown)
                                return format;
                        }
                    }
                }
            }
            return TensorFormatUnknown;
        }

        static TensorFormat CurrentTensorFormat(const LayerParams& layers, const Strings& names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling)
        {
            for (size_t s = 0; s < names.size(); ++s)
            {
                for (size_t l = 0; l < layers.size(); ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        const String& dst = layers[l].dst()[d];
                        if (layers[l].type() == LayerTypeMeta || layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        if (names[s] == dst)
                        {
                            TensorFormat format = CurrentTensorFormat(layers, l, checkInnerProduct, checkPriorBox, globalPooling);
                            if (format != TensorFormatUnknown)
                                return format;
                        }
                    }
                }
            }
            return TensorFormatUnknown;
        }

        static size_t UserCount(const LayerParams& layers, size_t index)
        {
            size_t users = 0;
            for (size_t j = index + 1; j < layers.size(); ++j)
                for (size_t k = 0; k < layers[j].src().size(); ++k)
                    if (layers[j].src()[k] == layers[index].name())
                        users++;
            return users;
        }

        static bool RemoveUnusedConst(LayerParams& layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const LayerParam& curr = layers[i];
                if (curr.type() == LayerTypeConst || (curr.type() == LayerTypeMeta && curr.meta().type() == MetaTypeConst))
                {
                    if (UserCount(layers, i) == 0)
                        layers.erase(layers.begin() + i), i -= 1;
                    continue;
                }
                if (i >= 1 && curr.type() == LayerTypeReshape)
                {
                    const LayerParam& prev = layers[i - 1];
                    if (prev.type() == LayerTypeConst && curr.src().size() == 1 && curr.src()[0] == prev.name())
                    {
                        if (UserCount(layers, i) == 0 && UserCount(layers, i - 1) == 1)
                            layers.erase(layers.begin() + i - 1, layers.begin() + i + 1), i -= 2;
                    }
                }
                if (i >= 2 && curr.type() == LayerTypeTile)
                {
                    const LayerParam& prev1 = layers[i - 1];
                    if (prev1.type() == LayerTypeTile && curr.src().size() == 1 && curr.src()[0] == prev1.name())
                    {
                        const LayerParam& prev2 = layers[i - 2];
                        if (prev2.type() == LayerTypeConst && prev1.src().size() == 1 && prev1.src()[0] == prev2.name())
                        {
                            if (UserCount(layers, i) == 0 && UserCount(layers, i - 1) == 1 && UserCount(layers, i - 2) == 1)
                                layers.erase(layers.begin() + i - 2, layers.begin() + i + 1), i -= 3;
                        }
                    }
                }
                if (curr.type() == LayerTypeStub)
                {
                    if (UserCount(layers, i) == 0)
                    {
                        size_t p = GetLayerIndex(layers, curr.src()[0]);
                        if (p < layers.size())
                        {
                            if (layers[p].type() == LayerTypeConst && UserCount(layers, p) == 1)
                            {
                                layers.erase(layers.begin() + i);
                                layers.erase(layers.begin() + p);
                                i -= 2;
                                continue;
                            }
                        }
                    }
                    continue;
                }
            }
            return true;
        }

        static bool ReorderWeight(const Vector& srcBin, const Shape& input, LayerParam& layer, Vector& dstBin)
        {
            if (layer.weight().size() < 1)
            {
                std::cout << "There is no weight to reorder!" << std::endl;
                return false;
            }
            WeightParam& weight = layer.weight()[0];
            const float* pSrc = srcBin.data() + weight.offset() / sizeof(float);
            float* pDst = dstBin.data() + weight.offset() / sizeof(float);
            Shape& shape = weight.dim();
            weight.format() = TensorFormatNhwc;
            switch (layer.type())
            {
            case LayerTypeConvolution:
            {
                shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
                Tensor dst((uint8_t*)pDst, weight.size(), TensorType32f, shape, weight.format());
                for (size_t o = 0; o < shape[3]; ++o)
                    for (size_t i = 0; i < shape[2]; ++i)
                        for (size_t y = 0; y < shape[0]; ++y)
                            for (size_t x = 0; x < shape[1]; ++x)
                                dst.Data<float>(Shape({ y, x, i, o }))[0] = *pSrc++;
                break;
            }
            case LayerTypeDeconvolution:
            {
                shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                Tensor dst((uint8_t*)pDst, weight.size(), TensorType32f, shape, weight.format());
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t c = 0; c < shape[3]; ++c)
                        for (size_t y = 0; y < shape[1]; ++y)
                            for (size_t x = 0; x < shape[2]; ++x)
                                dst.Data<float>(Shape({ i, y, x, c }))[0] = *pSrc++;
                break;
            }
            case LayerTypeInnerProduct:
            {
                if (layer.innerProduct().transposeB())
                {
                    for (size_t c = 0; c < input[1]; c++)
                    {
                        for (size_t y = 0; y < input[2]; y++)
                        {
                            for (size_t x = 0; x < input[3]; x++)
                            {
                                size_t srcOffset = (input[2] * input[3] * c + input[3] * y + x) * shape[1];
                                size_t dstOffset = (input[3] * input[1] * y + input[1] * x + c) * shape[1];
                                for (size_t n = 0; n < shape[1]; n++)
                                    pDst[dstOffset + n] = pSrc[srcOffset + n];
                            }
                        }
                    }
                }
                else
                {
                    for (size_t n = 0; n < shape[0]; n++)
                    {
                        for (size_t c = 0; c < input[1]; c++)
                        {
                            for (size_t y = 0; y < input[2]; y++)
                            {
                                for (size_t x = 0; x < input[3]; x++)
                                {
                                    size_t srcOffset = input[2] * input[3] * c + input[3] * y + x;
                                    size_t dstOffset = input[3] * input[1] * y + input[1] * x + c;
                                    pDst[dstOffset] = pSrc[srcOffset];
                                }
                            }
                        }
                        pSrc += input[1] * input[2] * input[3];
                        pDst += input[1] * input[2] * input[3];
                    }
                }

                break;
            }
            default:
                SYNET_ERROR("Unsupported layer type " << Cpl::ToStr(layer.type()) << " to convert weight !");
            }
            return true;
        }

        template<class T> static bool AllAreEqualTo(const std::vector<T>& vector, T value)
        {
            for (size_t i = 0; i < vector.size(); ++i)
                if (vector[i] != value)
                    return false;
            return true;
        }
    };
}