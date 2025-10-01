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
#include "Synet/Tensor.h"

namespace Synet
{
    typedef std::vector<uint8_t> Bytes;
    typedef std::map<String, TensorFormat> TensorFormatMap;
    typedef std::vector<Synet::LayerParam> LayerParams;

    //-------------------------------------------------------------------------------------------------

    template<class T> SYNET_INLINE bool AllEqualTo(const std::vector<T>& vector, T value)
    {
        for (size_t i = 0; i < vector.size(); ++i)
            if (vector[i] != value)
                return false;
        return true;
    }

    SYNET_INLINE bool Equal(float a, float b, float e = 0.000001f)
    {
        return abs(a - b) < e;
    }

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE bool IsAdd(const LayerParam& layer)
    {
        if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeSum &&
            (layer.eltwise().coefficients().empty() || layer.eltwise().coefficients() == Floats({ 1.0f, 1.0f })) && layer.src().size() == 2)
            return true;
        if (layer.type() == LayerTypeAdd)
            return true;
        return false;
    }

    SYNET_INLINE bool IsMul(const LayerParam& layer)
    {
        if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeProduct && layer.src().size() == 2)
            return true;
        if (layer.type() == LayerTypeMul)
            return true;
        return false;
    }

    SYNET_INLINE bool IsSub(const LayerParam& layer)
    {
        if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeSum &&
            layer.eltwise().coefficients() == Floats({ 1.0f, -1.0f }) && layer.src().size() == 2)
            return true;
        if (layer.type() == LayerTypeBinaryOperation && layer.binaryOperation().type() == BinaryOperationTypeSub)
            return true;
        return false;
    }

    SYNET_INLINE bool IsMulConst(const LayerParam& layer, float value, float epsilon = 0.000001)
    {
        if (layer.type() == LayerTypePower && layer.power().power() == 1.0f && layer.power().shift() == 0.0f
            && abs(layer.power().scale() - value) < epsilon)
            return true;
        return false;
    }

    SYNET_INLINE bool IsAddConst(const LayerParam& layer, float value, float epsilon = 0.000001)
    {
        if (layer.type() == LayerTypePower && layer.power().power() == 1.0f && layer.power().scale() == 1.0f
            && abs(layer.power().shift() - value) < epsilon)
            return true;
        return false;
    }

    SYNET_INLINE bool IsMetaConst64i(const LayerParam& layer, Longs value = Longs())
    {
        if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst &&
            layer.meta().alpha().type() == TensorType64i && (value.empty() || layer.meta().alpha().i64() == value))
            return true;
        return false;
    }

    SYNET_INLINE bool IsMetaConst(const LayerParam& layer)
    {
        if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst)
            return true;
        return false;
    }

    SYNET_INLINE bool IsDeptwiseConvolution(const LayerParam& layer, const Shape& kernel, const Shape& stride, bool bias, ActivationFunctionType activation)
    {
        if (layer.type() == LayerTypeConvolution && layer.convolution().group() == layer.convolution().outputNum() &&
            layer.convolution().stride() == stride &&
            layer.convolution().kernel() == kernel &&
            layer.convolution().biasTerm() == bias &&
            layer.convolution().activationType() == activation)
            return true;
        return false;
    }

    //-------------------------------------------------------------------------------------------------

    inline bool CheckSignificantDims(const Shape& shape, size_t dims, const String& desc)
    {
        if (SignificantDimsCount(shape) != dims)
            SYNET_ERROR("Wrong " << desc << " shape " << ToStr(shape) << " !");
        return true;
    }

    inline bool CheckDims(const Shape& shape, size_t dims, const String& desc)
    {
        if (shape.size() != dims)
            SYNET_ERROR("Wrong " << desc << " shape " << ToStr(shape) << " !");
        return true;
    }

    inline bool CheckSourceNumber(const LayerParam& layer, size_t size)
    {
        if (layer.src().size() != size)
            SYNET_ERROR("Wrong number of sources (" << layer.src().size() << " instead of " << size << ") !");
        return true;
    }

    inline bool CheckSourceNumber(const LayerParam& layer, size_t min, size_t max)
    {
        if (layer.src().size() < min || layer.src().size() > max)
            SYNET_ERROR("Wrong number of sources (" << layer.src().size() << ". It must be in range [" << min << ", " << max << "] !");
        return true;
    }

    inline bool CheckDestinationNumber(const LayerParam& layer, size_t size)
    {
        if (layer.dst().size() != size)
            SYNET_ERROR("Wrong number of destinations (" << layer.dst().size() << " instead of " << size << ") !");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> inline T* GetWeight(Bytes& bin, size_t offset)
    {
        if (offset >= bin.size())
            SYNET_ERROR("Binary storage access overflow: " << offset << " >= " << bin.size() << " !");
        return (T*)(bin.data() + offset);
    }

    template<class T> inline const T* GetWeight(const Bytes& bin, size_t offset)
    {
        if (offset >= bin.size())
            SYNET_ERROR("Binary storage access overflow: " << offset << " >= " << bin.size() << " !");
        return (const T*)(bin.data() + offset);
    }

    template<class T> inline T* GetWeight(Bytes& bin, const WeightParam& param)
    {
        if (param.offset() + param.size() > bin.size())
            SYNET_ERROR("Binary storage access overflow: " << param.offset() + param.size() << " > " << bin.size() << " !");
        return GetWeight<T>(bin, param.offset());
    }

    template<class T> inline const T* GetWeight(const Bytes& bin, const WeightParam& param)
    {
        if (param.offset() + param.size() > bin.size())
            SYNET_ERROR("Binary storage access overflow: " << param.offset() + param.size() << " > " << bin.size() << " !");
        return GetWeight<T>(bin, param.offset());
    }

    template<class T> inline void PushBack(Bytes& bin, const T& value)
    {
        size_t offset = bin.size();
        bin.resize(offset + sizeof(T));
        GetWeight<T>(bin, offset)[0] = value;
    }

    inline void Append(Bytes& bin, const WeightParam& param, const void* src)
    {
        size_t offset = bin.size();
        if (param.offset() != offset)
            CPL_LOG_SS(Warning, "Binary storage wrong append offset: " << param.offset() << " != " << offset << " !");
        bin.resize(offset + param.size());
        memcpy(bin.data() + offset, src, param.size());
    }

    inline size_t WeightUserCount(const LayerParams& layers, const WeightParam& weight)
    {
        size_t users = 0;
        for (size_t l = 0; l < layers.size(); ++l)
        {
            for (size_t w = 0; w < layers[l].weight().size(); w++)
            {
                if (layers[l].weight()[w].offset() == weight.offset())
                    users++;
            }
        }
        return users;
    }

    //-------------------------------------------------------------------------------------------------

    struct Pin
    {
        String name;
        int index;
        Pin(const String& n = String(), int i = 0) : name(n), index(i) {}
    };

    inline Pin ParsePin(const String& name)
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

    inline const LayerParam* GetLayer(const LayerParams& layers, const String& name)
    {
        Pin pin = ParsePin(name);
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (pin.name == layers[i].name())
                return &layers[i];
            for (size_t d = 0; d < layers[i].dst().size(); ++d)
                if (pin.name == layers[i].dst()[d])
                    return &layers[i];
        }
        SYNET_ERROR("Can't found layer " << pin.name << " !");
    }

    inline LayerParam* GetLayer(LayerParams& layers, const String& name)
    {
        Pin pin = ParsePin(name);
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (pin.name == layers[i].name())
                return &layers[i];
            for (size_t d = 0; d < layers[i].dst().size(); ++d)
                if (pin.name == layers[i].dst()[d])
                    return &layers[i];
        }
        SYNET_ERROR("Can't found layer " << pin.name << " !");
    }

    inline size_t GetLayerIndex(const LayerParams& layers, const String& name)
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

    inline const LayerParam* GetWeightLayer(const LayerParams& layers, const String& name, bool* shared = NULL)
    {
        const LayerParam* curr = GetLayer(layers, name);
        if (curr == NULL || curr->type() == LayerTypeConst)
        {
            if (shared)
                *shared = false;
            return curr;
        }
        if (curr->type() == LayerTypeStub)
        {
            if (curr->src().size() != 1)
                SYNET_ERROR("Stub layer " << name << " has wrong inputs number!");
            const LayerParam* next = GetLayer(layers, curr->src()[0]);
            if (next == NULL || next->type() == LayerTypeConst)
            {
                if (shared)
                    *shared = true;
                return next;
            }
        }
        SYNET_ERROR("Can't found weight " << name << " !");
    }

    inline const LayerParam* GetLayerByName(const LayerParams& layers, const String& name)
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i].name() == name)
                return &layers[i];
        }
        return NULL;
    }

    inline size_t GetIndexByName(const LayerParams& layers, const String& name)
    {
        size_t i = 0;
        for (; i < layers.size(); ++i)
            if (layers[i].name() == name)
                break;
        return i;
    }

    inline LayerType GetLayerType(const LayerParams& layers, const String& name)
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i].name() == name)
                return layers[i].type();
            for (size_t d = 0; d < layers[i].dst().size(); ++d)
                if (layers[i].dst()[d] == name)
                    return layers[i].type();
        }
        CPL_LOG_SS(Error, "Can't find layer " << name << " !");
        return LayerTypeUnknown;
    }

    //-------------------------------------------------------------------------------------------------

    inline TensorFormat Cache(const LayerParam& layer, TensorFormat value, TensorFormatMap* tensorFormatMap = NULL)
    {
        if (tensorFormatMap)
            (*tensorFormatMap)[layer.name()] = value;
        return value;
    }

    inline TensorFormat CurrentTensorFormat(const LayerParams& layers, size_t current, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, TensorFormatMap* tensorFormatMap = NULL)
    {
        const LayerParam& layer = layers[current];
        if (tensorFormatMap && tensorFormatMap->find(layer.name()) != tensorFormatMap->end())
            return (*tensorFormatMap)[layer.name()];
        if (layer.type() == LayerTypeConvolution || layer.type() == LayerTypeDeconvolution)
            return Cache(layer, layer.weight()[0].format(), tensorFormatMap);
        if (layer.type() == LayerTypePermute && layer.permute().format() != TensorFormatUnknown)
            return Cache(layer, layer.permute().format(), tensorFormatMap);
        if (layer.type() == LayerTypeInnerProduct)
        {
            if (layer.weight().size())
                return Cache(layer, layer.weight()[0].format(), tensorFormatMap);
            if (checkInnerProduct)
                return Cache(layer, TensorFormatNchw, tensorFormatMap);
        }
        if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
            return Cache(layer, TensorFormatNchw, tensorFormatMap);
        if (globalPooling && layer.type() == LayerTypePooling && layer.pooling().globalPooling())
            return Cache(layer, TensorFormatNchw, tensorFormatMap);
        if (layer.type() == LayerTypeInput && layer.input().shape().size())
            return Cache(layer, layer.input().shape()[0].format(), tensorFormatMap);
        if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeShape && layer.meta().version() == 2)
            return Cache(layer, TensorFormatNchw, tensorFormatMap);
        for (size_t s = 0; s < layer.src().size(); ++s)
        {
            const String& src = layer.src()[s];
            for (size_t l = 0; l < current; ++l)
            {
                for (size_t d = 0; d < layers[l].dst().size(); ++d)
                {
                    if (tensorFormatMap && IsMetaConst(layers[l]))
                        continue;
                    if (!tensorFormatMap && layers[l].type() == LayerTypeMeta)
                        continue;
                    if (layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                        continue;
                    const String& dst = layers[l].dst()[d];
                    if (src == dst)
                    {
                        TensorFormat format = CurrentTensorFormat(layers, l, checkInnerProduct, checkPriorBox, globalPooling, tensorFormatMap);
                        Cache(layers[l], format, tensorFormatMap);
                        if (format != TensorFormatUnknown)
                            return format;
                    }
                }
            }
        }
        return TensorFormatUnknown;
    }

    inline TensorFormat CurrentTensorFormat(const LayerParams& layers, const Strings& names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, TensorFormatMap* tensorFormatMap = NULL)
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
                        TensorFormat format = CurrentTensorFormat(layers, l, checkInnerProduct, checkPriorBox, globalPooling, tensorFormatMap);
                        if (format != TensorFormatUnknown)
                            return format;
                    }
                }
            }
        }
        return TensorFormatUnknown;
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> inline bool ReorderWeight(const Bytes& srcBin, const Shape& input, LayerParam& layer, Bytes& dstBin)
    {
        WeightParam& weight = layer.weight()[0];
        const T* pSrc = GetWeight<T>(srcBin, weight);
        T* pDst = GetWeight<T>(dstBin, weight);
        Shape& shape = weight.dim();
        weight.format() = TensorFormatNhwc;
        switch (layer.type())
        {
        case LayerTypeConvolution:
        case LayerTypeQuantizedConvolution:
        {
            shape = Shape({ shape[2], shape[3], shape[1], shape[0] });
            Tensor<float> dst((uint8_t*)pDst, weight.size(), weight.type(), shape, weight.format());
            for (size_t o = 0; o < shape[3]; ++o)
                for (size_t i = 0; i < shape[2]; ++i)
                    for (size_t y = 0; y < shape[0]; ++y)
                        for (size_t x = 0; x < shape[1]; ++x)
                            dst.Data<T>(Shape({ y, x, i, o }))[0] = *pSrc++;
            break;
        }
        case LayerTypeDeconvolution:
        {
            shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
            Tensor<float> dst((uint8_t*)pDst, weight.size(), TensorType32f, shape, weight.format());
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t c = 0; c < shape[3]; ++c)
                    for (size_t y = 0; y < shape[1]; ++y)
                        for (size_t x = 0; x < shape[2]; ++x)
                            dst.Data<T>(Shape({ i, y, x, c }))[0] = *pSrc++;
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

    inline bool ReorderWeight(const Bytes& srcBin, const Shape& input, LayerParam& layer, Bytes& dstBin)
    {
        if (layer.weight().size() < 1)
            SYNET_ERROR("There is no weight to reorder!");
        const WeightParam& weight = layer.weight()[0];
        switch (weight.type())
        {
        case TensorType32f: return ReorderWeight<float>(srcBin, input, layer, dstBin);
        case TensorType8i: return ReorderWeight<int8_t>(srcBin, input, layer, dstBin);
        default:
            SYNET_ERROR("ReorderWeight: unsupported type: " << weight.type() << " !");
        }
    }

    //-------------------------------------------------------------------------------------------------

    inline size_t UserCount(const LayerParams& layers, size_t index)
    {
        size_t users = 0;
        for (size_t j = index + 1; j < layers.size(); ++j)
            for (size_t k = 0; k < layers[j].src().size(); ++k)
                if (layers[j].src()[k] == layers[index].name())
                    users++;
        return users;
    }

    inline bool RemoveUnusedConst(LayerParams& layers)
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            const LayerParam& curr = layers[i];
            if (
                curr.type() == LayerTypeConst ||
                (curr.type() == LayerTypeMeta && curr.meta().type() == MetaTypeConst) ||
                (curr.type() == LayerTypeDequantizeLinear && curr.src().empty()))
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
                        if (layers[p].type() == LayerTypeConst)
                        {
                            size_t constUserCount = UserCount(layers, p);
                            if (constUserCount > 1)
                            {
                                layers.erase(layers.begin() + i);
                                i -= 1;
                            }
                            else if (constUserCount == 1)
                            {
                                layers.erase(layers.begin() + i);
                                layers.erase(layers.begin() + p);
                                i -= 2;
                            }
                            continue;
                        }
                    }
                }
                continue;
            }
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    inline bool CompactShape(Shape& shape)
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

    //-------------------------------------------------------------------------------------------------

    class SynetUtils
    {
    protected:
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::map<String, bool> PermuteMap;

        //-------------------------------------------------------------------------------------------------

        static String NotImplementedMarker()
        {
            return "~~~NOT_IMPLEMENTED~~~";
        }

        //-------------------------------------------------------------------------------------------------

        static bool Cache(const LayerParam& layer, bool value, PermuteMap* permuteMap = NULL)
        {
            if (permuteMap)
                (*permuteMap)[layer.name()] = value;
            return value;
        }

        static bool PermutedToNchw(const LayerParams& layers, size_t current, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, PermuteMap *permuteMap = NULL)
        {
            const LayerParam& layer = layers[current];
            if (permuteMap && permuteMap->find(layer.name()) != permuteMap->end())
                return (*permuteMap)[layer.name()];
            if (layer.type() == LayerTypeConvolution && layer.weight()[0].format() == TensorFormatNhwc)
                return Cache(layer, false, permuteMap);
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNhwc)
                return Cache(layer, false, permuteMap);
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNchw)
                return Cache(layer, true, permuteMap);
            if (checkInnerProduct && layer.type() == LayerTypeInnerProduct)
                return Cache(layer, true, permuteMap);
            if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
                return Cache(layer, true, permuteMap);
            if (globalPooling && layer.type() == LayerTypePooling && layer.pooling().globalPooling())
                return Cache(layer, true, permuteMap);
            if (layer.type() == LayerTypeInput && layer.input().shape().size() > 0 &&
                layer.input().shape()[0].format() == TensorFormatNchw)
                return Cache(layer, true, permuteMap);
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                const String & src = layer.src()[s];
                for (size_t l = 0; l < current; ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        if (permuteMap && IsMetaConst(layers[l]))
                            continue;
                        if (!permuteMap && layers[l].type() == LayerTypeMeta)
                            continue;
                        if (layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        const String & dst = layers[l].dst()[d];
                        if (src == dst && PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling, permuteMap))
                            return Cache(layer, true, permuteMap);
                    }
                }
            }
            return Cache(layer, false, permuteMap);
        }

        static bool PermutedToNchw(const LayerParams& layers, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, PermuteMap* permuteMap = NULL)
        {
            size_t start = layers.size() - 1;
            if (layers[start].type() == LayerTypeConst && start)
                start--;
            return PermutedToNchw(layers, start, checkInnerProduct, checkPriorBox, globalPooling, permuteMap);
        }

        static bool PermutedToNchw(const LayerParams& layers, const Strings & names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, PermuteMap* permuteMap = NULL)
        {
            for (size_t s = 0; s < names.size(); ++s)
            {
                for (size_t l = 0; l < layers.size(); ++l)
                {
                    for (size_t d = 0; d < layers[l].dst().size(); ++d)
                    {
                        const String & dst = layers[l].dst()[d];
                        if (permuteMap && IsMetaConst(layers[l]))
                            continue;
                        if (!permuteMap && layers[l].type() == LayerTypeMeta)
                            continue;
                        if (layers[l].type() == LayerTypeConst || layers[l].type() == LayerTypeUnknown)
                            continue;
                        if (names[s] == dst && PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling, permuteMap))
                            return true;
                    }
                }
            }
            return false;
        }

        static int PermutedToNchw(const LayerParams& layers, const Strings& names, bool checkInnerProduct, bool checkPriorBox, bool globalPooling, Ints & stat, PermuteMap* permuteMap = NULL)
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
                            PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox, globalPooling, permuteMap))
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
    };
}