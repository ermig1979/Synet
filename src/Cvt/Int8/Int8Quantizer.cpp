/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Cvt/Int8/Int8Quantizer.h"

#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Utils/ConvParam.h"

#include "Simd/SimdLib.h"

namespace Synet
{
    namespace
    {
        bool Overflow16i()
        {
#if defined(__MINGW32__) || defined(__MINGW64__)
            return true;
#else
            return SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#endif
        }

        void AppendWeight(Bytes& bin, const void* data, size_t bytes, const Shape& dim, TensorType type, TensorFormat format, WeightParam& param)
        {
            param.dim() = dim;
            param.type() = type;
            param.format() = format;
            param.offset() = bin.size();
            param.size() = bytes;
            size_t offset = bin.size();
            bin.resize(offset + bytes);
            if (bytes)
                memcpy(bin.data() + offset, data, bytes);
        }

        bool QuantizeWeight(const float* src, size_t K, size_t N, bool transB, std::vector<int8_t>& dst, std::vector<float>& scale)
        {
            const bool overflow = Overflow16i();
            const int lo = overflow ? -64 : -128;
            const int hi = overflow ? 63 : 127;
            const float qmax = overflow ? 63.0f : 127.0f;
            dst.resize(K * N);
            scale.resize(N);
            for (size_t j = 0; j < N; ++j)
            {
                float maxAbs = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    float value = transB ? src[k * N + j] : src[j * K + k];
                    maxAbs = std::max(maxAbs, std::abs(value));
                }
                float range = std::max(0.000001f, maxAbs);
                scale[j] = range / qmax;
                float invScale = qmax / range;
                for (size_t k = 0; k < K; ++k)
                {
                    float value = transB ? src[k * N + j] : src[j * K + k];
                    dst[k * N + j] = (int8_t)RestrictRange<int>(NearByInt(value * invScale), lo, hi);
                }
            }
            return true;
        }

        void TransposeWeight(WeightParam &param, Bytes& bin)
        {
            const size_t K = param.dim()[1], N = param.dim()[0];
            Floats buf(K * N);
            float* src = GetWeight<float>(bin, param), *dst = buf.data();
            for (size_t k = 0; k < K; k++)
                for (size_t j = 0; j < N; j++)
                    dst[k * N + j] = src[j * K + k];
            memcpy(src, dst, K * N * sizeof(float));
            param.dim() = Shp(K, N);
        }

        bool IsOnes(const Shape& shape, bool emptyMeansOne)
        {
            if (shape.empty())
                return emptyMeansOne;
            if (shape.size() == 1)
                return shape[0] == 1;
            return shape.size() == 2 && shape[0] == 1 && shape[1] == 1;
        }

        bool IsZeroPad(const Shape& pad)
        {
            if (pad.size() == 3 || pad.size() > 4)
                return false;
            for (size_t i = 0; i < pad.size(); ++i)
                if (pad[i] != 0)
                    return false;
            return true;
        }

        bool SupportedActivation(ActivationFunctionType type)
        {
            switch (type)
            {
            case ActivationFunctionTypeIdentity:
            case ActivationFunctionTypeRelu:
            case ActivationFunctionTypeLeakyRelu:
            case ActivationFunctionTypeRestrictRange:
            case ActivationFunctionTypePrelu:
            case ActivationFunctionTypeElu:
            case ActivationFunctionTypeHswish:
            case ActivationFunctionTypeMish:
            case ActivationFunctionTypeHardSigmoid:
            case ActivationFunctionTypeSwish:
            case ActivationFunctionTypeGelu:
                return true;
            default:
                return false;
            }
        }

        bool NameUsed(const std::vector<LayerParam>& layers, const String& name)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                if (layers[i].name() == name)
                    return true;
                for (size_t d = 0; d < layers[i].dst().size(); ++d)
                    if (layers[i].dst()[d] == name)
                        return true;
            }
            return false;
        }

        String FreshName(const std::vector<LayerParam>& layers, const String& base)
        {
            if (!NameUsed(layers, base))
                return base;
            for (size_t i = 0; ; ++i)
            {
                String name = base + "_" + std::to_string(i);
                if (!NameUsed(layers, name))
                    return name;
            }
        }

        bool SharesSrcDst(const LayerParam& layer)
        {
            for (size_t s = 0; s < layer.src().size(); ++s)
                for (size_t d = 0; d < layer.dst().size(); ++d)
                    if (layer.src()[s] == layer.dst()[d])
                        return true;
            return false;
        }
    }

    Int8Quantizer::Int8Quantizer(const Int8OptParam& param)
        : _param(param)
    {
    }

    bool Int8Quantizer::Run(Synet::NetworkParam& network, Bytes& bin)
    {
        if (!_param.enable())
            return true;
        size_t convertedIp = 0, convertedConv = 0;
        LayerParams& layers = network.layers();
        for (size_t i = 0; i < layers.size(); ++i)
        {
            bool changed = false;
            if (!ConvertInnerProduct(layers[i], bin, changed))
                return false;
            if (changed)
                convertedIp++;
            changed = false;
            if (!ConvertConvolution(layers, i, bin, changed))
                return false;
            if (changed)
                convertedConv++;
        }
        if (convertedIp || convertedConv)
        {
            CPL_LOG_SS(Info, "Int8Quantizer: converted " << convertedIp << " InnerProduct and " << convertedConv << " Convolution layer(s) to DynamicQuantizedInnerProduct.");
            if (!CompactWeights(layers, bin))
                return false;
        }
        return true;
    }

    bool Int8Quantizer::ConvertInnerProduct(LayerParam& layer, Bytes& bin, bool& converted)
    {
        converted = false;
        if (layer.type() != LayerTypeInnerProduct || !_param.innerProduct())
            return true;

        const bool biasTerm = layer.innerProduct().biasTerm();
        const bool prelu = layer.innerProduct().activationType() == ActivationFunctionTypePrelu;
        const size_t need = (layer.src().size() == 1 ? 1 : 0) + (biasTerm ? 1 : 0) + (prelu ? 1 : 0);
        if (layer.weight().size() < need)
            SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' has " << layer.weight().size() << " weights instead of " << need << "!");

        size_t srcIndex = 0;
        std::vector<WeightParam> weights;
        if (_param.innerProductNoConstWeight() && layer.src().size() == 2)
        {

        }
        else
        {
            if (layer.src().size() != 1)
                return true;
            if (layer.weight().empty())
                return true;
            const WeightParam& srcWeight = layer.weight()[0];
            if (srcWeight.type() != TensorType32f || srcWeight.dim().size() != 2)
                return true;

            const bool transB = layer.innerProduct().transposeB();
            const size_t K = transB ? srcWeight.dim()[0] : srcWeight.dim()[1];
            const size_t N = transB ? srcWeight.dim()[1] : srcWeight.dim()[0];
            if (K < _param.minK() || N < _param.minN())
                return true;
            if (layer.innerProduct().transposeB())
                return true;
            if (layer.innerProduct().outputNum() != 0 && layer.innerProduct().outputNum() != N)
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' outputNum does not match weight shape!");
            if (srcWeight.size() < K * N * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' weight[0] is truncated!");
            
            srcIndex++;
            weights.resize(1);
            weights[0] = srcWeight;
            if (!transB)
                TransposeWeight(weights[0], bin);
            layer.innerProduct().outputNum() = (uint32_t)N;
        }

        if (biasTerm)
        {
            const WeightParam& biasWeight = layer.weight()[srcIndex++];
            if (biasWeight.type() != TensorType32f)
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias must be FP32 !");
            if (layer.src().size() == 1 && TensorSize(biasWeight.dim()) != layer.innerProduct().outputNum())
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias must has size " << layer.innerProduct().outputNum() << "!");
            if (biasWeight.size() < layer.innerProduct().outputNum() * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias is truncated!");
            weights.push_back(biasWeight);
        }
        if (prelu)
        {
            const WeightParam& srcPrelu = layer.weight()[srcIndex++];
            size_t count = TensorSize(srcPrelu.dim());
            if (srcPrelu.type() != TensorType32f || srcPrelu.size() < count * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' Prelu weight must be FP32!");
            weights.push_back(srcPrelu);
        }

        layer.type() = LayerTypeDynamicQuantizedInnerProduct;
        layer.weight() = weights;
        converted = true;
        return true;
    }

    bool Int8Quantizer::ConvertConvolution(LayerParams& layers, size_t& index, Bytes& bin, bool& converted)
    {
        converted = false;
        LayerParam& layer = layers[index];
        if (layer.type() != LayerTypeConvolution || !_param.convolution())
            return true;
        if (layer.src().size() != 1 || layer.dst().empty() || SharesSrcDst(layer))
            return true;

        const ConvolutionParam& conv = layer.convolution();
        if (conv.group() != 1 || conv.autoPad())
            return true;
        if (conv.kernel() != Shp(1, 1))
            return true;
        if (!IsOnes(conv.stride(), true) || !IsOnes(conv.dilation(), true) || !IsZeroPad(conv.pad()))
            return true;
        if (!SupportedActivation(conv.activationType()))
            return true;
        if (conv.outputNum() < _param.minN())
            return true;
        if (layer.weight().empty())
            return true;

        const WeightParam& srcWeight = layer.weight()[0];
        if (srcWeight.type() != TensorType32f || srcWeight.dim().size() != 4)
            return true;
        const bool nhwc = srcWeight.format() == TensorFormatNhwc;
        const bool nchw = srcWeight.format() == TensorFormatNchw;
        if (!nhwc && !nchw)
            return true;
        if (nchw && layer.dst().size() != 1)
            return true;

        size_t srcC = 0, dstC = 0;
        if (nhwc)
        {
            if (srcWeight.dim()[0] != 1 || srcWeight.dim()[1] != 1)
                return true;
            srcC = srcWeight.dim()[2];
            dstC = srcWeight.dim()[3];
        }
        else
        {
            if (srcWeight.dim()[2] != 1 || srcWeight.dim()[3] != 1)
                return true;
            dstC = srcWeight.dim()[0];
            srcC = srcWeight.dim()[1];
        }
        if (srcC < _param.minK())
            return true;
        if (conv.outputNum() != dstC)
            SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' outputNum does not match weight shape!");
        if (srcWeight.size() < srcC * dstC * sizeof(float))
            SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' weight[0] is truncated!");

        const bool biasTerm = conv.biasTerm();
        const bool prelu = conv.activationType() == ActivationFunctionTypePrelu;
        const size_t need = 1 + (biasTerm ? 1 : 0) + (prelu ? 1 : 0);
        if (layer.weight().size() < need)
            SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' has " << layer.weight().size() << " weights instead of " << need << "!");

        size_t srcIndex = 1;
        if (biasTerm)
        {
            const WeightParam& biasWeight = layer.weight()[srcIndex++];
            if (biasWeight.type() != TensorType32f || TensorSize(biasWeight.dim()) != dstC)
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias must be FP32 of size " << dstC << "!");
            if (biasWeight.size() < dstC * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias is truncated!");
        }
        if (prelu)
        {
            const WeightParam& srcPrelu = layer.weight()[srcIndex++];
            size_t count = TensorSize(srcPrelu.dim());
            if (srcPrelu.type() != TensorType32f || (count != 1 && count != dstC) || srcPrelu.size() < count * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' Prelu weight must be FP32 of size 1 or " << dstC << "!");
        }

        WeightParam matrix = srcWeight;
        matrix.format() = TensorFormatNchw;
        if (nhwc)
            matrix.dim() = Shp(srcC, dstC);
        else if (WeightUserCount(layers, srcWeight) > 1)
        {
            std::vector<float> copy(srcC * dstC);
            memcpy(copy.data(), GetWeight<float>(bin, srcWeight), copy.size() * sizeof(float));
            AppendWeight(bin, copy.data(), copy.size() * sizeof(float), Shp(dstC, srcC), TensorType32f, TensorFormatNchw, matrix);
            TransposeWeight(matrix, bin);
        }
        else
        {
            matrix.dim() = Shp(dstC, srcC);
            TransposeWeight(matrix, bin);
        }

        std::vector<WeightParam> weights;
        weights.push_back(matrix);
        srcIndex = 1;
        if (biasTerm)
            weights.push_back(layer.weight()[srcIndex++]);
        if (prelu)
            weights.push_back(layer.weight()[srcIndex++]);

        LayerParam dqip = layer;
        dqip.type() = LayerTypeDynamicQuantizedInnerProduct;
        dqip.convolution() = ConvolutionParam();
        dqip.innerProduct().outputNum() = (uint32_t)dstC;
        dqip.innerProduct().biasTerm() = biasTerm;
        dqip.innerProduct().transposeA() = false;
        dqip.innerProduct().transposeB() = false;
        dqip.innerProduct().axis() = -1;
        dqip.innerProduct().activationType() = conv.activationType();
        dqip.innerProduct().activationParam0() = conv.activationParam0();
        dqip.innerProduct().activationParam1() = conv.activationParam1();
        dqip.weight() = weights;

        if (nhwc)
        {
            layers[index] = dqip;
        }
        else
        {
            // DynamicQuantizedInnerProduct reduces the last axis. An NCHW 1x1 convolution
            // matches that GEMM only after the activation is permuted to NHWC and back.
            const String toNhwcName = FreshName(layers, layer.name() + "_to_nhwc");
            const String nhwcName = FreshName(layers, layer.name() + "_nhwc");
            const String toNchwName = FreshName(layers, layer.name() + "_to_nchw");

            LayerParam toNhwc;
            toNhwc.type() = LayerTypePermute;
            toNhwc.name() = toNhwcName;
            toNhwc.src() = layer.src();
            toNhwc.dst().push_back(toNhwcName);
            toNhwc.permute().order() = Shp(0, 2, 3, 1);
            toNhwc.permute().format() = TensorFormatNhwc;

            dqip.src().resize(1);
            dqip.src()[0] = toNhwcName;
            dqip.dst().resize(1);
            dqip.dst()[0] = nhwcName;

            LayerParam toNchw;
            toNchw.type() = LayerTypePermute;
            toNchw.name() = toNchwName;
            toNchw.src().push_back(nhwcName);
            toNchw.dst() = layer.dst();
            toNchw.permute().order() = Shp(0, 3, 1, 2);
            toNchw.permute().format() = TensorFormatNchw;

            layers[index] = toNhwc;
            layers.insert(layers.begin() + index + 1, dqip);
            layers.insert(layers.begin() + index + 2, toNchw);
            index += 2;
        }
        converted = true;
        return true;
    }

    bool Int8Quantizer::CompactWeights(LayerParams& layers, Bytes& bin)
    {
        std::map<size_t, size_t> remap;
        Bytes compact;
        for (size_t l = 0; l < layers.size(); ++l)
        {
            for (size_t w = 0; w < layers[l].weight().size(); ++w)
            {
                WeightParam& weight = layers[l].weight()[w];
                if (weight.size() == 0 || weight.size() == (size_t)-1)
                    continue;
                size_t old = weight.offset();
                std::map<size_t, size_t>::iterator it = remap.find(old);
                if (it != remap.end())
                {
                    weight.offset() = it->second;
                    continue;
                }
                if (old + weight.size() > bin.size())
                    SYNET_ERROR("Int8Quantizer: weight is out of binary storage!");
                size_t offset = compact.size();
                compact.resize(offset + weight.size());
                memcpy(compact.data() + offset, bin.data() + old, weight.size());
                remap[old] = offset;
                weight.offset() = offset;
            }
        }
        bin.swap(compact);
        return true;
    }

    bool Int8QuantizeSynetModel(const String& srcXml, const String& srcBin, const String& dstXml, const String& dstBin, const Int8OptParam& param)
    {
        NetworkParamHolder network;
        if (!network.Load(srcXml))
        {
            std::cout << "Can't load Synet model '" << srcXml << "' !" << std::endl;
            return false;
        }
        Bytes bin;
        if (!srcBin.empty() && !LoadBinaryData(srcBin, bin))
        {
            std::cout << "Can't load Synet weight '" << srcBin << "' !" << std::endl;
            return false;
        }
        Int8OptParam effective = param;
        effective.enable() = true;
        Int8Quantizer quantizer(effective);
        if (!quantizer.Run(network(), bin))
        {
            std::cout << "Can't quantize Synet model!" << std::endl;
            return false;
        }
        if (!network.Save(dstXml, false))
        {
            std::cout << "Can't save Synet model '" << dstXml << "' !" << std::endl;
            return false;
        }
        if (!dstBin.empty() && !SaveBinaryData(bin, dstBin))
        {
            std::cout << "Can't save Synet weight '" << dstBin << "' !" << std::endl;
            return false;
        }
        return true;
    }
}
