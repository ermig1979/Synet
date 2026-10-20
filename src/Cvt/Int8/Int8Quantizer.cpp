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
    }

    Int8Quantizer::Int8Quantizer(const Int8OptParam& param)
        : _param(param)
    {
    }

    bool Int8Quantizer::Run(Synet::NetworkParam& network, Bytes& bin)
    {
        if (!_param.enable())
            return true;
        size_t converted = 0;
        LayerParams& layers = network.layers();
        for (size_t i = 0; i < layers.size(); ++i)
        {
            LayerParam& layer = layers[i];
            bool changed = false;
            if (!ConvertInnerProduct(layer, bin, changed))
                return false;
            if (changed)
                converted++;
        }
        if (converted)
        {
            CPL_LOG_SS(Info, "Int8Quantizer: converted " << converted << " InnerProduct layer(s) to DynamicQuantizedInnerProduct.");
            if (!CompactWeights(layers, bin))
                return false;
        }
        return true;
    }

    bool Int8Quantizer::ConvertInnerProduct(LayerParam& layer, Bytes& bin, bool& converted)
    {
        converted = false;
        if (layer.type() != LayerTypeInnerProduct || layer.src().size() != 1)
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

        const bool biasTerm = layer.innerProduct().biasTerm();
        const bool prelu = layer.innerProduct().activationType() == ActivationFunctionTypePrelu;
        const size_t need = 1 + (biasTerm ? 1 : 0) + (prelu ? 1 : 0);
        if (layer.weight().size() < need)
            SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' has " << layer.weight().size() << " weights instead of " << need << "!");

        std::vector<float> matrix(K * N);
        memcpy(matrix.data(), GetWeight<float>(bin, srcWeight), K * N * sizeof(float));

        std::vector<float> bias;
        size_t next = 1;
        if (biasTerm)
        {
            const WeightParam& biasWeight = layer.weight()[next++];
            if (biasWeight.type() != TensorType32f || TensorSize(biasWeight.dim()) != N)
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias must be FP32 of size " << N << "!");
            if (biasWeight.size() < N * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' bias is truncated!");
            bias.resize(N);
            memcpy(bias.data(), GetWeight<float>(bin, biasWeight), N * sizeof(float));
        }

        std::vector<float> preluWeight;
        Shape preluDim;
        TensorFormat preluFormat = TensorFormatNchw;
        if (prelu)
        {
            const WeightParam& srcPrelu = layer.weight()[next++];
            size_t count = TensorSize(srcPrelu.dim());
            if (srcPrelu.type() != TensorType32f || srcPrelu.size() < count * sizeof(float))
                SYNET_ERROR("Int8Quantizer: layer '" << layer.name() << "' Prelu weight must be FP32!");
            preluWeight.resize(count);
            memcpy(preluWeight.data(), GetWeight<float>(bin, srcPrelu), count * sizeof(float));
            preluDim = srcPrelu.dim();
            preluFormat = srcPrelu.format();
        }

        std::vector<int8_t> quantized;
        std::vector<float> scale;
        if (!QuantizeWeight(matrix.data(), K, N, transB, quantized, scale))
            return false;
        std::vector<int8_t> zero(N, 0);

        const TensorFormat format = srcWeight.format();
        std::vector<WeightParam> weights(3 + (biasTerm ? 1 : 0) + (prelu ? 1 : 0));
        size_t index = 0;
        AppendWeight(bin, quantized.data(), quantized.size() * sizeof(int8_t), Shp(K, N), TensorType8i, format, weights[index++]);
        AppendWeight(bin, zero.data(), zero.size() * sizeof(int8_t), Shp(N), TensorType8i, TensorFormatNchw, weights[index++]);
        AppendWeight(bin, scale.data(), scale.size() * sizeof(float), Shp(N), TensorType32f, TensorFormatNchw, weights[index++]);
        if (biasTerm)
            AppendWeight(bin, bias.data(), bias.size() * sizeof(float), Shp(N), TensorType32f, TensorFormatNchw, weights[index++]);
        if (prelu)
            AppendWeight(bin, preluWeight.data(), preluWeight.size() * sizeof(float), preluDim, TensorType32f, preluFormat, weights[index++]);

        layer.type() = LayerTypeDynamicQuantizedInnerProduct;
        layer.innerProduct().outputNum() = (uint32_t)N;
        //layer.innerProduct().transposeB() = true;
        layer.weight() = weights;
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
