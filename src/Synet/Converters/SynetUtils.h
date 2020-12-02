/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

namespace Synet
{
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
            {
                std::cout << "Wrong " << desc << " shape " << ShapeToStr(shape) << " !" << std::endl;
                return false;
            }
            return true;
        }

        static bool CheckSourceNumber(const LayerParam& layer, size_t size)
        {
            if (layer.src().size() != size)
            {
                std::cout << "Wrong number of sources (" << layer.src().size() << " instead of " << size << ") !" << std::endl;
                return false;
            }
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
            {
                std::cout << "Can't compact shape " << ShapeToStr(shape) << " !" << std::endl;
                return false;
            }
            shape = Shp(value);
            return true;
        }

        static const LayerParam* GetLayer(const LayerParams& layers, const String& name)
        {
            Pin pin = ParsePin(name);
            for (size_t i = 0; i < layers.size(); ++i)
                if (pin.name == layers[i].name())
                    return &layers[i];
            std::cout << "Can't found layer " << pin.name << " !" << std::endl;
            return NULL;
        }

        template<class T> static const T* GetWeight(const Vector& bin, size_t offset)
        {
            return (const T*)((const uint8_t*)bin.data() + offset);
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

        static bool PermutedToNchw(const LayerParams& layers, size_t current, bool checkInnerProduct, bool checkPriorBox)
        {
            const LayerParam& layer = layers[current];
            if (layer.type() == LayerTypeConvolution && layer.weight()[0].format() == TensorFormatNhwc)
                return false;
            if (layer.type() == LayerTypePermute && layer.permute().format() == TensorFormatNchw)
                return true;
            if (checkInnerProduct && layer.type() == LayerTypeInnerProduct)
                return true;
            if (checkPriorBox && (layer.type() == LayerTypePriorBox || layer.type() == LayerTypePriorBoxClustered))
                return true;
            for (size_t s = 0; s < layer.src().size(); ++s)
            {
                Pin src = ParsePin(layer.src()[s]);
                for (size_t l = 0; l < current; ++l)
                {
                    if (src.name == layers[l].name() && layers[l].type() != LayerTypeMeta &&
                        PermutedToNchw(layers, l, checkInnerProduct, checkPriorBox))
                        return true;
                }
            }
            return false;
        }

        static bool PermutedToNchw(const LayerParams& layers, bool checkInnerProduct, bool checkPriorBox)
        {
            size_t start = layers.size() - 1;
            if (layers[start].type() == LayerTypeConst && start)
                start--;
            return PermutedToNchw(layers, start, checkInnerProduct, checkPriorBox);
        }

        static bool RemoveUnusedConst(LayerParams& layers)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const LayerParam& layer = layers[i];
                if (layer.type() == LayerTypeConst || (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst))
                {
                    const String& name = layer.name();
                    bool unused = true;
                    for (size_t j = i + 1; j < layers.size() && unused; ++j)
                        for (size_t k = 0; k < layers[j].src().size() && unused; ++k)
                            if (layers[j].src()[k] == name)
                                unused = false;
                    if (unused)
                        layers.erase(layers.begin() + i), i--;
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
                Tensor dst(pDst, weight.size() / sizeof(float), shape, weight.format());
                for (size_t o = 0; o < shape[3]; ++o)
                    for (size_t i = 0; i < shape[2]; ++i)
                        for (size_t y = 0; y < shape[0]; ++y)
                            for (size_t x = 0; x < shape[1]; ++x)
                                dst.CpuData(Shape({ y, x, i, o }))[0] = *pSrc++;
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
                std::cout << "Unsupported layer type " << ValueToString(layer.type()) << " to convert weight !" << std::endl;
                return false;
            }
            return true;
        }

        static String ShapeToStr(const Shape& shape)
        {
            std::stringstream ss;
            ss << "{";
            for (size_t i = 0; i < shape.size(); ++i)
                ss << " " << shape[i];
            ss << " }";
            return ss.str();
        }

        static size_t TensorSize(const Shape& shape)
        {
            if (shape.empty())
                return 0;
            else
            {
                size_t size = 1;
                for (size_t i = 0; i < shape.size(); ++i)
                    size *= shape[i];
                return size;
            }
        }
    };
}