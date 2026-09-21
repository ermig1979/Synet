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

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    template<class T> inline bool ReorderDynamicConvolutionWeight(const Bytes& srcBin, const Shape& input, LayerParam& layer, Bytes& dstBin)
    {

        WeightParam& weight = layer.weight()[0];
        const T* pSrc = GetWeight<T>(srcBin, weight);
        T* pDst = GetWeight<T>(dstBin, weight);
        weight.format() = TensorFormatNhwc;
        Shape shape = Shp(weight.dim()[0], input[2], input[3], input[1], input[0]);
        Tensor dst((uint8_t*)pDst, weight.size(), weight.type(), shape, weight.format());
        for (size_t n = 0; n < shape[0]; ++n)
            for (size_t o = 0; o < shape[4]; ++o)
                for (size_t i = 0; i < shape[3]; ++i)
                    for (size_t y = 0; y < shape[1]; ++y)
                        for (size_t x = 0; x < shape[2]; ++x)
                            dst.Data<T>(Shape({ n, y, x, i, o }))[0] = *pSrc++;
        return true;
    }

    inline bool ReorderDynamicConvolutionWeight(const Bytes& srcBin, const Shape& input, LayerParam& layer, Bytes& dstBin)
    {
        if (layer.weight().size() < 1)
            SYNET_ERROR("There is no weight to reorder!");
        const WeightParam& weight = layer.weight()[0];
        switch (weight.type())
        {
        case TensorType32f: return ReorderDynamicConvolutionWeight<float>(srcBin, input, layer, dstBin);
        case TensorType8i: return ReorderDynamicConvolutionWeight<int8_t>(srcBin, input, layer, dstBin);
        default:
            SYNET_ERROR("ReorderDynamicWeight: unsupported type: " << weight.type() << " !");
        }
    }

    bool ConvertConvOrConvTransposeNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap, UniqNames& merged)
    {
        if (node.op_type() == "Conv")
            layer.type() = Synet::LayerTypeConvolution;
        else if (node.op_type() == "ConvTranspose")
            layer.type() = Synet::LayerTypeDeconvolution;
        else
            return false;
        if (!CheckSourceNumber(layer, 2, 3))
            return false;
        if (!ConvertAtrributeInts(node, "dilations", layer.convolution().dilation(), true))
            return false;
        if (!ConvertAtrributeInt(node, "group", layer.convolution().group(), true, 1u))
            return false;
        if (!ConvertAtrributeInts(node, "kernel_shape", layer.convolution().kernel(), true))
            return false;
        String autoPad;
        ConvertAtrributeString(node, "auto_pad", autoPad, true);
        layer.convolution().autoPad() = (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER");
        if (!ConvertAtrributeInts(node, "pads", layer.convolution().pad(), true))
            return false;
        if (!ConvertAtrributeInts(node, "strides", layer.convolution().stride()))
            return false;
        if (GetLayerType(layers, layer.src()[0]) == LayerTypeDequantizeLinear &&
            GetLayerType(layers, layer.src()[1]) == LayerTypeDequantizeLinear)
        {
            const LayerParam* dequantized = GetLayer(layers, layer.src()[1]);
            if (dequantized->weight().empty())
                return false;
            const Shape& shape = dequantized->weight()[0].dim();
            layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.type() == Synet::LayerTypeConvolution)
            {
                layer.type() = Synet::LayerTypeQuantizedConvolution;
                if (!MoveDequantizeLinearToLayer(layers, layer, merged))
                    return false;
            }
            else
                return false;
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
            return true;
        }
        LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL)
            return false;
        if (src1->type() == LayerTypeReshape)
        {
            Shape shape = src1->reshape().shape();
            if (layer.convolution().kernel().empty())
            {
                if (shape.size() != 4)
                    SYNET_ERROR("Convolution weight must be 4D tensor!");
                layer.convolution().kernel() = Shp(shape[2], shape[3]);
            }
            layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.convolution().biasTerm())
            {
                const LayerParam* bias = GetWeightLayer(layers, layer.src()[2]);
                if (bias == NULL || bias->type() != LayerTypeConst)
                    return false;
                layer.weight().resize(1);
                layer.weight()[0] = bias->weight()[0];
                layer.src().resize(2);
            }
            if (trans && CurrentTensorFormat(layers, Strings({ layer.src()[0] }), false, false, false, tensorFormatMap) == TensorFormatNhwc)
            {
                LayerParam* ip = GetLayer(layers, src1->src()[0]);
                if (ip == NULL || ip->type() != LayerTypeInnerProduct)
                    return false;
                src1->reshape().shape() = Shp(shape[2], shape[3], shape[1], shape[0]);
                layer.convolution().format() = TensorFormatNhwc;
                return ReorderDynamicConvolutionWeight(srcBin, shape, *ip, dstBin);
            }
        }
        else
        {
            const LayerParam* weight = GetWeightLayer(layers, layer.src()[1]);
            if (weight == NULL || weight->type() != LayerTypeConst)
                return false;
            const Shape& shape = weight->weight()[0].dim();
            if (layer.convolution().kernel().empty())
            {
                if (shape.size() != 4)
                    SYNET_ERROR("Convolution weight must be 4D tensor!");
                layer.convolution().kernel() = Shp(shape[2], shape[3]);
            }
            layer.weight().resize(layer.src().size() - 1);
            layer.weight()[0] = weight->weight()[0];
            layer.convolution().outputNum() = uint32_t(layer.type() == Synet::LayerTypeConvolution ? shape[0] : shape[1] * layer.convolution().group());
            layer.convolution().biasTerm() = layer.src().size() > 2;
            if (layer.convolution().biasTerm())
            {
                const LayerParam* bias = GetWeightLayer(layers, layer.src()[2]);
                if (bias == NULL || bias->type() != LayerTypeConst)
                    return false;
                layer.weight()[1] = bias->weight()[0];
            }
            layer.src().resize(1);
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, false, tensorFormatMap) == TensorFormatNhwc)
                return ReorderWeight(srcBin, Shape(), layer, dstBin);
        }
        return true;
    }
}

#endif