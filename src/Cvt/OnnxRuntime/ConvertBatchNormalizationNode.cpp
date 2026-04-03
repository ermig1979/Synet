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
    bool ConvertBatchNormalizationNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered)
    {
        if (!CheckSourceNumber(layer, 5))
            return false;

        const LayerParam* src1 = GetWeightLayer(layers, layer.src()[1]);
        if (src1 == NULL || src1->type() != LayerTypeConst)
            SYNET_ERROR("BatchNormalization src[1] must be Const type!");
        const float* gamma = GetWeight<float>(original, src1->weight()[0]);

        const LayerParam* src2 = GetWeightLayer(layers, layer.src()[2]);
        if (src2 == NULL || src2->type() != LayerTypeConst)
            SYNET_ERROR("BatchNormalization src[2] must be Const type!");
        const float* beta = GetWeight<float>(original, src2->weight()[0]);

        const LayerParam* src3 = GetWeightLayer(layers, layer.src()[3]);
        if (src3 == NULL || src3->type() != LayerTypeConst)
            SYNET_ERROR("BatchNormalization src[3] must be Const type!");
        const float* mean = GetWeight<float>(original, src3->weight()[0]);

        const LayerParam* src4 = GetWeightLayer(layers, layer.src()[4]);
        if (src4 == NULL || src4->type() != LayerTypeConst)
            SYNET_ERROR("BatchNormalization src[4] must be Const type!");
        const float* var = GetWeight<float>(original, src4->weight()[0]);

        float epsilon, momentum;
        if (!ConvertAtrributeFloat(node, "epsilon", epsilon))
            return false;
        if (!ConvertAtrributeFloat(node, "momentum", momentum, true, 0.9f))
            return false;

        layer.type() = Synet::LayerTypeScale;
        layer.src().resize(1);
        layer.scale().biasTerm() = true;
        layer.weight().resize(2);
        layer.weight()[0] = src1->weight()[0];
        if (WeightUserCount(layers, layer.weight()[0]) > 1)
        {
            size_t size = TensorSize(layer.weight()[0].dim()), offset = reordered.size();
            original.resize(offset + size * 4);
            reordered.resize(offset + size * 4);
            layer.weight()[0].offset() = offset;
        }
        layer.weight()[1] = src2->weight()[0];
        if (WeightUserCount(layers, layer.weight()[1]) > 1)
        {
            size_t size = TensorSize(layer.weight()[1].dim()), offset = reordered.size();
            original.resize(offset + size * 4);
            reordered.resize(offset + size * 4);
            layer.weight()[1].offset() = offset;
        }
        float* scale = GetWeight<float>(reordered, layer.weight()[0]);
        float* shift = GetWeight<float>(reordered, layer.weight()[1]);
        size_t channels = layer.weight()[0].dim()[0];
        for (size_t c = 0; c < channels; c++)
        {
            scale[c] = gamma[c] / sqrt(var[c] + epsilon);
            shift[c] = -scale[c] * mean[c] + beta[c];
        }
        return true;
    }
}

#endif