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
    bool ConvertNonMaxSuppressionNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& bin, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 4, 5))
            return false;

        const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
        if (src2 == NULL || src2->type() != LayerTypeMeta || src2->meta().type() != MetaTypeConst)
            return false;
        layer.nonMaxSuppression().maxOutputBoxesPerClass() = src2->meta().alpha().i64()[0];

        const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
        if (src3 == NULL || src3->type() != LayerTypeConst)
            return false;
        const float* wgt3 = GetWeight<float>(bin, src3->weight()[0]);
        if (wgt3 == NULL)
            return false;
        layer.nonMaxSuppression().iouThreshold() = wgt3[0];

        if (layer.src().size() > 4)
        {
            const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
            if (src4 == NULL || src4->type() != LayerTypeConst)
                return false;
            const float* wgt4 = GetWeight<float>(bin, src4->weight()[0]);
            if (wgt4 == NULL)
                return false;
            layer.nonMaxSuppression().scoreThreshold() = wgt4[0];
        }

        layer.type() = Synet::LayerTypeNonMaxSuppression;
        layer.src().resize(2);

        return true;
    }
}

#endif