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

namespace Synet
{
    bool ConvertPreluNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src1 == NULL)
            return false;
        layer.type() = Synet::LayerTypePrelu;
        if (src1->type() == LayerTypeConst)
        {
            layer.weight() = src1->weight();
        }
        else if (src1->type() == LayerTypeExpandDims)
        {
            if (!CheckSourceNumber(*src1, 1))
                return false;
            const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
            if (src10 == NULL || src10->type() != LayerTypeConst)
                return false;
            layer.weight() = src10->weight();
            layers.erase(layers.begin() + (src1 - layers.data()));
        }
        else if (src1->type() == LayerTypeReshape)
        {
            if (!CheckSourceNumber(*src1, 1, 2))
                return false;
            const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
            if (src10 == NULL || src10->type() != LayerTypeConst)
                return false;
            layer.weight() = src10->weight();
            layers.erase(layers.begin() + (src1 - layers.data()));
        }
        else
            SYNET_ERROR("PreluNode: can't find weight!");
        layer.src().resize(1);
        if (!CompactShape(layer.weight()[0].dim()))
            return false;
        return true;
    }
}

#endif