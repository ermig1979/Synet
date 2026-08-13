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

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    bool ConvertGridSampleNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 2))
            return false;
        const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
        const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
        if (src0 == NULL || src1 == NULL)
            return false;

        layer.type() = LayerTypeGridSample;
        if (!ConvertAtrributeInt(node, "align_corners", layer.gridSample().alignCorners()))
            return false;

        String interpMode;
        if (!ConvertAtrributeString(node, "mode", interpMode))
            return false;
        if (interpMode == "bilinear")
            layer.gridSample().interpMode() = GridSampleInterpModeBilinear;
        else if (interpMode == "nearest")
            layer.gridSample().interpMode() = GridSampleInterpModeNearest;
        else if (interpMode == "bicubic")
            layer.gridSample().interpMode() = GridSampleInterpModeBicubic;
        else
            SYNET_ERROR("Unsupported interpolation mode '" << interpMode << "' !");

        String paddingMode;
        if (!ConvertAtrributeString(node, "padding_mode", paddingMode))
            return false;
        if (paddingMode == "zeros")
            layer.gridSample().paddingMode() = GridSamplePaddingModeZeros;
        else if (paddingMode == "border")
            layer.gridSample().paddingMode() = GridSamplePaddingModeBorder;
        else if (paddingMode == "reflection")
            layer.gridSample().paddingMode() = GridSamplePaddingModeReflection;
        else
            SYNET_ERROR("Unsupported padding mode '" << paddingMode << "' !");

        return true;
    }
}

#endif
