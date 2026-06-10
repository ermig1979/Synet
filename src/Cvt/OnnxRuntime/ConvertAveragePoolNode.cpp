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
    bool ConvertAveragePoolNode(const onnx::NodeProto& node, LayerParam& layer)
    {
        layer.type() = Synet::LayerTypePooling;
        layer.pooling().method() = PoolingMethodTypeAverage;
        if (!ConvertAtrributeInts(node, "kernel_shape", layer.pooling().kernel()))
            return false;
        if (GetAtrribute(node, "pads"))
        {
            if (!ConvertAtrributeInts(node, "pads", layer.pooling().pad()))
                return false;
        }
        if (!ConvertAtrributeInts(node, "strides", layer.pooling().stride()))
            return false;
        if (GetAtrribute(node, "ceil_mode") == NULL)
            layer.pooling().roundingType() = RoundingTypeFloor;
        else
        {
            int ceilMode;
            if (!ConvertAtrributeInt(node, "ceil_mode", ceilMode))
                return false;
            layer.pooling().roundingType() = ceilMode ? RoundingTypeCeil : RoundingTypeFloor;
        }
        if (GetAtrribute(node, "count_include_pad"))
        {
            int64_t countIncludePad;
            if (!ConvertAtrributeInt(node, "count_include_pad", countIncludePad))
                return false;
            layer.pooling().excludePad() = (countIncludePad == 0);
        }
        return true;
    }
}

#endif