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
    bool ConvertInput(const onnx::ValueInfoProto& input, bool trans, Synet::NetworkParam& network, Renames& renames)
    {
        LayerParam layer;
        layer.type() = LayerTypeInput;
        layer.name() = ValidName(input.name(), renames);
        layer.dst().push_back(input.name());
        layer.input().shape().resize(1);
        Shape shape = Convert(input.type().tensor_type().shape());
        if (trans)
        {
            if (shape.size() == 4)
            {
                shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                layer.input().shape()[0].format() = TensorFormatNhwc;
            }
        }
        if (shape.size() > 1 && shape[0] == -1)
            shape[0] = 1;
        layer.input().shape()[0].dim() = shape;
        switch (input.type().tensor_type().elem_type())
        {
        case onnx::TensorProto_DataType_FLOAT: layer.input().shape()[0].type() = Synet::TensorType32f; break;
        case onnx::TensorProto_DataType_INT32: layer.input().shape()[0].type() = Synet::TensorType32i; break;
        default:
            SYNET_ERROR(" Unknown input tensor type " << input.type().tensor_type().elem_type() << " !");
        }
        network.layers().push_back(layer);
        return true;
    }
}

#endif