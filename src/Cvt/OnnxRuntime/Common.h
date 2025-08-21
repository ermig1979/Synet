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

#pragma once

#include "Synet/Common.h"

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

namespace Synet
{
    typedef std::map<String, String> Renames;
    typedef std::set<String> Consts;

    //--------------------------------------------------------------------------------------------------

    inline String ValidName(const String& src, Renames& renames)
    {
        String dst = src;
        for (size_t i = 0; i < dst.size(); ++i)
        {
            if (dst[i] == ':' || dst[i] == ' ')
                dst[i] = '_';
        }
        if (dst != src)
            renames[src] = dst;
        return dst;
    }

    //--------------------------------------------------------------------------------------------------

    inline Shape Convert(const onnx::TensorShapeProto& shapeProto)
    {
        Shape shape;
        for (size_t i = 0; i < shapeProto.dim_size(); ++i)
        {
            if (shapeProto.dim(i).has_dim_value())
                shape.push_back((size_t)shapeProto.dim(i).dim_value());
            else
                shape.push_back(size_t(-1));
        }
        return shape;
    }

    //-----------------------------------------------------------------------------------------

    inline bool MoveDequantizeLinearToLayer(LayerParams& layers, LayerParam& layer)
    {
        for (int s = 0; s < (int)layer.src().size(); ++s)
        {
            LayerParam* dequantize = GetLayer(layers, layer.src()[s]);
            if (dequantize->type() != LayerTypeDequantizeLinear)
                SYNET_ERROR("MoveDequantizeLinearToLayer can move only DequantizeLinearLayer layers!");
            layer.qSrc().push_back(dequantize->quantize());
            layer.qSrc().back().weights() = dequantize->weight().size();
            for (size_t w = 0; w < dequantize->weight().size(); ++w)
                layer.weight().push_back(dequantize->weight()[w]);
            if (!dequantize->src().empty())
            {
                layer.src()[s] = dequantize->src()[0];
                dequantize->src().clear();
            }
            else
            {
                layer.src().erase(layer.src().begin() + s, layer.src().begin() + s + 1);
                --s;
            }
        }
        return true;
    }


}

#endif