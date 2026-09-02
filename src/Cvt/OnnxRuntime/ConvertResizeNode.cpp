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
    bool ConvertResizeNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
    {
        if (!CheckSourceNumber(layer, 1, 4))
            return false;
        if (layer.src().size() == 4)
        {
            const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                SYNET_ERROR("Resize src[1] must be empty Const type!");
            const LayerParam * src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL)
                return false;
            if (src2->type() != Synet::LayerTypeConst || src2->weight()[0].dim()[0] != 0)
                SYNET_ERROR("Resize src[2] must be empty Const type!");
            layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 3);
            const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
            if (src1b == NULL)
                return false;
            if (src1b->type() == Synet::LayerTypeConst)
            {
                layer.weight() = src1b->weight();
                layer.src().resize(1);
            }
        }           
        else if (layer.src().size() == 3)
        {
            const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                SYNET_ERROR("Resize src[1] must be empty Const type!");
            layer.src().erase(layer.src().begin() + 1);
            const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
            if (src1b == NULL)
                return false;
            if (src1b->type() == Synet::LayerTypeConst)
            {
                layer.weight() = src1b->weight();
                layer.src().resize(1);
            }
        }
        if (layer.src().size() == 2)
        {
            const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() == Synet::LayerTypeMeta && src1->meta().type() == Synet::MetaTypeConst)
            {
                const TensorParam & alpha = src1->meta().alpha();
                if (alpha.shape().size() == 1 && alpha.shape()[0] == 4)
                {
                    layer.interp().height() = (int32_t)alpha.i64()[2];
                    layer.interp().width() = (int32_t)alpha.i64()[3];
                    layer.src().resize(1);
                }
                else
                    SYNET_ERROR("Resize src[1] alpha must have 1D shape with 4 values!");
            }
            else if (src1->type() == Synet::LayerTypeConst)
            {
                layer.weight() = src1->weight();
                layer.src().resize(1);
            }
        }

        String mode;
        if (!ConvertAtrributeString(node, "mode", mode))
            return false;
        if (mode == "nearest")
            layer.interp().interpolationType() = InterpolationTypeNearest;
        else if (mode == "linear")
            layer.interp().interpolationType() = InterpolationTypeBilinear;
        else
            SYNET_ERROR("Unsupported interpolation mode '" << mode << "' !");

        if (GetAtrribute(node, "coordinate_transformation_mode"))
        {
            String coordTransf;
            if (!ConvertAtrributeString(node, "coordinate_transformation_mode", coordTransf))
                return false;
            if (coordTransf == "pytorch_half_pixel")
                layer.interp().coordinateTransformType() = CoordinateTransformTypeHalfPixel;
            else if (coordTransf == "asymmetric")
                layer.interp().coordinateTransformType() = CoordinateTransformTypePytorch;
            else if (coordTransf == "half_pixel")
                layer.interp().coordinateTransformType() = CoordinateTransformTypeHalfPixel;
            else if (coordTransf == "align_corners")
                layer.interp().coordinateTransformType() = CoordinateTransformTypeCaffe;
            else
                SYNET_ERROR("Unsupported coordinate_transformation_mode '" << coordTransf << "' !");
        }

        layer.type() = Synet::LayerTypeInterp;
        return true;
    }
}

#endif
