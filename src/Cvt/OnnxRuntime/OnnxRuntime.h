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

//#define SYNET_ONNX_PARSE_STOP_ON_ERROR

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/SynetUtils.h"
#include "Cvt/Optimizer/Optimizer.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(String srcGraphPath, bool trans, const String& dstModelPath, const String& dstWeightPath, const OnnxParam& onnxParam, const OptimizerParam& optParam);

    private:
        bool LoadModel(const String& path, onnx::ModelProto& model);

        bool ConvertModel(const onnx::ModelProto& model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Bytes& reordered);

        void SetSrcAndDst(const onnx::NodeProto& node, Renames& renames, LayerParam& layer);

        bool ManualInsertToNchwPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        bool ManualInsertToNhwcPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        //-----------------------------------------------------------------------------------------

        bool ConvertAbsNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeAbs;
            return true;
        }

        bool ConvertAndNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            layer.type() = Synet::LayerTypeBinaryOperation;
            layer.binaryOperation().type() = BinaryOperationTypeAnd;
            return true;
        }

        bool ConvertArgMaxNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            layer.type() = Synet::LayerTypeArgMax;
            if (!ConvertAtrributeInt(node, "axis", layer.argMax().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.argMax().keepDims()))
                return false;
            return true;
        }

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

        bool ConvertCastNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            int to;
            if (!ConvertAtrributeInt(node, "to", to))
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeCast;
                if (to == onnx::TensorProto_DataType_FLOAT)
                    layer.meta().alpha().type() = TensorType32f;
                else if (to == onnx::TensorProto_DataType_INT32)
                    layer.meta().alpha().type() = TensorType32i;
                else if (to == onnx::TensorProto_DataType_INT64)
                    layer.meta().alpha().type() = TensorType64i;
                else
                    SYNET_ERROR("Unsupported cast type!");
            }
            else
            {
                layer.type() = Synet::LayerTypeCast;
                if (to == onnx::TensorProto_DataType_FLOAT)
                    layer.cast().type() = TensorType32f;
                else if (to == onnx::TensorProto_DataType_INT32)
                    layer.cast().type() = TensorType32i;
                else if (to == onnx::TensorProto_DataType_INT64)
                    layer.cast().type() = TensorType64i;
                else if (to == onnx::TensorProto_DataType_UINT8)
                    layer.cast().type() = TensorType8u;
                else
                    SYNET_ERROR("Unsupported cast type!");
                if (src0->type() == LayerTypeConst && src0->weight().size() && src0->weight()[0].type() == layer.cast().type())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.cast().type() = TensorTypeUnknown;
                }
                if (src0->type() == LayerTypeQuantizeLinear && src0->quantize().type() == layer.cast().type())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.cast().type() = TensorTypeUnknown;
                }
            }
            return true;
        }

        bool ConvertCeilNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeCeil;
            }
            else
            {
                SYNET_ERROR("Unsupported src type!");
            }
            return true;
        }

        bool ConvertClipNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRestrictRange;
            if (layer.src().size() > 1)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeConst || src1->weight().size() != 1)
                    return false;
                const float* min = GetWeight<float>(original, src1->weight()[0]);
                layer.restrictRange().lower() = min[0];
                if (layer.src().size() > 2)
                {
                    const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                    if (src2 == NULL || src2->type() != LayerTypeConst || src2->weight().size() != 1)
                        return false;
                    const float* max = GetWeight<float>(original, src2->weight()[0]);
                    layer.restrictRange().upper() = max[0];
                }
                layer.src().resize(1);
            }
            else
            {
                if (!ConvertAtrributeFloat(node, "min", layer.restrictRange().lower(), true, -FLT_MAX))
                    return false;
                if (!ConvertAtrributeFloat(node, "max", layer.restrictRange().upper(), true, +FLT_MAX))
                    return false;
            }
            return true;
        }

        bool ConvertConcatNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            const LayerParam* src1 = layer.src().size() < 2 ? 0 : GetLayer(layers, layer.src()[1]);
            if (src0->type() == Synet::LayerTypeMeta || (src1 && src1->type() == Synet::LayerTypeMeta))
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypePack;
            }
            else
            {
                layer.type() = Synet::LayerTypeConcat;
                if (!ConvertAtrributeInt(node, "axis", layer.concat().axis()))
                    return false;
                if (trans && CurrentTensorFormat(layers, layer.src(), true, true, true) == TensorFormatNhwc)
                {
                    Shape nchw = Shape({ 0, 3, 1, 2 });
                    if(layer.concat().axis() >= 0 && layer.concat().axis() < 4)
                        layer.concat().axis() = (uint32_t)nchw[layer.concat().axis()];
                }
            }
            return true;
        }

        bool ConvertConstantOfShapeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& original, Bytes& reordered)
        {
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)// || src0->type() != Synet::LayerTypeMeta)
                return false;
            Shape shape;
            if (IsMetaConst64i(*src0))
                shape = Shp(src0->meta().alpha().i64());
            const onnx::AttributeProto * attribute = GetAtrribute(node, "value");
            if (attribute && attribute->type() == onnx::AttributeProto_AttributeType_TENSOR)
            {
                const onnx::TensorProto& tensor = attribute->t();
                if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                {
                    int64_t value;
                    if (tensor.int64_data_size())
                        value = tensor.int64_data(0);
                    else if (tensor.has_raw_data())
                        value = ((int64_t*)tensor.raw_data().c_str())[0];
                    else
                        return false;
                    if (src0->type() != Synet::LayerTypeMeta)
                        return false;
                    if (src0->meta().type() == Synet::MetaTypeConst)
                    {
                        if (src0->meta().alpha().type() != Synet::TensorType64i || src0->meta().alpha().shape().size() != 1 || src0->meta().alpha().shape()[0] != 1)
                            return false;
                        layer.type() = Synet::LayerTypeMeta;
                        layer.meta().type() = Synet::MetaTypeConst;
                        layer.meta().alpha().type() = Synet::TensorType64i;
                        layer.meta().alpha().shape().push_back(src0->meta().alpha().i64()[0]);
                        layer.meta().alpha().i64().resize(src0->meta().alpha().i64()[0], value);
                        layer.src().resize(0); 
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeMeta;
                        layer.meta().type() = Synet::MetaTypeConstantOfShape;
                        layer.meta().alpha().type() = Synet::TensorType64i;
                        layer.meta().alpha().shape() = Shp(1);
                        layer.meta().alpha().i64().resize(1, value);
                    }
                }
                else if (tensor.data_type() == onnx::TensorProto_DataType_FLOAT)
                {
                    float value;
                    if (tensor.float_data_size())
                        value = tensor.float_data(0);
                    else if (tensor.has_raw_data())
                        value = ((float*)tensor.raw_data().c_str())[0];
                    else
                        return false;
                    layer.type() = Synet::LayerTypeConstantOfShape;
                    layer.constantOfShape().value().type() = TensorType32f;
                    layer.constantOfShape().value().shape() = Shp(1);
                    layer.constantOfShape().value().f32().resize(1, value);
                }
                else if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                {
                    int32_t value;
                    if (tensor.int32_data_size())
                        value = tensor.int32_data(0);
                    else if (tensor.has_raw_data())
                        value = ((int32_t*)tensor.raw_data().c_str())[0];
                    else
                        return false;                    
                    if (shape.empty())
                    {
                        layer.type() = Synet::LayerTypeConstantOfShape;
                        layer.constantOfShape().value().type() = TensorType32i;
                        layer.constantOfShape().value().shape() = Shp(1);
                        layer.constantOfShape().value().i32().resize(1, value);
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeConst;
                        layer.weight().resize(1);
                        layer.weight()[0].type() = Synet::TensorType32i;
                        layer.weight()[0].dim() = shape;
                        layer.weight()[0].scalar() = true;
                        layer.weight()[0].offset() = original.size();
                        layer.weight()[0].size() = sizeof(int32_t);
                        PushBack<int32_t>(original, value);
                        PushBack<int32_t>(reordered, value);
                        layer.src().clear();
                    }
                }
                else
                    return false;
            }
            else
            {
                CPL_LOG_SS(Error, "Unsupported type of attribute 'value'");
                return false;
            }
            return true;
        }

        bool ConvertCosNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeCos;
            return true;
        }

        bool ConvertDivNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = float(1.0 / double(pScale[0]));
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
            {
                const float* pSrc0 = GetWeight<float>(original, src0->weight()[0]);
                if (pSrc0[0] != 1.0f)
                    return false;
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = UnaryOperationTypeRcp;
                layer.src().erase(layer.src().begin());
                return true;
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
            {
                const float* pSrc = GetWeight<float>(original, src1->weight()[0]);
                size_t size = TensorSize(src1->weight()[0].dim());
                bool uniform = true;
                for (size_t i = 1; i < size && uniform; ++i)
                    uniform = (pSrc[i] == pSrc[0]);
                if (uniform)
                {
                    layer.type() = Synet::LayerTypePower;
                    layer.power().scale() = 1.0f / pSrc[0];
                }
                else
                {
                    layer.weight() = src1->weight();
                    const Shape& dim = src1->weight()[0].dim();
                    if ((dim.size() == 4 && dim[1] != 1) || (dim.size() == 3 && dim[0] != 1) || dim.size() == 1)
                    {
                        layer.type() = Synet::LayerTypeScale;
                        layer.scale().biasTerm() = false;
                        if (dim.size() == 1)
                            layer.scale().axis() = -1;
                        if (!CompactShape(layer.weight()[0].dim()))
                            return false;                    
                    }
                    else
                    {
                        layer.type() = Synet::LayerTypeMul;
                    }
                    float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                    for (size_t i = 0; i < size; ++i)
                        pDst[i] = 1.0f / pSrc[i];
                }
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeDiv;
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeDiv;
            }
            return true;
        }

        bool ConvertErfNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeErf;
            return true;
        }

        bool ConvertEqualNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeEqual;
            }
            else
            {
                layer.type() = Synet::LayerTypeCompare;
                layer.compare().compareType() = CompareTypeEqual;
            }
            return true;
        }

        bool ConvertExpNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeExp;
            return true;
        }

        bool ConvertExpandNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeMeta)
            {
                const MetaParam & meta = src1->meta();
                if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i && AllEqualTo(meta.alpha().i64(), int64_t(1)))
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.src().resize(1);
                }
                else if (meta.type() == MetaTypeConst && meta.alpha().type() == TensorType64i &&
                    src0->type() == LayerTypeConst && Shp(meta.alpha().i64()) == src0->weight()[0].dim())
                {
                    layer.type() = Synet::LayerTypeStub;
                    layer.src().resize(1);
                }
                else
                {
                    layer.type() = Synet::LayerTypeTile;
                }
            }
            else
                return false;
            return true;
        }

        bool ConvertFloorNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeFloor;
            }
            else
            {
                layer.type() = Synet::LayerTypeUnaryOperation;
                layer.unaryOperation().type() = Synet::UnaryOperationTypeFloor;
            }
            return true;
        }

        bool ConvertGatherNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeGather;
            }
            else
            {
                layer.type() = LayerTypeGather;
                if (node.op_type() == "Gather")
                {
                    if (!ConvertAtrributeInt(node, "axis", layer.gather().axis()))
                        return false;
                }
                if (node.op_type() == "GatherElements")
                {
                    if (!ConvertAtrributeInt(node, "axis", layer.gather().axis()))
                        return false;
                    layer.gather().version() = 1;
                }
                if (node.op_type() == "GatherND")
                {
                    if (!ConvertAtrributeInt(node, "batch_dims", layer.gather().axis(), true, 0))
                        return false;
                }
            }
            return true;
        }

        bool ConvertGreaterNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeCompare;
            layer.compare().compareType() = CompareTypeGreaterThan;
            return true;
        }

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
                return false;

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
                return false;

            return true;
        }

        bool ConvertHardSigmoidNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeHardSigmoid;
            if (!ConvertAtrributeFloat(node, "alpha", layer.hardSigmoid().scale()))
                return false;
            return true;
        }

        bool ConvertIdentityNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeStub;
            }
            else
            {
                layer.type() = LayerTypeStub;
            }
            return true;
        }

        bool ConvertInstanceNormalizationNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            layer.type() = Synet::LayerTypeNormalize;
            layer.normalize().version() = 3;
            if (!ConvertAtrributeFloat(node, "epsilon", layer.normalize().eps()))
                return false;
            layer.weight().resize(2);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            layer.weight()[0] = src1->weight()[0];
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = src2->weight()[0];
            layer.src().resize(1);
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                layer.normalize().axis() = -1;
            }
            else
                layer.normalize().axis() = 1;
            return true;
        }

        bool ConvertLayerNormalizationNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            layer.type() = Synet::LayerTypeNormalize;
            layer.normalize().version() = 3;
            if (!ConvertAtrributeFloat(node, "epsilon", layer.normalize().eps()))
                return false;
            layer.weight().resize(2);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            layer.weight()[0] = src1->weight()[0];
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            layer.weight()[1] = src2->weight()[0];
            layer.src().resize(1);
            if (GetAtrribute(node, "axis"))
            {
                if (!ConvertAtrributeInt(node, "axis", layer.normalize().axis()))
                    return false;
                if(layer.normalize().axis() == -1)
                    layer.normalize().version() = 2;
            }
            else
            {
                if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
                {
                    layer.normalize().axis() = -1;
                }
                else
                    layer.normalize().axis() = 1;
            }
            return true;
        }

        bool ConvertLeakyReluNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            if (!ConvertAtrributeFloat(node, "alpha", layer.relu().negativeSlope()))
                return false;
            return true;
        }

        bool ConvertLessNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeCompare;
            layer.compare().compareType() = CompareTypeLessThan;
            return true;
        }

        bool ConvertLogNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeLog;
            return true;
        }

        bool ConvertLogSoftmaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            if (!ConvertAtrributeInt(node, "axis", layer.softmax().axis()))
                return false;
            if (trans && !PermutedToNchw(layers, layer.src(), false, false, false))
            {
                CPL_LOG_SS(Error, "This layer can work only in NCHW format!");
                return false;
            }
            layer.softmax().log() = true;
            return true;
        }

        bool ConvertLstmNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeLstm;
            if (!ConvertAtrributeInt(node, "hidden_size", layer.lstm().hiddenSize()))
                return false;
            String direction;
            if (!ConvertAtrributeString(node, "direction", direction))
                return false;
            if (direction == "forward")
                layer.lstm().direction() = LstmDirectionTypeForward;
            else if (direction == "reverse")
                layer.lstm().direction() = LstmDirectionTypeReverse;
            else if (direction == "bidirectional")
                layer.lstm().direction() = LstmDirectionTypeBidirectional;
            else
                return false;
            if (!CheckSourceNumber(layer, 6))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL || src1->type() != LayerTypeConst)
                return false;
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src2 == NULL || src2->type() != LayerTypeConst)
                return false;
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst)
                return false;
            layer.weight().resize(3);
            layer.weight()[0] = src1->weight()[0];
            layer.weight()[1] = src2->weight()[0];           
            layer.weight()[2] = src3->weight()[0];
            layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 4);
            layer.dst().resize(1);
            return true;
        }

        bool ConvertMatMulNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, TensorFormatMap *tensorFormatMap)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            layer.type() = Synet::LayerTypeInnerProduct;
            int transB = false;
            layer.weight().resize(layer.src().size() - 1);
            layer.innerProduct().biasTerm() = false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst)
            {
                layer.weight()[0] = src1->weight()[0];
            }
            else if (src1->type() == LayerTypePermute)
            {
                if (!CheckSourceNumber(*src1, 1))
                    return false;
                const LayerParam* src10 = GetLayer(layers, src1->src()[0]);
                if (src10 == NULL) 
                    return false;
                if (src10->type() == LayerTypeConst)
                {
                    transB = true;
                    layer.weight() = src10->weight();
                    layers.erase(layers.begin() + (src1 - layers.data()));
                }
            }
            Shape weight = layer.weight()[0].dim();
            layer.innerProduct().transposeB() = !transB;
            if (weight.empty())
            {
                layer.weight().clear();
                layer.innerProduct().outputNum() = 0;
                layer.innerProduct().axis() = -1;
            }
            else
            {
                //if (!CheckSignificantDims(weight, 2, "MatMul weight"))
                //    return false;
                if(weight.size() > 2)
                    layer.innerProduct().axis() = weight.size() - 1;
                layer.innerProduct().outputNum() = (uint32_t)(transB ? weight[weight.size() - 2] : weight[weight.size() - 1]);
                layer.src().resize(1);
                if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, tensorFormatMap) == TensorFormatNhwc)
                    SYNET_ERROR("Can 't convert MatMul node for NHWC format!");
            }
            return true;
        }

        bool ConvertMaxPoolNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypePooling;
            layer.pooling().method() = PoolingMethodTypeMax;
            if (!ConvertAtrributeInts(node, "kernel_shape", layer.pooling().kernel()))
                return false;
            if (!ConvertAtrributeInts(node, "pads", layer.pooling().pad()))
                return false;
            if (!ConvertAtrributeInts(node, "strides", layer.pooling().stride()))
                return false;

            if(GetAtrribute(node, "ceil_mode") == NULL)
                layer.pooling().roundingType() = RoundingTypeFloor;
            else
            {
                int ceilMode;
                if (!ConvertAtrributeInt(node, "ceil_mode", ceilMode))
                    return false;
                layer.pooling().roundingType() = ceilMode ? RoundingTypeCeil : RoundingTypeFloor;
            }
            return true;
        }

        bool ConvertModNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeMod;
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeMod;
            }
            return true;
        }

        bool ConvertMulNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeConst)
            {
                std::swap(src0, src1);
                std::swap(layer.src()[0], layer.src()[1]);
            }
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pScale = GetWeight<float>(original, src1->weight()[0]);
                layer.power().scale() = pScale[0];
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 3 && src1->weight()[0].dim()[0] != 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.weight() = src1->weight();
                //if (!CompactShape(layer.weight()[0].dim()))
                //    return false;
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1 && src1->weight()[0].dim().size() == 4 && src1->weight()[0].dim()[1] != 1)
            {
                layer.type() = Synet::LayerTypeScale;
                layer.weight() = src1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeMul;
            }
            else
            {
                if (onnxParam.mulToEltwise())
                {
                    layer.type() = Synet::LayerTypeEltwise;
                    layer.eltwise().operation() = EltwiseOperationTypeProduct;
                }
                else
                    layer.type() = Synet::LayerTypeMul;
                if (src0->type() == LayerTypeConst && src1->type() != LayerTypeConst)
                    std::swap(layer.src()[0], layer.src()[1]);
            }
            return true;
        }

        bool ConvertNegNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = Synet::UnaryOperationTypeNeg;
            return true;
        }

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
            layer.nonMaxSuppression().iouThreshold() = bin[src3->weight()[0].offset() / sizeof(float)];

            if (layer.src().size() > 4)
            {
                const LayerParam* src4 = GetLayer(layers, layer.src()[4]);
                if (src4 == NULL || src4->type() != LayerTypeConst)
                    return false;
                layer.nonMaxSuppression().scoreThreshold() = bin[src4->weight()[0].offset() / sizeof(float)];
            }

            layer.type() = Synet::LayerTypeNonMaxSuppression;
            layer.src().resize(2);

            return true;
        }

        bool ConvertNonZeroNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            layer.type() = Synet::LayerTypeNonZero;
            return true;
        }

        bool ConvertNotNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = Synet::UnaryOperationTypeNot;
            return true;
        }

        bool ConvertPadNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 3))
                return false;

            layer.type() = Synet::LayerTypePad;
            String mode;
            if (!ConvertAtrributeString(node, "mode", mode, true, "constant"))
                return false;
            if (mode == "constant")
                layer.pad().mode() = PadModeConstant;
            else if (mode == "reflect")
                layer.pad().mode() = PadModeReflect;
            else if (mode == "edge")
                layer.pad().mode() = PadModeEdge;
            else if (mode == "wrap")
                layer.pad().mode() = PadModeWrap;
            else
                SYNET_ERROR("Unknown type of pad mode: " << mode << " !");
            
            if (layer.src().size() > 1)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeMeta)
                    return false;
                if (layer.src().size() > 2)
                {
                    const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                    if (src2 == NULL || src2->type() != LayerTypeConst || src2->weight()[0].type() != TensorType32f)
                        return false;
                    if(GetWeight<float>(original, src2->weight()[0])[0] != 0)
                        SYNET_ERROR("Synet support only pad value == 0!");
                    layer.src().resize(2);
                }
            }  
            else
            {
                if (!ConvertAtrributeInts(node, "pads", layer.pad().pads()))
                    return false;
            }

            return true;
        }

        bool ConvertPowNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pPower = GetWeight<float>(original, src1->weight()[0]);
                layer.power().power() = pPower[0];
                layer.src().resize(1);
            }
            else
            {
                std::cout << "PowerNode error: src1 { type: " << Cpl::ToStr(src1->type()) << " size: " << TensorSize(src1->weight()[0].dim()) << " }" << std::endl;
                return false;
            }
            return true;
        }

        bool ConvertRangeNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() != LayerTypeMeta && src0->type() != LayerTypeConst)
                return false;
            if (src1->type() != LayerTypeMeta && src1->type() != LayerTypeConst)
                return false;
            if (src2->type() != LayerTypeMeta && src2->type() != LayerTypeConst)
                return false;
            layer.type() = Synet::LayerTypeMeta;
            layer.meta().type() = Synet::MetaTypeRange;
            return true;
        }

        bool ConvertReduceL2Node(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeL2;
            if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && !PermutedToNchw(layers, false, true, true))
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for (size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
            return true;
        }

        bool ConvertReduceMaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeMax;
            if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                return false;
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && !PermutedToNchw(layers, false, true, true))
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for(size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
            return true;
        }

        bool ConvertReduceSumNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            layer.type() = Synet::LayerTypeReduction;
            layer.reduction().type() = ReductionTypeSum;
            if (layer.src().size() == 2)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1 == NULL || src1->type() != LayerTypeMeta || src1->meta().type() != Synet::MetaTypeConst)
                    return false;
                const TensorParam& alpha = src1->meta().alpha();
                if (alpha.type() != TensorType64i)
                    return false;
                layer.reduction().axis().resize(alpha.i64().size());
                for (size_t i = 0; i < alpha.i64().size(); ++i)
                    layer.reduction().axis()[i] = (int)alpha.i64()[i];
                layer.src().resize(1);
            }
            else
            {
                if (!ConvertAtrributeInts(node, "axes", layer.reduction().axis()))
                    return false;
            }
            if (!ConvertAtrributeInt(node, "keepdims", layer.reduction().keepDims()))
                return false;
            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true) == TensorFormatNhwc)
            {
                Ints nchw = Ints({ 0, 3, 1, 2 }), axis = layer.reduction().axis();
                for (size_t i = 0; i < axis.size(); ++i)
                    layer.reduction().axis()[i] = nchw[axis[i]];
            }
            return true;
        }

        bool ConvertReluNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeRelu;
            return true;
        }

        bool ConvertReshapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer, TensorFormatMap* tensorFormatMap)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL || src1->type() != LayerTypeMeta)
                return false;
            if (src1->meta().type() == MetaTypeStub)
                src1 = GetLayer(layers, src1->src()[0]);
            if (src0->type() == Synet::LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = Synet::MetaTypeReshape;
            }            
            else if (src1->meta().type() == MetaTypeConst)
            {
                const TensorParam & alpha = src1->meta().alpha();
                if (alpha.shape().size() != 1)
                    return false;
                Shape& shape = layer.reshape().shape();
                layer.type() = LayerTypeReshape;
                shape = Shp(alpha.i64().data(), alpha.shape()[0]);
                layer.src().resize(1);
                if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, tensorFormatMap) == TensorFormatNhwc)
                {
                    if (shape.size() == 5)
                    {
                        shape = Shp( shape[0], shape[3], shape[4], shape[1], shape[2]);
                    }
                    if (shape.size() == 4)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[3], shape[1] });
                    }
                    if (shape.size() == 3)
                    {
                        shape = Shape({ shape[0], shape[2] , shape[1] });
                    }
                }
            }
            else
            {
                layer.type() = LayerTypeReshape;
            }
            if (onnxParam.setReshapeAxis1() && layer.type() == LayerTypeReshape)
            {
                layer.reshape().axis() = 1;
                //if (layer.reshape().shape().size() > 1 && layer.reshape().shape()[0] == 1)
                //    layer.reshape().shape().erase(layer.reshape().shape().begin(), layer.reshape().shape().begin() + 1);
            }
            return true;
        }

        bool ConvertResizeNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            if (layer.src().size() == 4)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                    return false;
                const LayerParam * src2 = GetLayer(layers, layer.src()[2]);
                if (src2->type() != Synet::LayerTypeConst || src2->weight()[0].dim()[0] != 0)
                    return false;
                layer.src().erase(layer.src().begin() + 1, layer.src().begin() + 3);
                const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
                if (src1b->type() == Synet::LayerTypeConst)
                {
                    layer.weight() = src1b->weight();
                    layer.src().resize(1);
                }
            }           
            else if (layer.src().size() == 3)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() != Synet::LayerTypeConst || src1->weight()[0].dim()[0] != 0)
                    return false;
                layer.src().erase(layer.src().begin() + 1);
                const LayerParam* src1b = GetLayer(layers, layer.src()[1]);
                if (src1b->type() == Synet::LayerTypeConst)
                {
                    layer.weight() = src1b->weight();
                    layer.src().resize(1);
                }
            }
            if (layer.src().size() == 2)
            {
                const LayerParam * src1 = GetLayer(layers, layer.src()[1]);
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
                        return false;
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
                return false;

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
                    return false;
            }

            layer.type() = Synet::LayerTypeInterp;
            return true;
        }

        bool ConvertScaledDotProductAttentionNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 4))
                return false;
            const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
            if (src3 == NULL || src3->type() != LayerTypeConst || src3->weight()[0].type() != TensorType32f || src3->weight()[0].dim() != Shp(1))
                return false;
            layer.type() = Synet::LayerTypeScaledDotProductAttention;
            layer.src().resize(3);
            return true;
        }

        bool ConvertScatterElementsNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            layer.type() = Synet::LayerTypeScatterNd;
            layer.scatter().version() = 1;
            if (!ConvertAtrributeInt(node, "axis", layer.scatter().axis()))
                return false;
            String reduction;
            if (!ConvertAtrributeString(node, "reduction", reduction) || reduction != "none")
                return false;
            return true;
        }

        bool ConvertScatterNdNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes & original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            layer.type() = Synet::LayerTypeScatterNd;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst)
            {
                const TensorParam & alpha = src1->meta().alpha();
                size_t size = TensorSize(alpha.shape()), offset = reordered.size();
                layer.type() = Synet::LayerTypeScatterNd;
                layer.weight().resize(1);
                layer.weight()[0].dim() = alpha.shape();
                layer.weight()[0].type() = TensorType32i;
                layer.weight()[0].offset() = offset;
                layer.weight()[0].size() = size * 4;
                layer.src().erase(layer.src().begin() + 1);
                original.resize(offset + size * 4);
                reordered.resize(offset + size * 4);
                if (alpha.type() == TensorType64i)
                {
                    const int64_t* src = alpha.i64().data();
                    int32_t * dst = GetWeight<int32_t>(reordered, layer.weight()[0]);
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = (int32_t)src[i];
                }
                else
                {
                    std::cout << "src[1] type must be meta const int64!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        bool ConvertShapeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const OnnxParam& onnxParam, LayerParam& layer)
        {
            layer.type() = LayerTypeMeta;
            layer.meta().type() = MetaTypeShape;
            layer.meta().version() = 1;
            if (trans)
            {
                for (size_t i = 0; i < onnxParam.shapeV2s().size(); ++i)
                    if (layer.name() == onnxParam.shapeV2s()[i])
                        layer.meta().version() = 2;
            }
            return true;
        }

        bool ConvertSigmoidNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSigmoid;
            return true;
        }

        bool ConvertSinNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeSin;
            return true;
        }

        bool ConvertSliceNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer, TensorFormatMap *tensorFormatMap)
        {
            if (layer.src().size() == 1)
            {
                if (!ConvertAtrributeInts(node, "axes", layer.stridedSlice().axes()))
                    return false;
                if (!ConvertAtrributeInts(node, "starts", layer.stridedSlice().beginDims()))
                    return false;
                if (!ConvertAtrributeInts(node, "ends", layer.stridedSlice().endDims()))
                    return false;
                layer.type() = Synet::LayerTypeStridedSlice;
            }
            else if (layer.src().size() == 3)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                if (src0 == NULL || src1 == NULL || src2 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = MetaTypeSlice;
                }
                else
                    return false;
            }
            else if(layer.src().size() >= 4 && layer.src().size() <= 5)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
                const LayerParam* src3 = GetLayer(layers, layer.src()[3]);
                if (src0 == NULL || src1 == NULL || src2 == NULL || src3 == NULL)
                    return false;
                const LayerParam* src4 = NULL;
                if (layer.src().size() > 4)
                {
                    src4 = GetLayer(layers, layer.src()[4]);
                    if (src4 == NULL)
                        return false;
                }
                if (src0->type() == LayerTypeMeta)
                {
                    if (!CheckSourceNumber(layer, 4, 5))
                        return false;
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = MetaTypeSlice;
                }
                else
                {
                    layer.type() = Synet::LayerTypeStridedSlice;
                    if (layer.src().size() == 4)
                    {
                        if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta)
                            return false;
                        if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                            src3->meta().type() == Synet::MetaTypeConst)
                        {
                            if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 ||
                                src3->meta().alpha().i64().size() != 1)
                                return false;
                            layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                            layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                            layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                            //layer.stridedSlice().strideDims().push_back((size_t)src4->meta().alpha().i64()[0]);
                            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
                            {
                                Shape nchw = Shape({ 0, 3, 1, 2 });
                                layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                            }
                            layer.src().resize(1);
                        }
                    }
                    else if (layer.src().size() == 5)
                    {
                        if (src1->type() != LayerTypeMeta || src2->type() != LayerTypeMeta || src3->type() != LayerTypeMeta || src4->type() != LayerTypeMeta)
                            return false;
                        if (src1->meta().type() == Synet::MetaTypeConst && src2->meta().type() == Synet::MetaTypeConst &&
                            src3->meta().type() == Synet::MetaTypeConst && src4->meta().type() == Synet::MetaTypeConst)
                        {
                            if (src1->meta().alpha().i64().size() != 1 || src2->meta().alpha().i64().size() != 1 || 
                                src3->meta().alpha().i64().size() != 1 || src4->meta().alpha().i64().size() != 1)
                                return false;
                            layer.stridedSlice().axes().push_back((size_t)src3->meta().alpha().i64()[0]);
                            layer.stridedSlice().beginDims().push_back(src1->meta().alpha().i64()[0]);
                            layer.stridedSlice().endDims().push_back(src2->meta().alpha().i64()[0]);
                            layer.stridedSlice().strideDims().push_back(src4->meta().alpha().i64()[0]);
                            if (trans && CurrentTensorFormat(layers, layer.src(), false, true, true, tensorFormatMap) == TensorFormatNhwc)
                            {
                                Shape nchw = Shape({ 0, 3, 1, 2 });
                                layer.stridedSlice().axes()[0] = nchw[layer.stridedSlice().axes()[0]];
                            }
                            layer.src().resize(1);
                        }
                    }
                }
            }
            return true;
        }

        bool ConvertSoftmaxNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeSoftmax;
            if (!ConvertAtrributeInt(node, "axis", layer.softmax().axis()))
                return false;
            if (trans && !PermutedToNchw(layers, layer.src(), true, false, true))
            {
                CPL_LOG_SS(Error, "This layer can work only in NCHW format!");
                return false;
            }
            return true;
        }

        bool ConvertSplitNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
                if (!ConvertAtrributeInts(node, "split", layer.unpack().parts(), true))
                    return false;
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src1->type() == LayerTypeMeta &&  src1->meta().type() == Synet::MetaTypeConst)
                {
                    const TensorParam & alpha = src1->meta().alpha();
                    assert(alpha.shape().size() == 1);
                    for (size_t i = 0; i < alpha.shape()[0]; ++i)
                        layer.unpack().parts().push_back((int32_t)alpha.i64()[i]);
                    layer.src().resize(1);
                }
                else
                    assert(0);
            }
            if (!ConvertAtrributeInt(node, "axis", layer.unpack().axis()))
                return false;
            layer.type() = Synet::LayerTypeUnpack;
            if (trans && CurrentTensorFormat(layers, layer.src(), true, false, true, NULL) == TensorFormatNhwc)
            {
                Shape nchw = Shape({ 0, 3, 1, 2 });
                layer.unpack().axis() = nchw[layer.unpack().axis()];
            }
            return true;
        }

        bool ConvertSqrtNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeSqrt;
            return true;
        }

        bool ConvertSqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                if (src0 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = LayerTypeMeta;
                    layer.meta().type() = MetaTypeSqueeze;
                }
                else
                {
                    layer.type() = Synet::LayerTypeSqueeze;
                    if (!ConvertAtrributeInts(node, "axes", layer.squeeze().axes()))
                        return false;
                }
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src0 == NULL || src1 == NULL)
                    return false;
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const TensorParam & alpha = src1->meta().alpha();
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = LayerTypeMeta;
                    layer.meta().type() = MetaTypeSqueeze;
                    layer.meta().alpha() = alpha;
                }
                else
                {
                    layer.type() = Synet::LayerTypeSqueeze;
                    if (alpha.type() == TensorType64i)
                    {
                        layer.squeeze().axes().resize(alpha.i64().size());
                        for (size_t i = 0; i < alpha.i64().size(); ++i)
                            layer.squeeze().axes()[i] = (int)alpha.i64()[i];
                    }
                    else
                        return false;
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertSubNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src0 == NULL || src1 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta)
            {
                layer.type() = LayerTypeMeta;
                layer.meta().type() = MetaTypeSub;
            }
            else if (src1->type() == LayerTypeConst && TensorSize(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                const float* pShift = GetWeight<float>(original, src1->weight()[0]);
                layer.power().shift() = -pShift[0];
                layer.src().resize(1);
            }
            else if (src0->type() == LayerTypeConst && TensorSize(src0->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypePower;
                layer.power().scale() = -1.0f;
                const float* pShift = GetWeight<float>(original, src0->weight()[0]);
                layer.power().shift() = pShift[0];
                layer.src()[0] = layer.src()[1];
                layer.src().resize(1);
            }
            else if (src1->type() == LayerTypeConst && SignificantDimsCount(src1->weight()[0].dim()) == 1)
            {
                layer.type() = Synet::LayerTypeBias;
                layer.weight() = src1->weight();
                if (!CompactShape(layer.weight()[0].dim()))
                    return false;
                const float* pSrc = GetWeight<float>(original, layer.weight()[0]);
                float* pDst = GetWeight<float>(reordered, layer.weight()[0]);
                size_t size = TensorSize(layer.weight()[0].dim());
                for (size_t i = 0; i < size; ++i)
                    pDst[i] = -pSrc[i];
                layer.src().resize(1);
            }
            else
            {
                layer.type() = Synet::LayerTypeBinaryOperation;
                layer.binaryOperation().type() = BinaryOperationTypeSub;
            }
            return true;
        }

        bool ConvertTanhNode(const onnx::NodeProto& node, LayerParam& layer)
        {
            layer.type() = Synet::LayerTypeUnaryOperation;
            layer.unaryOperation().type() = UnaryOperationTypeTanh;
            return true;
        }

        bool ConvertTileNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;
            layer.type() = Synet::LayerTypeTile;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst && src1->meta().alpha().type() == TensorType64i)
            {
                Longs shape = src1->meta().alpha().i64();
                if (trans && !PermutedToNchw(layers, false, false, false))
                {
                    return false;
                }                
                for (size_t i = 0, already = 0; i < shape.size(); ++i)
                {
                    if (shape[i] != 1)
                    {
                        if (already)
                            return false;
                        layer.tile().axis() = i;
                        layer.tile().tiles() = (uint32_t)shape[i];
                        already = 1;
                    }
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertTopKNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 2))
                return false;
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            if (src1 == NULL)
                return false;

            layer.type() = Synet::LayerTypeTopK;
            if (src1->type() == LayerTypeMeta && src1->meta().type() == MetaTypeConst && src1->meta().alpha().type() == TensorType64i)
            {
                layer.topK().k() = src1->meta().alpha().i64()[0];
                layer.src().resize(1);
            }
            if (!ConvertAtrributeInt(node, "axis", layer.topK().axis()))
                return false;
            int64_t largest;
            if (!ConvertAtrributeInt(node, "largest", largest))
                return false;
            layer.topK().mode() = largest ? TopKModeMax : TopKModeMin;
            int64_t sorted;
            if (!ConvertAtrributeInt(node, "sorted", sorted))
                return false;
            layer.topK().sort() = sorted ? TopKSortValue : TopKSortIndex;
            layer.topK().indexElementType() = TensorType64i;

            return true;
        }

        bool ConvertTransposeNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const OnnxParam& onnxParam, LayerParam& layer, TensorFormatMap* tensorFormatMap)
        {
            if (!CheckSourceNumber(layer, 1))
                return false;
            Shape order;
            if (!ConvertAtrributeInts(node, "perm", order))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            if (src0 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypePermute;
                layer.meta().alpha().shape() = Shp(order.size());
                layer.meta().alpha().type() = TensorType64i;
                for (size_t i = 0; i < order.size(); ++i)
                    layer.meta().alpha().i64().push_back(order[i]);
            }
            else
            {
                layer.type() = Synet::LayerTypePermute;
                if (trans)
                {
                    bool permutedToNchw = CurrentTensorFormat(layers, layer.src(), true, false, onnxParam.globalPoolingPermuteToNchw(), tensorFormatMap) != TensorFormatNhwc;
                    if (!permutedToNchw)
                    {
                        if (order == Shape({ 0, 2, 1, 3, 4 }))
                            order = Shape({ 0, 1, 2, 4, 3 });
                        if (order == Shp(0, 1, 3, 2))
                            order = Shp(0, 2, 1, 3);
                        else if (order == Shape({ 0, 2, 3, 1 }))
                        {
                            order = Shape({ 0, 1, 2, 3 });
                            layer.permute().format() = TensorFormatNchw;
                        }
                        if (order == Shape({ 0, 2, 1 }))
                        {
                            order = Shape({ 0, 1, 2 });
                            layer.permute().format() = TensorFormatNchw;
                        }
                    }
                    else 
                    {
                        if (order == Shape({ 0, 3, 1, 2 }) && onnxParam.transpose0312PermuteToNhwc())
                        {
                            order = Shape({ 0, 1, 2, 3 });
                            layer.permute().format() = TensorFormatNhwc;
                        }
                    }
                }
                layer.permute().order() = order;
            }
            return true;
        }

        bool ConvertUnsqueezeNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 1, 2))
                return false;
            if (layer.src().size() == 1)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                if (src0 == NULL)
                    return false;
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeExpandDims;
                    layer.meta().alpha().type() = TensorType64i;
                    if (!ConvertAtrributeInts(node, "axes", layer.meta().alpha().i64()))
                        return false;
                    layer.meta().alpha().shape().resize(1, layer.meta().alpha().i64().size());
                }
                else
                {
                    layer.type() = Synet::LayerTypeExpandDims;
                    if (!ConvertAtrributeInts(node, "axes", layer.expandDims().axes()))
                        return false;
                }
            }
            else if (layer.src().size() == 2)
            {
                const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
                const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
                if (src0 == NULL || src1 == NULL)
                    return false;
                if (src1->type() != LayerTypeMeta || src1->meta().type() != MetaTypeConst)
                    return false;
                const TensorParam & alpha = src1->meta().alpha();
                if (src0->type() == LayerTypeMeta)
                {
                    layer.type() = Synet::LayerTypeMeta;
                    layer.meta().type() = Synet::MetaTypeExpandDims;
                    layer.meta().alpha() = alpha;
                }
                else
                {
                    layer.type() = Synet::LayerTypeExpandDims;
                    if (alpha.type() == TensorType64i)
                    {
                        layer.expandDims().axes().resize(alpha.i64().size());
                        for (size_t i = 0; i < alpha.i64().size(); ++i)
                            layer.expandDims().axes()[i] = (int)alpha.i64()[i];
                    }
                    else
                        return false;
                }
                layer.src().resize(1);
            }
            return true;
        }

        bool ConvertWhereNode(const onnx::NodeProto& node, const LayerParams& layers, LayerParam& layer)
        {
            if (!CheckSourceNumber(layer, 3))
                return false;
            const LayerParam* src0 = GetLayer(layers, layer.src()[0]);
            const LayerParam* src1 = GetLayer(layers, layer.src()[1]);
            const LayerParam* src2 = GetLayer(layers, layer.src()[2]);
            if (src0 == NULL || src1 == NULL || src2 == NULL)
                return false;
            if (src0->type() == LayerTypeMeta && src1->type() == LayerTypeMeta && src2->type() == LayerTypeMeta)
            {
                layer.type() = Synet::LayerTypeMeta;
                layer.meta().type() = MetaTypeSelect;
            }
            else
                layer.type() = Synet::LayerTypeWhere;
            return true;
        }

        //-----------------------------------------------------------------------------------------

        bool PrintGraph(const onnx::GraphProto& graph, std::ostream & os, bool printConst = false, bool filterInput = true)
        {
            os << std::endl;
            os << "graph name: " << graph.name() << std::endl;
            Consts consts;
            for (size_t i = 0; i < graph.initializer_size(); ++i)
                consts.insert(graph.initializer(i).name());
            for (size_t i = 0; i < graph.input_size(); ++i)
            {
                if (filterInput && consts.find(graph.input(i).name()) != consts.end())
                    continue;
                os << " input[" << i << "] " << ValueInfoString(graph.input(i)) << std::endl;
            }
            if (printConst)
            {
                for (size_t i = 0; i < graph.initializer_size(); ++i)
                    os << " const[" << i << "] " << TensorString(graph.initializer(i), 5) << std::endl;
            }
            for (size_t i = 0; i < graph.node_size(); ++i)
                os << " node[" << i << "] " << NodeString(graph.node(i)) << std::endl;
            for (size_t i = 0; i < graph.output_size(); ++i)
                os << " output[" << i << "] " << ValueInfoString(graph.output(i)) << std::endl;
            os << std::endl;
            return true;
        }

        String ValueInfoString(const onnx::ValueInfoProto& info)
        {
            std::stringstream ss;
            ss << info.name();
            if (info.type().has_tensor_type())
            {
                Shape shape = Synet::Convert(info.type().tensor_type().shape());
                ss << " {";
                for (size_t j = 0; j < shape.size(); ++j)
                    ss << " " << ptrdiff_t(shape[j]);
                ss << " }";
            }
            if (info.type().has_sequence_type())
            {
                ss <<  " ValueInfoString: I can't print sequence!";
            }
            if (info.type().has_map_type())
            {
                ss << " ValueInfoString: I can't print map!";
            }
            return ss.str();
        }

        String TensorString(const onnx::TensorProto& tensor, size_t printSizeMax = 3)
        {
            std::stringstream ss;
            ss << tensor.name() << " ";
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT: ss << "f32"; break;
            case onnx::TensorProto_DataType_INT32: ss << "i32"; break;
            case onnx::TensorProto_DataType_UINT8: ss << "u8"; break;
            case onnx::TensorProto_DataType_INT8: ss << "i8"; break;
            case onnx::TensorProto_DataType_INT64: ss << "i64"; break;
            case onnx::TensorProto_DataType_DOUBLE: ss << "f64"; break;
            default: ss << " unknown-" << tensor.data_type();
            }
            if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
                ss << " external";
            ss << " {";
            uint64_t size = 1;
            for (size_t i = 0; i < tensor.dims_size(); ++i)
            {
                ss << " " << tensor.dims(i);
                size *= tensor.dims(i);
            }
            size_t printSize = std::min<size_t>(printSizeMax, size);
            ss << " }";

            ss << "[";
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT: 
            {
                ss << std::fixed << std::setprecision(3);
                if (tensor.float_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.float_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((float*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_INT32:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((int32_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_UINT8:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << (int)((uint8_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_INT8:
            {
                if (tensor.int32_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int32_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << (int)((int8_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_INT64:
            {
                if (tensor.int64_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.int64_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((int64_t*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            case onnx::TensorProto_DataType_DOUBLE:
            {
                if (tensor.double_data_size())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << tensor.double_data(i);
                }
                if (tensor.has_raw_data())
                {
                    for (size_t i = 0; i < printSize; ++i)
                        ss << " " << ((double*)tensor.raw_data().c_str())[i];
                }
                break;
            }
            }
            if (size > printSize)
                ss << " ...";
            ss << " ]";
            return ss.str();
        }

        String AttributeString(const onnx::AttributeProto& attribute)
        {
            std::stringstream ss;
            ss << attribute.name() << ":(";
            switch (attribute.type())
            {
            case onnx::AttributeProto_AttributeType_INT:
                ss << attribute.i();
                break;
            case onnx::AttributeProto_AttributeType_FLOAT:
                ss << attribute.f();
                break;
            case onnx::AttributeProto_AttributeType_STRING:
                ss << attribute.s();
                break;
            case onnx::AttributeProto_AttributeType_TENSOR:
                ss << TensorString(attribute.t());
                break;
            case onnx::AttributeProto_AttributeType_INTS:
                for(size_t i = 0; i < attribute.ints_size(); ++i)
                    ss << (i ? " " : "") << attribute.ints(i);
                break;
            case onnx::AttributeProto_AttributeType_FLOATS:
                for (size_t i = 0; i < attribute.floats_size(); ++i)
                    ss << (i ? " " : "") << attribute.floats(i);
                break;
            default:
                ss << "unknown-" << attribute.type();
            }
            ss << ")";
            return ss.str();
        }

        String NodeString(const onnx::NodeProto& node)
        {
            std::stringstream ss;
            ss << "type: " << node.op_type() << ", name: " << node.name() << " (";
            for (size_t j = 0; j < node.input_size(); ++j)
                ss << " " << node.input(j);
            ss << " ) -> (";
            for (size_t j = 0; j < node.output_size(); ++j)
                ss << " " << node.output(j);
            ss << " ) {";
            for (size_t j = 0; j < node.attribute_size(); ++j)
                ss << " " << AttributeString(node.attribute(j));
            ss << " }";
            return ss.str();
        }

        void NotImplemented(const onnx::NodeProto& node, LayerParam& dst)
        {
            //dst.type() = LayerTypeStub;
            dst.debug().clear();
            dst.debug().push_back(NotImplementedMarker());
            dst.debug().push_back(node.op_type());
        }

        bool ErrorMessage(size_t index, const onnx::NodeProto& node)
        {
            SYNET_ERROR("Can't convert node[" << index << "]: " << NodeString(node) << " !");
        }
    };

    //---------------------------------------------------------------------------------------------

    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, const OnnxParam& onnxParam, const OptimizerParam& optParam);
}

#endif