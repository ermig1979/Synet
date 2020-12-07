/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
#include "Synet/Param.h"

namespace Synet
{
    SYNET_PARAM_ENUM(LayerType,
        LayerTypeAdd,
        LayerTypeBatchNorm,
        LayerTypeBias,
        LayerTypeBinaryOperation,
        LayerTypeCast,
        LayerTypeConcat,
        LayerTypeConst,
        LayerTypeConvolution,
        LayerTypeCtcGreedyDecoder,
        LayerTypeDeconvolution,
        LayerTypeDetectionOutput,
        LayerTypeDropout,
        LayerTypeEltwise,
        LayerTypeExpandDims,
        LayerTypeFill,
        LayerTypeFlatten,
        LayerTypeFused,
        LayerTypeGather,
        LayerTypeHswish,
        LayerTypeInnerProduct,
        LayerTypeInput,
        LayerTypeInterp,
        LayerTypeInterp2,
        LayerTypeLog,
        LayerTypeLrn,
        LayerTypeMergedConvolution,
        LayerTypeMeta,
        LayerTypeMish,
        LayerTypeNormalize,
        LayerTypePad,
        LayerTypePermute,
        LayerTypePooling,
        LayerTypePower,
        LayerTypePrelu,
        LayerTypePriorBox,
        LayerTypePriorBoxClustered,
        LayerTypeReduction,
        LayerTypeRegion,
        LayerTypeRelu,
        LayerTypeElu,
        LayerTypeReorg,
        LayerTypeReshape,
        LayerTypeRestrictRange,
        LayerTypeScale,
        LayerTypeShortcut,
        LayerTypeShuffle,
        LayerTypeSigmoid,
        LayerTypeSlice,
        LayerTypeSoftmax,
        LayerTypeSoftplus,
        LayerTypeSqueeze,
        LayerTypeSqueezeExcitation,
        LayerTypeStridedSlice,
        LayerTypeStub,
        LayerTypeSwitch,
        LayerTypeTile,
        LayerTypeUnaryOperation,
        LayerTypeUnpack,
        LayerTypeUpsample,
        LayerTypeYolo); 

    SYNET_PARAM_ENUM(ActivationFunctionType,
        ActivationFunctionTypeIdentity,
        ActivationFunctionTypeRelu,
        ActivationFunctionTypeLeakyRelu,
        ActivationFunctionTypeRestrictRange,
        ActivationFunctionTypePrelu,
        ActivationFunctionTypeElu,
        ActivationFunctionTypeHswish,
        ActivationFunctionTypeMish);

    SYNET_PARAM_ENUM(BinaryOperationType,
        BinaryOperationTypeDiv,
        BinaryOperationTypeSub);

    SYNET_PARAM_ENUM(EltwiseOperationType,
        EltwiseOperationTypeProduct,
        EltwiseOperationTypeSum,
        EltwiseOperationTypeMax,
        EltwiseOperationTypeMin);

    SYNET_PARAM_ENUM(InterpolationType,
        InterpolationTypeBilinear,
        InterpolationTypeNearest);

    SYNET_PARAM_ENUM(MetaType,
        MetaTypeAdd,
        MetaTypeCast,
        MetaTypeConst,
        MetaTypeExpandDims,
        MetaTypeFill,
        MetaTypeGather,
        MetaTypeGreater,
        MetaTypeInput,
        MetaTypeInputWithDefault,
        MetaTypeMaximum,
        MetaTypeMinimum,
        MetaTypeMul,
        MetaTypePack,
        MetaTypeRange,
        MetaTypeRealDiv,
        MetaTypeReduceProd,
        MetaTypeReshape,
        MetaTypeRsqrt,
        MetaTypeShape,
        MetaTypeSlice,
        MetaTypeSqrt,
        MetaTypeStridedSlice,
        MetaTypeStub,
        MetaTypeSub,
        MetaTypeSwitch,
        MetaTypeTensorArray,
        MetaTypeTensorArrayRead,
        MetaTypeTensorArraySize,
        MetaTypeTensorArrayWrite,
        MetaTypeTile,
        MetaTypeUnpack);

    SYNET_PARAM_ENUM(NormRegionType,
        NormRegionTypeAcrossChannels,
        NormRegionTypeWithinChannel);
    
    SYNET_PARAM_ENUM(PoolingMethodType,
        PoolingMethodTypeMax,
        PoolingMethodTypeAverage,
        PoolingMethodTypeStochastic);

    SYNET_PARAM_ENUM(PoolingPadType,
        PoolingPadTypeTensorflowSame);

    SYNET_PARAM_ENUM_(PriorBoxCodeType, Undefined, Size,
        PriorBoxCodeTypeUnknown,
        PriorBoxCodeTypeCorner,
        PriorBoxCodeTypeCenterSize,
        PriorBoxCodeTypeCornerSize);    
    
    SYNET_PARAM_ENUM(ReductionType,
        ReductionTypeMax,
        ReductionTypeSum);

    SYNET_PARAM_ENUM(RoundingType,
        RoundingTypeCeil,
        RoundingTypeFloor);

    SYNET_PARAM_ENUM(QuantizationMethod,
        QuantizationMethodIECompatible,
        QuantizationMethodSymmetricNarrowed,
        QuantizationMethodUnifiedNarrowed);

    SYNET_PARAM_ENUM(TensorFormat,
        TensorFormatNchw,
        TensorFormatNhwc,
        TensorFormatNchw4c,
        TensorFormatNchw8c,
        TensorFormatNchw16c,
        TensorFormatNchwXc,
        TensorFormatOiyx,
        TensorFormatYxio,
        TensorFormatOyxi4o,
        TensorFormatOyxi8o,
        TensorFormatOyxi16o,
        TensorFormatOyxiXo);

    SYNET_PARAM_ENUM(TensorType,
        TensorType32f,
        TensorType32i,
        TensorType8i,
        TensorType8u,
        TensorType64i);

    SYNET_PARAM_ENUM(UnaryOperationType,
        UnaryOperationTypeAbs,
        UnaryOperationTypeExp,
        UnaryOperationTypeLog,
        UnaryOperationTypeNeg,
        UnaryOperationTypeRsqrt,
        UnaryOperationTypeSqrt,
        UnaryOperationTypeTanh,
        UnaryOperationTypeZero);

    //-------------------------------------------------------------------------

    struct TensorParam
    {
        SYNET_PARAM_VALUE(TensorType, type, TensorTypeUnknown);
        SYNET_PARAM_VALUE(TensorFormat, format, TensorFormatUnknown);
        SYNET_PARAM_VALUE(Shape, shape, Shape());
        SYNET_PARAM_VALUE(Ints, i32, Ints());
        SYNET_PARAM_VALUE(Floats, f32, Floats());
        SYNET_PARAM_VALUE(Longs, i64, Longs());
    };

    struct WeightParam
    {
        SYNET_PARAM_VALUE(Shape, dim, Shape());
        SYNET_PARAM_VALUE(TensorType, type, TensorType32f);
        SYNET_PARAM_VALUE(TensorFormat, format, TensorFormatNchw);
        SYNET_PARAM_VALUE(size_t, offset, -1);
        SYNET_PARAM_VALUE(size_t, size, -1);
    };

    struct NonMaximumSuppressionParam
    {
        SYNET_PARAM_VALUE(float, nmsThreshold, 0.3f);
        SYNET_PARAM_VALUE(int32_t, topK, -1);
        SYNET_PARAM_VALUE(float, eta, 1.0f);
    };

    struct ShapeParam
    {
        SYNET_PARAM_VALUE(Shape, dim, Shape());
        SYNET_PARAM_VALUE(TensorFormat, format, TensorFormatNchw);
    };

    struct CastParam
    {
        SYNET_PARAM_VALUE(TensorType, type, TensorTypeUnknown);
    };

    struct BatchNormParam
    {
        SYNET_PARAM_VALUE(bool, useGlobalStats, true);
        SYNET_PARAM_VALUE(float, movingAverageFraction, 0.999f);
        SYNET_PARAM_VALUE(float, eps, 0.00001f);
        SYNET_PARAM_VALUE(bool, yoloCompatible, false);
    };

    struct BiasParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, numAxes, 1);
    };

    struct BinaryOperationParam
    {
        SYNET_PARAM_VALUE(BinaryOperationType, type, BinaryOperationTypeUnknown);
    };

    struct ConcatParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(bool, fixed, false);
    };

    struct ConvolutionParam
    {
        SYNET_PARAM_VALUE(uint32_t, outputNum, 0);
        SYNET_PARAM_VALUE(bool, biasTerm, true);
        SYNET_PARAM_VALUE(bool, autoPad, false);
        SYNET_PARAM_VALUE(Shape, kernel, Shape());
        SYNET_PARAM_VALUE(Shape, pad, Shape());
        SYNET_PARAM_VALUE(Shape, stride, Shape());
        SYNET_PARAM_VALUE(Shape, dilation, Shape());
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, group, 1);
        SYNET_PARAM_VALUE(ActivationFunctionType, activationType, ActivationFunctionTypeIdentity);
        SYNET_PARAM_VALUE(float, activationParam0, 0.0f);
        SYNET_PARAM_VALUE(float, activationParam1, 6.0f);
        SYNET_PARAM_VALUE(TensorType, quantizationLevel, TensorType32f);
    };

    struct DetectionOutputParam
    {
        SYNET_PARAM_VALUE(uint32_t, numClasses, 0);
        SYNET_PARAM_VALUE(bool, shareLocation, true);
        SYNET_PARAM_VALUE(int32_t, backgroundLabelId, 0);
        SYNET_PARAM_STRUCT(NonMaximumSuppressionParam, nms);
        SYNET_PARAM_VALUE(PriorBoxCodeType, codeType, PriorBoxCodeTypeCorner);
        SYNET_PARAM_VALUE(bool, varianceEncodedInTarget, false);
        SYNET_PARAM_VALUE(int32_t, keepTopK, -1);
        SYNET_PARAM_VALUE(float, confidenceThreshold, -FLT_MAX);
        SYNET_PARAM_VALUE(bool, keepMaxClassScoresOnly, false);
        SYNET_PARAM_VALUE(bool, clip, false);
    };

    struct ExpandDimsParam
    {
        SYNET_PARAM_VALUE(int32_t, axis, 0);
    };

    struct EltwiseParam
    {
        SYNET_PARAM_VALUE(EltwiseOperationType, operation, EltwiseOperationTypeSum);
        SYNET_PARAM_VALUE(Floats, coefficients, Floats());
    };

    struct EluParam
    {
        SYNET_PARAM_VALUE(float, alpha, 1.0f);
    };

    struct FillParam
    {
        SYNET_PARAM_VALUE(float, value, 0.0f);
    }; 

    struct FlattenParam
    {
        SYNET_PARAM_VALUE(int32_t, axis, 1);
        SYNET_PARAM_VALUE(int32_t, endAxis, -1);
    };

    struct FusedParam
    {
        SYNET_PARAM_VALUE(int, type, -1);
        SYNET_PARAM_VALUE(int32_t, axis, 1);
        SYNET_PARAM_VALUE(Floats, floats, Floats());
    };

    struct HswishParam
    {
        SYNET_PARAM_VALUE(float, shift, 3.0f);
        SYNET_PARAM_VALUE(float, scale, 1.0f / 6.0f);
    };
    
    struct InnerProductParam
    {
        SYNET_PARAM_VALUE(uint32_t, outputNum, 0);
        SYNET_PARAM_VALUE(bool, biasTerm, true);
        SYNET_PARAM_VALUE(bool, transposeA, false);
        SYNET_PARAM_VALUE(bool, transposeB, false);
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(TensorType, quantizationLevel, TensorType32f);
    };

    struct InputParam
    {
        SYNET_PARAM_VECTOR(ShapeParam, shape);
    };

    struct InterpParam
    {
        SYNET_PARAM_VALUE(int32_t, height, 0);
        SYNET_PARAM_VALUE(int32_t, width, 0);
        SYNET_PARAM_VALUE(int32_t, zoomFactor, 1);
        SYNET_PARAM_VALUE(int32_t, shrinkFactor, 1);
        SYNET_PARAM_VALUE(int32_t, cropBeg, 0);
        SYNET_PARAM_VALUE(int32_t, cropEnd, 0);
        SYNET_PARAM_VALUE(bool, useTensorSize, false);
        SYNET_PARAM_VALUE(InterpolationType, interpolationType, InterpolationTypeBilinear);
    };

    struct Interp2Param
    {
        SYNET_PARAM_VALUE(float, factor, 1.0f);
        SYNET_PARAM_VALUE(int32_t, height, 0);
        SYNET_PARAM_VALUE(int32_t, width, 0);
        SYNET_PARAM_VALUE(Shape, pad, Shape());
        SYNET_PARAM_VALUE(bool, alignCorners, false);
    };

    struct LogParam
    {
        SYNET_PARAM_VALUE(float, base, -1.0f);
        SYNET_PARAM_VALUE(float, scale, 1.0f);
        SYNET_PARAM_VALUE(float, shift, 0.0f);
    };

    struct LrnParam
    {
        SYNET_PARAM_VALUE(uint32_t, localSize, 5);
        SYNET_PARAM_VALUE(float, alpha, 1.0f);
        SYNET_PARAM_VALUE(float, beta, 0.75f);
        SYNET_PARAM_VALUE(NormRegionType, normRegion, NormRegionTypeAcrossChannels);
        SYNET_PARAM_VALUE(float, k, 1.0f);
    };

    struct MergedConvolutionParam
    {
        SYNET_PARAM_VECTOR(ConvolutionParam, conv);
        SYNET_PARAM_VALUE(bool, add, false);
    };

    struct MetaParam
    {
        SYNET_PARAM_VALUE(MetaType, type, MetaTypeUnknown);
        SYNET_PARAM_VALUE(int32_t, version, 0);
        SYNET_PARAM_STRUCT(TensorParam, alpha);
    };

    struct NormalizeParam
    {
        SYNET_PARAM_VALUE(bool, acrossSpatial, true);
        SYNET_PARAM_VALUE(bool, channelShared, true);
        SYNET_PARAM_VALUE(float, eps, 1e-10f);
    };

    struct PermuteParam
    {
        SYNET_PARAM_VALUE(Shape, order, Shape());
        SYNET_PARAM_VALUE(TensorFormat, format, TensorFormatUnknown);
    };

    struct PoolingParam
    {
        SYNET_PARAM_VALUE(PoolingMethodType, method, PoolingMethodTypeUnknown);
        SYNET_PARAM_VALUE(Shape, kernel, Shape());
        SYNET_PARAM_VALUE(Shape, pad, Shape());
        SYNET_PARAM_VALUE(Shape, stride, Shape());
        SYNET_PARAM_VALUE(bool, globalPooling, false);
        SYNET_PARAM_VALUE(int, yoloCompatible, 0);
        SYNET_PARAM_VALUE(PoolingPadType, padType, PoolingPadTypeUnknown);
        SYNET_PARAM_VALUE(RoundingType, roundingType, RoundingTypeCeil);
        SYNET_PARAM_VALUE(bool, excludePad, true);
    };

    struct PowerParam
    {
        SYNET_PARAM_VALUE(float, power, 1.0f);
        SYNET_PARAM_VALUE(float, scale, 1.0f);
        SYNET_PARAM_VALUE(float, shift, 0.0f);
    };

    struct PreluParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct PriorBoxParam
    {
        SYNET_PARAM_VALUE(int, version, 0);
        SYNET_PARAM_VALUE(Floats, minSize, Floats());
        SYNET_PARAM_VALUE(Floats, maxSize, Floats());
        SYNET_PARAM_VALUE(Floats, aspectRatio, Floats());
        SYNET_PARAM_VALUE(bool, flip, true);
        SYNET_PARAM_VALUE(bool, clip, false);
        SYNET_PARAM_VALUE(Floats, variance, Floats());
        SYNET_PARAM_VALUE(Shape, imgSize, Shape());
        SYNET_PARAM_VALUE(Floats, step, Floats());
        SYNET_PARAM_VALUE(float, offset, 0.5f);
        SYNET_PARAM_VALUE(bool, scaleAllSizes, true);
    };

    struct PriorBoxClusteredParam
    {
        SYNET_PARAM_VALUE(Floats, widths, Floats());
        SYNET_PARAM_VALUE(Floats, heights, Floats());
        SYNET_PARAM_VALUE(bool, clip, false);
        SYNET_PARAM_VALUE(Floats, variance, Floats());
        SYNET_PARAM_VALUE(int, imgH, 0);
        SYNET_PARAM_VALUE(int, imgW, 0);
        SYNET_PARAM_VALUE(float, step, 0.0f);
        SYNET_PARAM_VALUE(float, stepH, 0.0f);
        SYNET_PARAM_VALUE(float, stepW, 0.0f);
        SYNET_PARAM_VALUE(float, offset, 0.5f);
    };

    struct ReductionParam
    {
        SYNET_PARAM_VALUE(ReductionType, type, ReductionTypeUnknown);
        SYNET_PARAM_VALUE(Ints, axis, Ints());
        SYNET_PARAM_VALUE(bool, keepDims, true);
    };

    struct RegionParam
    {
        SYNET_PARAM_VALUE(uint32_t, coords, 4);
        SYNET_PARAM_VALUE(uint32_t, classes, 20);
        SYNET_PARAM_VALUE(uint32_t, num, 1);
        SYNET_PARAM_VALUE(bool, softmax, false);
        SYNET_PARAM_VALUE(Floats, anchors, Floats());
    };

    struct ReluParam
    {
        SYNET_PARAM_VALUE(float, negativeSlope, 0.0f);
    };

    struct ReorgParam
    {
        SYNET_PARAM_VALUE(bool, reverse, true);
        SYNET_PARAM_VALUE(uint32_t, stride, 1);
    };

    struct ReshapeParam
    {
        SYNET_PARAM_VALUE(Shape, shape, Shape());
        SYNET_PARAM_VALUE(int32_t, axis, 0);
        SYNET_PARAM_VALUE(int32_t, numAxes, -1);
    };

    struct RestrictRangeParam
    {
        SYNET_PARAM_VALUE(float, lower, -FLT_MAX);
        SYNET_PARAM_VALUE(float, upper, +FLT_MAX);
    };

    struct ScaleParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, numAxes, 1);
        SYNET_PARAM_VALUE(bool, biasTerm, false);
    };

    struct ShuffleParam
    {
        SYNET_PARAM_VALUE(int, type, 0);
    };

    struct SliceParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(Index, slicePoint, Index());
    };

    struct SoftmaxParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct SoftplusParam
    {
        SYNET_PARAM_VALUE(float, beta, 1.0f);
        SYNET_PARAM_VALUE(float, threshold, 20.0f);
    };

    struct StridedSliceParam
    {
        SYNET_PARAM_VALUE(Shape, beginMask, Shape());
        SYNET_PARAM_VALUE(Shape, ellipsisMask, Shape());
        SYNET_PARAM_VALUE(Shape, endMask, Shape());
        SYNET_PARAM_VALUE(Shape, newAxisMask, Shape());
        SYNET_PARAM_VALUE(Shape, shrinkAxisMask, Shape());
        SYNET_PARAM_VALUE(Shape, beginDims, Shape());
        SYNET_PARAM_VALUE(Shape, endDims, Shape());
        SYNET_PARAM_VALUE(Shape, strideDims, Shape());
    };

    struct TileParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, tiles, 1);
    };

    struct UnaryOperationParam
    {
        SYNET_PARAM_VALUE(UnaryOperationType, type, UnaryOperationTypeUnknown);
    };

    struct UnpackParam
    {
        SYNET_PARAM_VALUE(int32_t, axis, 0);
    };

    struct UpsampleParam
    {
        SYNET_PARAM_VALUE(int32_t, stride, 2);
        SYNET_PARAM_VALUE(float, scale, 1.0f);
    };

    struct YoloParam
    {
        SYNET_PARAM_VALUE(uint32_t, classes, 20);
        SYNET_PARAM_VALUE(uint32_t, num, 1);
        SYNET_PARAM_VALUE(uint32_t, total, 1);
        SYNET_PARAM_VALUE(uint32_t, max, 30);
        SYNET_PARAM_VALUE(float, jitter, 0.2f);
        SYNET_PARAM_VALUE(float, ignoreThresh, 0.5f);
        SYNET_PARAM_VALUE(float, truthThresh, 1.0f);
        SYNET_PARAM_VALUE(Index, mask, Index());
        SYNET_PARAM_VALUE(Floats, anchors, Floats());
    };

    struct LayerParam
    {
        SYNET_PARAM_VALUE(LayerType, type, LayerTypeUnknown);
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VALUE(Strings, src, Strings());
        SYNET_PARAM_VALUE(Strings, dst, Strings());
        SYNET_PARAM_VECTOR(WeightParam, weight);
        SYNET_PARAM_VALUE(Strings, origin, Strings());

        SYNET_PARAM_STRUCT(BatchNormParam, batchNorm);
        SYNET_PARAM_STRUCT(BiasParam, bias);
        SYNET_PARAM_STRUCT(BinaryOperationParam, binaryOperation);
        SYNET_PARAM_STRUCT(CastParam, cast);
        SYNET_PARAM_STRUCT(ConcatParam, concat);
        SYNET_PARAM_STRUCT(ConvolutionParam, convolution);
        SYNET_PARAM_STRUCT(DetectionOutputParam, detectionOutput);
        SYNET_PARAM_STRUCT(EltwiseParam, eltwise);
        SYNET_PARAM_STRUCT(EluParam, elu);
        SYNET_PARAM_STRUCT(ExpandDimsParam, expandDims);
        SYNET_PARAM_STRUCT(FillParam, fill);
        SYNET_PARAM_STRUCT(FlattenParam, flatten);
        SYNET_PARAM_STRUCT(FusedParam, fused);
        SYNET_PARAM_STRUCT(HswishParam, hswish);
        SYNET_PARAM_STRUCT(InnerProductParam, innerProduct);
        SYNET_PARAM_STRUCT(InputParam, input);
        SYNET_PARAM_STRUCT(InterpParam, interp);
        SYNET_PARAM_STRUCT(Interp2Param, interp2);
        SYNET_PARAM_STRUCT(LogParam, log);
        SYNET_PARAM_STRUCT(LrnParam, lrn);
        SYNET_PARAM_STRUCT(MergedConvolutionParam, mergedConvolution);
        SYNET_PARAM_STRUCT(MetaParam, meta);
        SYNET_PARAM_STRUCT(NormalizeParam, normalize);
        SYNET_PARAM_STRUCT(PermuteParam, permute);
        SYNET_PARAM_STRUCT(PoolingParam, pooling);
        SYNET_PARAM_STRUCT(PowerParam, power);
        SYNET_PARAM_STRUCT(PreluParam, prelu);
        SYNET_PARAM_STRUCT(PriorBoxParam, priorBox);
        SYNET_PARAM_STRUCT(PriorBoxClusteredParam, priorBoxClustered);
        SYNET_PARAM_STRUCT(ReductionParam, reduction);
        SYNET_PARAM_STRUCT(RegionParam, region);
        SYNET_PARAM_STRUCT(ReluParam, relu);
        SYNET_PARAM_STRUCT(ReorgParam, reorg);
        SYNET_PARAM_STRUCT(ReshapeParam, reshape);
        SYNET_PARAM_STRUCT(RestrictRangeParam, restrictRange);
        SYNET_PARAM_STRUCT(ScaleParam, scale);
        SYNET_PARAM_STRUCT(ShuffleParam, shuffle);
        SYNET_PARAM_STRUCT(SliceParam, slice);
        SYNET_PARAM_STRUCT(SoftmaxParam, softmax);
        SYNET_PARAM_STRUCT(SoftplusParam, softplus);
        SYNET_PARAM_STRUCT(StridedSliceParam, stridedSlice);
        SYNET_PARAM_STRUCT(TileParam, tile);
        SYNET_PARAM_STRUCT(UnaryOperationParam, unaryOperation);
        SYNET_PARAM_STRUCT(UnpackParam, unpack);
        SYNET_PARAM_STRUCT(UpsampleParam, upsample);
        SYNET_PARAM_STRUCT(YoloParam, yolo);

        SYNET_PARAM_VALUE(Strings, debug, Strings());
    };

    struct StatisticParam
    {
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VALUE(Floats, min, Floats());
        SYNET_PARAM_VALUE(Floats, max, Floats());
    };

    struct QuantizationParam
    {
        SYNET_PARAM_VALUE(QuantizationMethod, method, QuantizationMethodUnknown);
        SYNET_PARAM_VECTOR(StatisticParam, statistics);
    };

    struct NetworkParam
    {
        SYNET_PARAM_VALUE(int32_t, version, 0);
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VALUE(Strings, dst, Strings());
        SYNET_PARAM_VECTOR(LayerParam, layers);
        SYNET_PARAM_STRUCT(QuantizationParam, quantization);
    };

    SYNET_PARAM_HOLDER(NetworkParamHolder, NetworkParam, network);
}