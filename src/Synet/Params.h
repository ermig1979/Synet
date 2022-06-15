/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

#define SYNET_PARAM_ENUM1(ns1, type, beg, end, ...) \
    namespace ns1 { CPL_PARAM_ENUM_DECL(type, beg, end, __VA_ARGS__) } \
    CPL_PARAM_ENUM_CONV(ns1, type, beg, end, __VA_ARGS__)

CPL_PARAM_ENUM1(Synet, LayerType,
    LayerTypeAdd,
    LayerTypeBatchNorm,
    LayerTypeBias,
    LayerTypeBinaryOperation,
    LayerTypeBroadcast,
    LayerTypeCast,
    LayerTypeConcat,
    LayerTypeConst,
    LayerTypeConvolution,
    LayerTypeCtcGreedyDecoder,
    LayerTypeDeconvolution,
    LayerTypeDetectionOutput,
    LayerTypeDropout,
    LayerTypeEltwise,
    LayerTypeElu,
    LayerTypeExpandDims,
    LayerTypeFill,
    LayerTypeFlatten,
    LayerTypeFused,
    LayerTypeGather,
    LayerTypeHardSigmoid,
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
    LayerTypeNonMaxSuppression,
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
    LayerTypeReorg,
    LayerTypeReshape,
    LayerTypeRestrictRange,
    LayerTypeReverseSequence,
    LayerTypeRnnGruBd,
    LayerTypeScale,
    LayerTypeShortcut,
    LayerTypeShuffle,
    LayerTypeSigmoid,
    LayerTypeSlice,
    LayerTypeSoftmax,
    LayerTypeSoftplus,
    LayerTypeSpaceToDepth,
    LayerTypeSqueeze,
    LayerTypeSqueezeExcitation,
    LayerTypeStridedSlice,
    LayerTypeStub,
    LayerTypeSwish,
    LayerTypeSwitch,
    LayerTypeTensorIterator,
    LayerTypeTile,
    LayerTypeTopK,
    LayerTypeUnaryOperation,
    LayerTypeUnpack,
    LayerTypeUpsample,
    LayerTypeYolo); 

CPL_PARAM_ENUM1(Synet, ActivationFunctionType,
    ActivationFunctionTypeIdentity,
    ActivationFunctionTypeRelu,
    ActivationFunctionTypeLeakyRelu,
    ActivationFunctionTypeRestrictRange,
    ActivationFunctionTypePrelu,
    ActivationFunctionTypeElu,
    ActivationFunctionTypeHswish,
    ActivationFunctionTypeMish,
    ActivationFunctionTypeHardSigmoid,
    ActivationFunctionTypeSwish);

CPL_PARAM_ENUM1(Synet, BinaryOperationType,
    BinaryOperationTypeDiv,
    BinaryOperationTypeSub);

CPL_PARAM_ENUM1(Synet, BoxEncodingType,
    BoxEncodingTypeCorner,
    BoxEncodingTypeCenter);

CPL_PARAM_ENUM1(Synet, CoordinateTransformType,
    CoordinateTransformTypeLegacy,
    CoordinateTransformTypeHalfPixel,
    CoordinateTransformTypeCaffe,
    CoordinateTransformTypePytorch);

CPL_PARAM_ENUM1(Synet, EltwiseOperationType,
    EltwiseOperationTypeProduct,
    EltwiseOperationTypeSum,
    EltwiseOperationTypeMax,
    EltwiseOperationTypeMin);

CPL_PARAM_ENUM1(Synet, InterpolationType,
    InterpolationTypeBilinear,
    InterpolationTypeNearest);

CPL_PARAM_ENUM1(Synet, MetaType,
    MetaTypeAdd,
    MetaTypeCast,
    MetaTypeConst,
    MetaTypeDiv,
    MetaTypeEqual,
    MetaTypeExpandDims,
    MetaTypeFill,
    MetaTypeFloor,
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
    MetaTypeReduceMin,
    MetaTypeReduceProd,
    MetaTypeReshape,
    MetaTypeRsqrt,
    MetaTypeSelect,
    MetaTypeShape,
    MetaTypeSlice,
    MetaTypeSqrt,
    MetaTypeSqueeze,
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

CPL_PARAM_ENUM1(Synet, NormRegionType,
    NormRegionTypeAcrossChannels,
    NormRegionTypeWithinChannel);
    
CPL_PARAM_ENUM1(Synet, PoolingMethodType,
    PoolingMethodTypeMax,
    PoolingMethodTypeAverage,
    PoolingMethodTypeStochastic);

CPL_PARAM_ENUM1(Synet, PoolingPadType,
    PoolingPadTypeTensorflowSame);

SYNET_PARAM_ENUM1(Synet, PriorBoxCodeType, Undefined, Size,
    PriorBoxCodeTypeUnknown,
    PriorBoxCodeTypeCorner,
    PriorBoxCodeTypeCenterSize,
    PriorBoxCodeTypeCornerSize);    
    
CPL_PARAM_ENUM1(Synet, ReductionType,
    ReductionTypeMax,
    ReductionTypeMin,
    ReductionTypeSum,
    ReductionTypeProd);

CPL_PARAM_ENUM1(Synet, RoundingType,
    RoundingTypeCeil,
    RoundingTypeFloor);

CPL_PARAM_ENUM1(Synet, QuantizationMethod,
    QuantizationMethodIECompatible,
    QuantizationMethodSymmetricNarrowed,
    QuantizationMethodUnifiedNarrowed);

CPL_PARAM_ENUM1(Synet, TensorFormat,
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

CPL_PARAM_ENUM1(Synet, TensorType,
    TensorType32f,
    TensorType32i,
    TensorType8i,
    TensorType8u,
    TensorType64i,
    TensorType64u);

CPL_PARAM_ENUM1(Synet, TopKMode,
    TopKModeMax,
    TopKModeMin);

CPL_PARAM_ENUM1(Synet, TopKSort,
    TopKSortValue,
    TopKSortIndex,
    TopKSortNone);

CPL_PARAM_ENUM1(Synet, UnaryOperationType,
    UnaryOperationTypeAbs,
    UnaryOperationTypeExp,
    UnaryOperationTypeLog,
    UnaryOperationTypeNeg,
    UnaryOperationTypeRsqrt,
    UnaryOperationTypeSqrt,
    UnaryOperationTypeTanh,
    UnaryOperationTypeZero);

namespace Synet
{
    //-------------------------------------------------------------------------

    struct TensorParam
    {
        CPL_PARAM_VALUE(TensorType, type, TensorTypeUnknown);
        CPL_PARAM_VALUE(TensorFormat, format, TensorFormatUnknown);
        CPL_PARAM_VALUE(Shape, shape, Shape());
        CPL_PARAM_VALUE(Ints, i32, Ints());
        CPL_PARAM_VALUE(Floats, f32, Floats());
        CPL_PARAM_VALUE(Longs, i64, Longs());
        CPL_PARAM_VALUE(ULongs, u64, ULongs());
    };

    struct WeightParam
    {
        CPL_PARAM_VALUE(Shape, dim, Shape());
        CPL_PARAM_VALUE(TensorType, type, TensorType32f);
        CPL_PARAM_VALUE(TensorFormat, format, TensorFormatNchw);
        CPL_PARAM_VALUE(size_t, offset, -1);
        CPL_PARAM_VALUE(size_t, size, -1);
    };

    struct NonMaximumSuppressionParam
    {
        CPL_PARAM_VALUE(float, nmsThreshold, 0.3f);
        CPL_PARAM_VALUE(int32_t, topK, -1);
        CPL_PARAM_VALUE(float, eta, 1.0f);
    };

    struct ShapeParam
    {
        CPL_PARAM_VALUE(Shape, dim, Shape());
        CPL_PARAM_VALUE(TensorFormat, format, TensorFormatNchw);
    };

    struct CastParam
    {
        CPL_PARAM_VALUE(TensorType, type, TensorTypeUnknown);
    };

    struct BatchNormParam
    {
        CPL_PARAM_VALUE(bool, useGlobalStats, true);
        CPL_PARAM_VALUE(float, movingAverageFraction, 0.999f);
        CPL_PARAM_VALUE(float, eps, 0.00001f);
        CPL_PARAM_VALUE(bool, yoloCompatible, false);
    };

    struct BiasParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(uint32_t, numAxes, 1);
    };

    struct BinaryOperationParam
    {
        CPL_PARAM_VALUE(BinaryOperationType, type, BinaryOperationTypeUnknown);
    };

    struct BroadcastParam
    {
        CPL_PARAM_VALUE(bool, fixed, false);
    };

    struct ConcatParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(bool, fixed, false);
        CPL_PARAM_VALUE(bool, can8i, true);
    };

    struct ConvolutionParam
    {
        CPL_PARAM_VALUE(uint32_t, outputNum, 0);
        CPL_PARAM_VALUE(bool, biasTerm, true);
        CPL_PARAM_VALUE(bool, autoPad, false);
        CPL_PARAM_VALUE(Shape, kernel, Shape());
        CPL_PARAM_VALUE(Shape, pad, Shape());
        CPL_PARAM_VALUE(Shape, stride, Shape());
        CPL_PARAM_VALUE(Shape, dilation, Shape());
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(uint32_t, group, 1);
        CPL_PARAM_VALUE(ActivationFunctionType, activationType, ActivationFunctionTypeIdentity);
        CPL_PARAM_VALUE(float, activationParam0, 0.0f);
        CPL_PARAM_VALUE(float, activationParam1, 6.0f);
        CPL_PARAM_VALUE(TensorType, quantizationLevel, TensorType32f);
        CPL_PARAM_VALUE(bool, bf16, false);
    };

    struct DetectionOutputParam
    {
        CPL_PARAM_VALUE(uint32_t, numClasses, 0);
        CPL_PARAM_VALUE(bool, shareLocation, true);
        CPL_PARAM_VALUE(int32_t, backgroundLabelId, 0);
        CPL_PARAM_STRUCT(NonMaximumSuppressionParam, nms);
        CPL_PARAM_VALUE(PriorBoxCodeType, codeType, PriorBoxCodeTypeCorner);
        CPL_PARAM_VALUE(bool, varianceEncodedInTarget, false);
        CPL_PARAM_VALUE(int32_t, keepTopK, -1);
        CPL_PARAM_VALUE(float, confidenceThreshold, -FLT_MAX);
        CPL_PARAM_VALUE(bool, keepMaxClassScoresOnly, false);
        CPL_PARAM_VALUE(bool, clip, false);
    };

    struct ExpandDimsParam
    {
        CPL_PARAM_VALUE(int32_t, axis, 0);
        CPL_PARAM_VALUE(Ints, axes, Ints());
    };

    struct EltwiseParam
    {
        CPL_PARAM_VALUE(EltwiseOperationType, operation, EltwiseOperationTypeSum);
        CPL_PARAM_VALUE(Floats, coefficients, Floats());
    };

    struct EluParam
    {
        CPL_PARAM_VALUE(float, alpha, 1.0f);
    };

    struct FillParam
    {
        CPL_PARAM_VALUE(float, value, 0.0f);
    }; 

    struct FlattenParam
    {
        CPL_PARAM_VALUE(int32_t, axis, 1);
        CPL_PARAM_VALUE(int32_t, endAxis, -1);
    };

    struct FusedParam
    {
        CPL_PARAM_VALUE(int, type, -1);
        CPL_PARAM_VALUE(int32_t, axis, 1);
        CPL_PARAM_VALUE(Floats, floats, Floats());
    };

    struct GatherParam
    {
        CPL_PARAM_VALUE(int, batchDims, 0);
        CPL_PARAM_VALUE(int, axis, 0);
    };

    struct HardSigmoidParam
    {
        CPL_PARAM_VALUE(float, scale, 1.0f / 6.0f);
        CPL_PARAM_VALUE(float, shift, 0.5f);
    };

    struct HswishParam
    {
        CPL_PARAM_VALUE(float, shift, 3.0f);
        CPL_PARAM_VALUE(float, scale, 1.0f / 6.0f);
    };
    
    struct InnerProductParam
    {
        CPL_PARAM_VALUE(uint32_t, outputNum, 0);
        CPL_PARAM_VALUE(bool, biasTerm, true);
        CPL_PARAM_VALUE(bool, transposeA, false);
        CPL_PARAM_VALUE(bool, transposeB, false);
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(TensorType, quantizationLevel, TensorType32f);
        CPL_PARAM_VALUE(bool, bf16, false);
    };

    struct InputParam
    {
        CPL_PARAM_VECTOR(ShapeParam, shape);
    };

    struct InterpParam
    {
        CPL_PARAM_VALUE(int32_t, height, 0);
        CPL_PARAM_VALUE(int32_t, width, 0);
        CPL_PARAM_VALUE(int32_t, zoomFactor, 1);
        CPL_PARAM_VALUE(int32_t, shrinkFactor, 1);
        CPL_PARAM_VALUE(int32_t, cropBeg, 0);
        CPL_PARAM_VALUE(int32_t, cropEnd, 0);
        CPL_PARAM_VALUE(bool, useTensorSize, false);
        CPL_PARAM_VALUE(InterpolationType, interpolationType, InterpolationTypeBilinear);
        CPL_PARAM_VALUE(CoordinateTransformType, coordinateTransformType, CoordinateTransformTypeLegacy);
    };

    struct Interp2Param
    {
        CPL_PARAM_VALUE(float, factor, 1.0f);
        CPL_PARAM_VALUE(int32_t, height, 0);
        CPL_PARAM_VALUE(int32_t, width, 0);
        CPL_PARAM_VALUE(Shape, pad, Shape());
        CPL_PARAM_VALUE(bool, alignCorners, false);
    };

    struct LogParam
    {
        CPL_PARAM_VALUE(float, base, -1.0f);
        CPL_PARAM_VALUE(float, scale, 1.0f);
        CPL_PARAM_VALUE(float, shift, 0.0f);
    };

    struct LrnParam
    {
        CPL_PARAM_VALUE(uint32_t, localSize, 5);
        CPL_PARAM_VALUE(float, alpha, 1.0f);
        CPL_PARAM_VALUE(float, beta, 0.75f);
        CPL_PARAM_VALUE(NormRegionType, normRegion, NormRegionTypeAcrossChannels);
        CPL_PARAM_VALUE(float, k, 1.0f);
    };

    struct MergedConvolutionParam
    {
        CPL_PARAM_VECTOR(ConvolutionParam, conv);
        CPL_PARAM_VALUE(bool, add, false);
    };

    struct MetaParam
    {
        CPL_PARAM_VALUE(MetaType, type, MetaTypeUnknown);
        CPL_PARAM_VALUE(int32_t, version, 0);
        CPL_PARAM_STRUCT(TensorParam, alpha);
    };

    struct NonMaxSuppressionParam
    {
        CPL_PARAM_VALUE(BoxEncodingType, boxEncoding, BoxEncodingTypeCorner);
        CPL_PARAM_VALUE(bool, sortResultDescending, false);
        CPL_PARAM_VALUE(TensorType, outputType, TensorType64i);
        CPL_PARAM_VALUE(int, maxOutputBoxesPerClass, 0);
        CPL_PARAM_VALUE(float, iouThreshold, 0);
        CPL_PARAM_VALUE(float, scoreThreshold, 0);
        CPL_PARAM_VALUE(float, softNmsSigma, 0);
    };

    struct NormalizeParam
    {
        CPL_PARAM_VALUE(bool, acrossSpatial, true);
        CPL_PARAM_VALUE(bool, channelShared, true);
        CPL_PARAM_VALUE(float, eps, 1e-10f);
    };

    struct PermuteParam
    {
        CPL_PARAM_VALUE(Shape, order, Shape());
        CPL_PARAM_VALUE(TensorFormat, format, TensorFormatUnknown);
    };

    struct PoolingParam
    {
        CPL_PARAM_VALUE(PoolingMethodType, method, PoolingMethodTypeUnknown);
        CPL_PARAM_VALUE(Shape, kernel, Shape());
        CPL_PARAM_VALUE(Shape, pad, Shape());
        CPL_PARAM_VALUE(Shape, stride, Shape());
        CPL_PARAM_VALUE(bool, globalPooling, false);
        CPL_PARAM_VALUE(int, yoloCompatible, 0);
        CPL_PARAM_VALUE(PoolingPadType, padType, PoolingPadTypeUnknown);
        CPL_PARAM_VALUE(RoundingType, roundingType, RoundingTypeCeil);
        CPL_PARAM_VALUE(bool, excludePad, true);
    };

    struct PowerParam
    {
        CPL_PARAM_VALUE(float, power, 1.0f);
        CPL_PARAM_VALUE(float, scale, 1.0f);
        CPL_PARAM_VALUE(float, shift, 0.0f);
    };

    struct PreluParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct PriorBoxParam
    {
        CPL_PARAM_VALUE(int, version, 0);
        CPL_PARAM_VALUE(Floats, minSize, Floats());
        CPL_PARAM_VALUE(Floats, maxSize, Floats());
        CPL_PARAM_VALUE(Floats, aspectRatio, Floats());
        CPL_PARAM_VALUE(bool, flip, true);
        CPL_PARAM_VALUE(bool, clip, false);
        CPL_PARAM_VALUE(Floats, variance, Floats());
        CPL_PARAM_VALUE(Shape, imgSize, Shape());
        CPL_PARAM_VALUE(Floats, step, Floats());
        CPL_PARAM_VALUE(float, offset, 0.5f);
        CPL_PARAM_VALUE(bool, scaleAllSizes, true);
    };

    struct PriorBoxClusteredParam
    {
        CPL_PARAM_VALUE(Floats, widths, Floats());
        CPL_PARAM_VALUE(Floats, heights, Floats());
        CPL_PARAM_VALUE(bool, clip, false);
        CPL_PARAM_VALUE(Floats, variance, Floats());
        CPL_PARAM_VALUE(int, imgH, 0);
        CPL_PARAM_VALUE(int, imgW, 0);
        CPL_PARAM_VALUE(float, step, 0.0f);
        CPL_PARAM_VALUE(float, stepH, 0.0f);
        CPL_PARAM_VALUE(float, stepW, 0.0f);
        CPL_PARAM_VALUE(float, offset, 0.5f);
    };

    struct ReductionParam
    {
        CPL_PARAM_VALUE(ReductionType, type, ReductionTypeUnknown);
        CPL_PARAM_VALUE(Ints, axis, Ints());
        CPL_PARAM_VALUE(bool, keepDims, true);
    };

    struct RegionParam
    {
        CPL_PARAM_VALUE(uint32_t, coords, 4);
        CPL_PARAM_VALUE(uint32_t, classes, 20);
        CPL_PARAM_VALUE(uint32_t, num, 1);
        CPL_PARAM_VALUE(bool, softmax, false);
        CPL_PARAM_VALUE(Floats, anchors, Floats());
    };

    struct ReluParam
    {
        CPL_PARAM_VALUE(float, negativeSlope, 0.0f);
    };

    struct ReorgParam
    {
        CPL_PARAM_VALUE(bool, reverse, true);
        CPL_PARAM_VALUE(uint32_t, stride, 1);
    };

    struct ReshapeParam
    {
        CPL_PARAM_VALUE(Shape, shape, Shape());
        CPL_PARAM_VALUE(int32_t, axis, 0);
        CPL_PARAM_VALUE(int32_t, numAxes, -1);
    };

    struct RestrictRangeParam
    {
        CPL_PARAM_VALUE(float, lower, -FLT_MAX);
        CPL_PARAM_VALUE(float, upper, +FLT_MAX);
    };

    struct ReverseSequenceParam
    {
        CPL_PARAM_VALUE(int32_t, batchAxis, 0);
        CPL_PARAM_VALUE(int32_t, seqAxis, 1);
    };

    struct ScaleParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(uint32_t, numAxes, 1);
        CPL_PARAM_VALUE(bool, biasTerm, false);
    };

    struct ShuffleParam
    {
        CPL_PARAM_VALUE(int, type, 0);
    };

    struct SliceParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(Index, slicePoint, Index());
    };

    struct SoftmaxParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(bool, log, false);
    };

    struct SoftplusParam
    {
        CPL_PARAM_VALUE(float, beta, 1.0f);
        CPL_PARAM_VALUE(float, threshold, 20.0f);
    };

    struct SqueezeParam
    {
        CPL_PARAM_VALUE(Ints, axes, Ints());
    };

    struct StridedSliceParam
    {
        CPL_PARAM_VALUE(Shape, beginMask, Shape());
        CPL_PARAM_VALUE(Shape, ellipsisMask, Shape());
        CPL_PARAM_VALUE(Shape, endMask, Shape());
        CPL_PARAM_VALUE(Shape, newAxisMask, Shape());
        CPL_PARAM_VALUE(Shape, shrinkAxisMask, Shape());
        CPL_PARAM_VALUE(Shape, beginDims, Shape());
        CPL_PARAM_VALUE(Shape, endDims, Shape());
        CPL_PARAM_VALUE(Shape, strideDims, Shape());
        CPL_PARAM_VALUE(Shape, axes, Shape());
    };

    struct TileParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 1);
        CPL_PARAM_VALUE(uint32_t, tiles, 1);
    };

    struct TopKParam
    {
        CPL_PARAM_VALUE(uint32_t, axis, 0);
        CPL_PARAM_VALUE(TopKMode, mode, TopKModeMax);
        CPL_PARAM_VALUE(TopKSort, sort, TopKSortValue); 
        CPL_PARAM_VALUE(TensorType, indexElementType, TensorType64i);
    };

    struct ConnectionParam
    {
        CPL_PARAM_VALUE(String, src, String());
        CPL_PARAM_VALUE(String, dst, String());
        CPL_PARAM_VALUE(int32_t, port, -1);
        CPL_PARAM_VALUE(int32_t, axis, -1);
    };

    struct TensorIteratorParam
    {
        CPL_PARAM_VECTOR(ConnectionParam, input);
        CPL_PARAM_VECTOR(ConnectionParam, output);
        CPL_PARAM_VECTOR(ConnectionParam, back);
    };

    struct UnaryOperationParam
    {
        CPL_PARAM_VALUE(UnaryOperationType, type, UnaryOperationTypeUnknown);
    };

    struct UnpackParam
    {
        CPL_PARAM_VALUE(int32_t, axis, 0);
        CPL_PARAM_VALUE(Shape, parts, Shape());
    };

    struct UpsampleParam
    {
        CPL_PARAM_VALUE(int32_t, stride, 2);
        CPL_PARAM_VALUE(float, scale, 1.0f);
    };

    struct YoloParam
    {
        CPL_PARAM_VALUE(uint32_t, classes, 20);
        CPL_PARAM_VALUE(uint32_t, num, 1);
        CPL_PARAM_VALUE(uint32_t, total, 1);
        CPL_PARAM_VALUE(uint32_t, max, 30);
        CPL_PARAM_VALUE(float, jitter, 0.2f);
        CPL_PARAM_VALUE(float, ignoreThresh, 0.5f);
        CPL_PARAM_VALUE(float, truthThresh, 1.0f);
        CPL_PARAM_VALUE(Index, mask, Index());
        CPL_PARAM_VALUE(Floats, anchors, Floats());
    };

    struct LayerParam
    {
        CPL_PARAM_VALUE(String, parent, String());
        CPL_PARAM_VALUE(LayerType, type, LayerTypeUnknown);
        CPL_PARAM_VALUE(String, name, String());
        CPL_PARAM_VALUE(Strings, src, Strings());
        CPL_PARAM_VALUE(Strings, dst, Strings());
        CPL_PARAM_VECTOR(WeightParam, weight);
        CPL_PARAM_VALUE(Strings, origin, Strings());

        CPL_PARAM_STRUCT(BatchNormParam, batchNorm);
        CPL_PARAM_STRUCT(BiasParam, bias);
        CPL_PARAM_STRUCT(BinaryOperationParam, binaryOperation);
        CPL_PARAM_STRUCT(BroadcastParam, broadcast);
        CPL_PARAM_STRUCT(CastParam, cast);
        CPL_PARAM_STRUCT(ConcatParam, concat);
        CPL_PARAM_STRUCT(ConvolutionParam, convolution);
        CPL_PARAM_STRUCT(DetectionOutputParam, detectionOutput);
        CPL_PARAM_STRUCT(EltwiseParam, eltwise);
        CPL_PARAM_STRUCT(EluParam, elu);
        CPL_PARAM_STRUCT(ExpandDimsParam, expandDims);
        CPL_PARAM_STRUCT(FillParam, fill);
        CPL_PARAM_STRUCT(FlattenParam, flatten);
        CPL_PARAM_STRUCT(FusedParam, fused);
        CPL_PARAM_STRUCT(GatherParam, gather);
        CPL_PARAM_STRUCT(HardSigmoidParam, hardSigmoid);
        CPL_PARAM_STRUCT(HswishParam, hswish);
        CPL_PARAM_STRUCT(InnerProductParam, innerProduct);
        CPL_PARAM_STRUCT(InputParam, input);
        CPL_PARAM_STRUCT(InterpParam, interp);
        CPL_PARAM_STRUCT(Interp2Param, interp2);
        CPL_PARAM_STRUCT(LogParam, log);
        CPL_PARAM_STRUCT(LrnParam, lrn);
        CPL_PARAM_STRUCT(MergedConvolutionParam, mergedConvolution);
        CPL_PARAM_STRUCT(MetaParam, meta);
        CPL_PARAM_STRUCT(NonMaxSuppressionParam, nonMaxSuppression);
        CPL_PARAM_STRUCT(NormalizeParam, normalize);
        CPL_PARAM_STRUCT(PermuteParam, permute);
        CPL_PARAM_STRUCT(PoolingParam, pooling);
        CPL_PARAM_STRUCT(PowerParam, power);
        CPL_PARAM_STRUCT(PreluParam, prelu);
        CPL_PARAM_STRUCT(PriorBoxParam, priorBox);
        CPL_PARAM_STRUCT(PriorBoxClusteredParam, priorBoxClustered);
        CPL_PARAM_STRUCT(ReductionParam, reduction);
        CPL_PARAM_STRUCT(RegionParam, region);
        CPL_PARAM_STRUCT(ReluParam, relu);
        CPL_PARAM_STRUCT(ReorgParam, reorg);
        CPL_PARAM_STRUCT(ReshapeParam, reshape);
        CPL_PARAM_STRUCT(RestrictRangeParam, restrictRange);
        CPL_PARAM_STRUCT(ReverseSequenceParam, reverseSequence);
        CPL_PARAM_STRUCT(ScaleParam, scale);
        CPL_PARAM_STRUCT(ShuffleParam, shuffle);
        CPL_PARAM_STRUCT(SliceParam, slice);
        CPL_PARAM_STRUCT(SoftmaxParam, softmax);
        CPL_PARAM_STRUCT(SoftplusParam, softplus);
        CPL_PARAM_STRUCT(SqueezeParam, squeeze);
        CPL_PARAM_STRUCT(StridedSliceParam, stridedSlice);
        CPL_PARAM_STRUCT(TileParam, tile);
        CPL_PARAM_STRUCT(TensorIteratorParam, tensorIterator);
        CPL_PARAM_STRUCT(TopKParam, topK);
        CPL_PARAM_STRUCT(UnaryOperationParam, unaryOperation);
        CPL_PARAM_STRUCT(UnpackParam, unpack);
        CPL_PARAM_STRUCT(UpsampleParam, upsample);
        CPL_PARAM_STRUCT(YoloParam, yolo);

        CPL_PARAM_VALUE(Strings, debug, Strings());
    };

    struct StatisticParam
    {
        CPL_PARAM_VALUE(String, name, String());
        CPL_PARAM_VALUE(Floats, min, Floats());
        CPL_PARAM_VALUE(Floats, max, Floats());
    };

    struct QuantizationParam
    {
        CPL_PARAM_VALUE(QuantizationMethod, method, QuantizationMethodUnknown);
        CPL_PARAM_VECTOR(StatisticParam, statistics);
    };

    struct InfoParam
    {
        CPL_PARAM_VALUE(int32_t, version, 0);
        CPL_PARAM_VALUE(String, name, String());
        CPL_PARAM_VALUE(String, from, String());
        CPL_PARAM_VALUE(String, when, String());
        CPL_PARAM_VALUE(String, synet, String());
    };

    struct NetworkParam
    {
        CPL_PARAM_STRUCT(InfoParam, info);
        CPL_PARAM_VALUE(Strings, dst, Strings());
        CPL_PARAM_VECTOR(LayerParam, layers);
        CPL_PARAM_STRUCT(QuantizationParam, quantization);
    };

    CPL_PARAM_HOLDER(NetworkParamHolder, NetworkParam, network);
}