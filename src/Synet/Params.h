/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
    template<typename Enum, int Size> SYNET_INLINE Enum StringToEnum(const String & string)
    {
        int type = Size - 1;
        for (; type >= 0; --type)
        {
            if (ValueToString<Enum>((Enum)type) == string)
                return (Enum)type;
        }
        return (Enum)type;
    }

    enum LayerType
    {
        LayerTypeUnknown = -1,
        LayerTypeBatchNorm,
        LayerTypeBias,
        LayerTypeConcat,
        LayerTypeConvolution,
        LayerTypeEltwise,
        LayerTypeDropout,
        LayerTypeInnerProduct,
        LayerTypeInput,
        LayerTypeLrn,
        LayerTypePooling,
        LayerTypeRelu,
        LayerTypeScale,
        LayerTypeSigmoid,
        LayerTypeSoftmax,
        LayerTypeTanh,
        LayerTypeSize
    };

    template<> SYNET_INLINE String ValueToString<LayerType>(const LayerType & value)
    {
        static const char * names[LayerTypeSize] =
        {
            "BatchNorm",
            "Bias",
            "Concat",
            "Convolution",
            "Eltwise",
            "Dropout",
            "InnerProduct",
            "Input",
            "LRN",
            "Pooling",
            "ReLU",
            "Scale",
            "Sigmoid",
            "Softmax",
            "Tanh",
        };
        return (value > LayerTypeUnknown && value < LayerTypeSize) ? names[value] : "";
    }

    template<> SYNET_INLINE void StringToValue<LayerType>(const String & string, LayerType & value)
    {
        value = StringToEnum<LayerType, LayerTypeSize>(string);
    }

    //-------------------------------------------------------------------------

    enum EltwiseOperationType
    {
        EltwiseOperationTypeUnknown = -1,
        EltwiseOperationTypeProduct,
        EltwiseOperationTypeSum,
        EltwiseOperationTypeMax,
        EltwiseOperationTypeSize
    };

    template<> SYNET_INLINE String ValueToString<EltwiseOperationType>(const EltwiseOperationType & value)
    {
        static const char * names[EltwiseOperationTypeSize] =
        {
            "Product",
            "Sum",
            "Max",
        };
        return (value > EltwiseOperationTypeUnknown && value < EltwiseOperationTypeSize) ? names[value] : "";
    }

    template<> SYNET_INLINE void StringToValue<EltwiseOperationType>(const String & string, EltwiseOperationType & value)
    {
        value = StringToEnum<EltwiseOperationType, EltwiseOperationTypeSize>(string);
    }

    //-------------------------------------------------------------------------

    enum NormRegionType
    {
        NormRegionTypeUnknown = -1,
        NormRegionTypeAcrossChannels,
        NormRegionTypeWithinChannel,
        NormRegionTypeSize
    };

    template<> SYNET_INLINE String ValueToString<NormRegionType>(const NormRegionType & value)
    {
        static const char * names[NormRegionTypeSize] =
        {
            "AcrossChannels",
            "WithinChannel",
        };
        return (value > NormRegionTypeUnknown && value < NormRegionTypeSize) ? names[value] : "";
    }

    template<> SYNET_INLINE void StringToValue<NormRegionType>(const String & string, NormRegionType & value)
    {
        value = StringToEnum<NormRegionType, NormRegionTypeSize>(string);
    }

    //-------------------------------------------------------------------------
    
    enum PoolingMethodType
    {
        PoolingMethodTypeUnknown = -1,
        PoolingMethodTypeMax,
        PoolingMethodTypeAverage,
        PoolingMethodTypeStochastic,
        PoolingMethodTypeSize
    };

    template<> SYNET_INLINE String ValueToString<PoolingMethodType>(const PoolingMethodType & value)
    {
        static const char * names[LayerTypeSize] =
        {
            "Max",
            "Average",
            "Stochastic",
        };
        return (value > PoolingMethodTypeUnknown && value < PoolingMethodTypeSize) ? names[value] : "";
    }

    template<> SYNET_INLINE void StringToValue<PoolingMethodType>(const String & string, PoolingMethodType & value)
    {
        value = StringToEnum<PoolingMethodType, PoolingMethodTypeSize>(string);
    }

    //-------------------------------------------------------------------------

    struct ShapeParam
    {
        SYNET_PARAM_VALUE(Shape, dim, Shape());
    };

    struct BatchNormParam
    {
        SYNET_PARAM_VALUE(bool, useGlobalStats, true);
        SYNET_PARAM_VALUE(float, movingAverageFraction, 0.999f);
        SYNET_PARAM_VALUE(float, eps, 0.00001f);
    };

    struct BiasParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, numAxes, 1);
    };

    struct ConcatParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct ConvolutionParam
    {
        SYNET_PARAM_VALUE(uint32_t, outputNum, 0);
        SYNET_PARAM_VALUE(bool, biasTerm, true);
        SYNET_PARAM_VALUE(Shape, kernel, Shape());
        SYNET_PARAM_VALUE(Shape, pad, Shape());
        SYNET_PARAM_VALUE(Shape, stride, Shape());
        SYNET_PARAM_VALUE(Shape, dilation, Shape());
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, group, 1);
    };

    struct EltwiseParam
    {
        SYNET_PARAM_VALUE(EltwiseOperationType, operation, EltwiseOperationTypeSum);
        SYNET_PARAM_VALUE(Floats, coefficients, Floats());
    };

    struct InnerProductParam
    {
        SYNET_PARAM_VALUE(uint32_t, outputNum, 0);
        SYNET_PARAM_VALUE(bool, biasTerm, true);
        SYNET_PARAM_VALUE(bool, transpose, false);
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct InputParam
    {
        SYNET_PARAM_VECTOR(ShapeParam, shape);
    };

    struct LrnParam
    {
        SYNET_PARAM_VALUE(uint32_t, localSize, 5);
        SYNET_PARAM_VALUE(float, alpha, 1.0f);
        SYNET_PARAM_VALUE(float, beta, 0.75f);
        SYNET_PARAM_VALUE(NormRegionType, normRegion, NormRegionTypeAcrossChannels);
        SYNET_PARAM_VALUE(float, k, 1.0f);
    };

    struct PoolingParam
    {
        SYNET_PARAM_VALUE(PoolingMethodType, method, PoolingMethodTypeUnknown);
        SYNET_PARAM_VALUE(Shape, kernel, Shape());
        SYNET_PARAM_VALUE(Shape, pad, Shape());
        SYNET_PARAM_VALUE(Shape, stride, Shape());
        SYNET_PARAM_VALUE(bool, globalPooling, false);
    };

    struct ReluParam
    {
        SYNET_PARAM_VALUE(float, negativeSlope, 0.0f);
    };

    struct ScaleParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
        SYNET_PARAM_VALUE(uint32_t, numAxes, 1);
        SYNET_PARAM_VALUE(bool, biasTerm, false);
    };

    struct SoftmaxParam
    {
        SYNET_PARAM_VALUE(uint32_t, axis, 1);
    };

    struct LayerParam
    {
        SYNET_PARAM_VALUE(LayerType, type, LayerTypeUnknown);
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VALUE(Strings, src, Strings());
        SYNET_PARAM_VALUE(Strings, dst, Strings());
        SYNET_PARAM_VECTOR(ShapeParam, data);

        SYNET_PARAM_STRUCT(BatchNormParam, batchNorm);
        SYNET_PARAM_STRUCT(BiasParam, bias);
        SYNET_PARAM_STRUCT(ConcatParam, concat);
        SYNET_PARAM_STRUCT(ConvolutionParam, convolution);
        SYNET_PARAM_STRUCT(EltwiseParam, eltwise);
        SYNET_PARAM_STRUCT(InnerProductParam, innerProduct);
        SYNET_PARAM_STRUCT(InputParam, input);
        SYNET_PARAM_STRUCT(LrnParam, lrn);
        SYNET_PARAM_STRUCT(PoolingParam, pooling);
        SYNET_PARAM_STRUCT(ReluParam, relu);
        SYNET_PARAM_STRUCT(ScaleParam, scale);
        SYNET_PARAM_STRUCT(SoftmaxParam, softmax);
    };

    struct NetworkParam
    {
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VECTOR(LayerParam, layers);
    };

    SYNET_PARAM_HOLDER(NetworkParamHolder, NetworkParam, network);
}