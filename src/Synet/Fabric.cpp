/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/Legacy/BroadcastLayer.h"
#include "Synet/Layers/Legacy/FusedLayer.h"
#include "Synet/Layers/Legacy/LrnLayer.h"
#include "Synet/Layers/Legacy/RegionLayer.h"
#include "Synet/Layers/Legacy/ReorgLayer.h"
#include "Synet/Layers/Legacy/SwitchLayer.h"
#include "Synet/Layers/Legacy/UpsampleLayer.h"

#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Layers/AddLayer.h"
#include "Synet/Layers/ArgMaxLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Layers/BinaryOperationLayer.h"
#include "Synet/Layers/CastLayer.h"
#include "Synet/Layers/CompareLayer.h"
#include "Synet/Layers/ConcatLayer.h"
#include "Synet/Layers/ConstLayer.h"
#include "Synet/Layers/ConstantOfShapeLayer.h"
#include "Synet/Layers/Convolution32fLayer.h"
#include "Synet/Layers/Convolution16bLayer.h"
#include "Synet/Layers/Convolution8iLayer.h"
#include "Synet/Layers/CtcGreedyDecoderLayer.h"
#include "Synet/Layers/DeconvolutionLayer32f.h"
#include "Synet/Layers/DetectionOutputLayer.h"
#include "Synet/Layers/EltwiseLayer.h"
#include "Synet/Layers/ExpandDimsLayer.h"
#include "Synet/Layers/FlattenLayer.h"
#include "Synet/Layers/GatherLayer.h"
#include "Synet/Layers/GridSampleLayer.h"
#include "Synet/Layers/InnerProduct32fLayer.h"
#include "Synet/Layers/InnerProduct16bLayer.h"
#include "Synet/Layers/InnerProduct8iLayer.h"
#include "Synet/Layers/InputLayer.h"
#include "Synet/Layers/InterpLayer.h"
#include "Synet/Layers/LstmLayer.h"
#include "Synet/Layers/MergedConvolution32fLayer.h"
#include "Synet/Layers/MergedConvolution16bLayer.h"
#include "Synet/Layers/MergedConvolution8iLayer.h"
#include "Synet/Layers/MetaLayer.h"
#include "Synet/Layers/MulLayer.h"
#include "Synet/Layers/NonZeroLayer.h"
#include "Synet/Layers/NormalizeLayer.h"
#include "Synet/Layers/PadLayer.h"
#include "Synet/Layers/PermuteLayer.h"
#include "Synet/Layers/PoolingLayer.h"
#include "Synet/Layers/PowerLayer.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Layers/PriorBoxLayer.h"
#include "Synet/Layers/PriorBoxClusteredLayer.h"
#include "Synet/Layers/ReductionLayer.h"
#include "Synet/Layers/ReshapeLayer.h"
#include "Synet/Layers/ReverseSequenceLayer.h"
#include "Synet/Layers/RnnGruBdLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/ScaledDotProductAttentionLayer.h"
#include "Synet/Layers/ScatterNdLayer.h"
#include "Synet/Layers/ShuffleLayer.h"
#include "Synet/Layers/SliceLayer.h"
#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Layers/SpaceToDepthLayer.h"
#include "Synet/Layers/SqueezeLayer.h"
#include "Synet/Layers/SqueezeExcitationLayer.h"
#include "Synet/Layers/StridedSliceLayer.h"
#include "Synet/Layers/StubLayer.h"
#include "Synet/Layers/TensorIteratorLayer.h"
#include "Synet/Layers/TileLayer.h"
#include "Synet/Layers/TiledScale2DLayer.h"
#include "Synet/Layers/TopKLayer.h"
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Layers/UnpackLayer.h"
#include "Synet/Layers/WhereLayer.h"
#include "Synet/Layers/YoloLayer.h"
#include "Synet/Layers/YoloV7Layer.h"

#include "Synet/Converters/SynetUtils.h"
#include "Synet/Fabric.h"

namespace Synet
{
    inline bool Use8i(const MergedConvolutionParam& param)
    {
        if (param.conv().size() == 3)
            return param.conv()[0].quantizationLevel() == TensorType8i && param.conv()[2].quantizationLevel() == TensorType8i;
        else
            return param.conv()[0].quantizationLevel() == TensorType8i || param.conv()[1].quantizationLevel() == TensorType8i;
    }

    inline bool Use16b(const MergedConvolutionParam& param)
    {
        if (param.conv().size() == 3)
            return param.conv()[0].quantizationLevel() == TensorType16b && param.conv()[2].quantizationLevel() == TensorType16b;
        else
            return param.conv()[0].quantizationLevel() == TensorType16b || param.conv()[1].quantizationLevel() == TensorType16b;
    }

    Layer* Fabric::Create(const LayerParam & param, Context* context, QuantizationMethod method)
    {
        switch (param.type())
        {
        case LayerTypeAdd: return new AddLayer(param, context, method);
        case LayerTypeArgMax: return new ArgMaxLayer(param, context);
        case LayerTypeBias: return new BiasLayer(param, context);
        case LayerTypeBinaryOperation: return new BinaryOperationLayer(param, context);
        case LayerTypeBroadcast: return new BroadcastLayer(param, context);
        case LayerTypeCast: return new CastLayer(param, context);
        case LayerTypeCompare: return new CompareLayer(param, context);
        case LayerTypeConcat: return new ConcatLayer(param, context);
        case LayerTypeConst: return new ConstLayer(param, context);
        case LayerTypeConstantOfShape: return new ConstantOfShapeLayer(param, context);
        case LayerTypeConvolution:
            if (param.convolution().quantizationLevel() == TensorType8i)
                return new Convolution8iLayer(param, context, method);
            else if (context->options.BFloat16Enable() && param.lowPrecision().bf16Type() == LowPrecisionTypeActive)
                return new Convolution16bLayer(param, context);
            else
                return new Convolution32fLayer(param, context);
        case LayerTypeCtcGreedyDecoder: return new CtcGreedyDecoderLayer(param, context);
        case LayerTypeDeconvolution: 
            if (context->options.BFloat16Enable() && param.lowPrecision().bf16Type() == LowPrecisionTypeActive)
                return NULL;// new Deconvolution16bLayer(param, context);
            else
                return new DeconvolutionLayer32f(param, context);
        case LayerTypeDetectionOutput: return new DetectionOutputLayer(param, context);
        case LayerTypeEltwise: 
            if(SynetUtils::IsAdd(param))
                return new AddLayer(param, context, method);
            else if (SynetUtils::IsMul(param))
                return new MulLayer(param, context);
            else
                return new EltwiseLayer(param, context);
        case LayerTypeElu: return new EluLayer(param, context);
        case LayerTypeExpandDims: return new ExpandDimsLayer(param, context);
        case LayerTypeFlatten: return new FlattenLayer(param, context);
        case LayerTypeFused: return new FusedLayer(param, context);
        case LayerTypeGather: return new GatherLayer(param, context);
        case LayerTypeGelu: return new GeluLayer(param, context);
        case LayerTypeGridSample: return new GridSampleLayer(param, context);
        case LayerTypeHswish: return new HswishLayer(param, context);
        case LayerTypeHardSigmoid: return new HardSigmoidLayer(param, context);
        case LayerTypeInnerProduct: 
            if (param.innerProduct().quantizationLevel() == TensorType8i)
                return new InnerProduct8iLayer(param, context, method);
            else if (context->options.BFloat16Enable() && param.lowPrecision().bf16Type() == LowPrecisionTypeActive)
                return new InnerProduct16bLayer(param, context);
            else
                return new InnerProduct32fLayer(param, context);
        case LayerTypeInput: return new InputLayer(param, context);
        case LayerTypeInterp: return new InterpLayer(param, context);
        case LayerTypeLrn: return new LrnLayer(param, context);
        case LayerTypeLstm: return new LstmLayer(param, context);
        case LayerTypeMergedConvolution:
            if (Use8i(param.mergedConvolution()))
                return new MergedConvolution8iLayer(param, context, method);
            else if (context->options.BFloat16Enable() && param.lowPrecision().bf16Type() == LowPrecisionTypeActive)
                return new MergedConvolution16bLayer(param, context);
            else
                return new MergedConvolution32fLayer(param, context);
        case LayerTypeMeta: return new MetaLayer(param, context);
        case LayerTypeMish: return new MishLayer(param, context);
        case LayerTypeMul: return new MulLayer(param, context);
        case LayerTypeNonMaxSuppression: return new StubLayer(param, context);
        case LayerTypeNonZero: return new NonZeroLayer(param, context);
        case LayerTypeNormalize: return new NormalizeLayer(param, context);
        case LayerTypePad: return new PadLayer(param, context);
        case LayerTypePermute: return new PermuteLayer(param, context);
        case LayerTypePooling: return new PoolingLayer(param, context);
        case LayerTypePower: return new PowerLayer(param, context);
        case LayerTypePrelu: return new PreluLayer(param, context);
        case LayerTypePriorBox: return new PriorBoxLayer(param, context);
        case LayerTypePriorBoxClustered: return new PriorBoxClusteredLayer(param, context);
        case LayerTypeReduction: return new ReductionLayer(param, context);
        case LayerTypeRegion: return new RegionLayer(param, context);
        case LayerTypeRelu: return new ReluLayer(param, context);
        case LayerTypeReorg: return new ReorgLayer(param, context);
        case LayerTypeReshape: return new ReshapeLayer(param, context);
        case LayerTypeRestrictRange: return new RestrictRangeLayer(param, context);
        case LayerTypeReverseSequence: return new ReverseSequenceLayer(param, context);
        case LayerTypeRnnGruBd: return new RnnGruBdLayer(param, context);
        case LayerTypeScale: return new ScaleLayer(param, context, method);
        case LayerTypeScaledDotProductAttention: return new ScaledDotProductAttentionLayer(param, context);
        case LayerTypeScatterNd: return new ScatterNdLayer(param, context);
        case LayerTypeShuffle: return new ShuffleLayer(param, context);
        case LayerTypeSigmoid: return new SigmoidLayer(param, context);
        case LayerTypeSlice: return new SliceLayer(param, context);
        case LayerTypeSoftmax: return new SoftmaxLayer(param, context);
        case LayerTypeSoftplus: return new SoftplusLayer(param, context);
        case LayerTypeSpaceToDepth: return new SpaceToDepthLayer(param, context);
        case LayerTypeSqueeze: return new SqueezeLayer(param, context);
        case LayerTypeSqueezeExcitation: return new SqueezeExcitationLayer(param, context, method);
        case LayerTypeStridedSlice: return new StridedSliceLayer(param, context);
        case LayerTypeStub: return new StubLayer(param, context);
        case LayerTypeSwish: return new SwishLayer(param, context);
        case LayerTypeSwitch: return new SwitchLayer(param, context);
        case LayerTypeTensorIterator: return new TensorIteratorLayer(param, context);
        case LayerTypeTile: return new TileLayer(param, context);
        case LayerTypeTiledScale2D: return new TiledScale2DLayer(param, context);
        case LayerTypeTopK: return new TopKLayer(param, context);
        case LayerTypeUnaryOperation: return new UnaryOperationLayer(param, context);
        case LayerTypeUnpack: return new UnpackLayer(param, context);
        case LayerTypeUpsample: return new UpsampleLayer(param, context);
        case LayerTypeWhere: return new WhereLayer(param, context);
        case LayerTypeYolo: return new YoloLayer(param, context);
        case LayerTypeYoloV7: return new YoloV7Layer(param, context);
        default:
            return NULL;
        }
    }
}
