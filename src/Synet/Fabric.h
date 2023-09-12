/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Layers/AddLayer.h"
#include "Synet/Layers/ArgMaxLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Layers/BinaryOperationLayer.h"
#include "Synet/Layers/BroadcastLayer.h"
#include "Synet/Layers/CastLayer.h"
#include "Synet/Layers/CompareLayer.h"
#include "Synet/Layers/ConcatLayer.h"
#include "Synet/Layers/ConstLayer.h"
#include "Synet/Layers/ConstantOfShapeLayer.h"
#include "Synet/Layers/Convolution32fLayer.h"
#include "Synet/Layers/Convolution8iLayer.h"
#include "Synet/Layers/CtcGreedyDecoderLayer.h"
#include "Synet/Layers/DeconvolutionLayer.h"
#include "Synet/Layers/DetectionOutputLayer.h"
#include "Synet/Layers/EltwiseLayer.h"
#include "Synet/Layers/ExpandDimsLayer.h"
#include "Synet/Layers/FlattenLayer.h"
#include "Synet/Layers/FusedLayer.h"
#include "Synet/Layers/GatherLayer.h"
#include "Synet/Layers/GridSampleLayer.h"
#include "Synet/Layers/InnerProductLayer.h"
#include "Synet/Layers/InputLayer.h"
#include "Synet/Layers/InterpLayer.h"
#include "Synet/Layers/LrnLayer.h"
#include "Synet/Layers/LstmLayer.h"
#include "Synet/Layers/MergedConvolution32fLayer.h"
#include "Synet/Layers/MergedConvolution8iLayer.h"
#include "Synet/Layers/MetaLayer.h"
#include "Synet/Layers/NonMaxSuppressionLayer.h"
#include "Synet/Layers/NormalizeLayer.h"
#include "Synet/Layers/PadLayer.h"
#include "Synet/Layers/PermuteLayer.h"
#include "Synet/Layers/PoolingLayer.h"
#include "Synet/Layers/PowerLayer.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Layers/PriorBoxLayer.h"
#include "Synet/Layers/PriorBoxClusteredLayer.h"
#include "Synet/Layers/ReductionLayer.h"
#include "Synet/Layers/RegionLayer.h"
#include "Synet/Layers/ReorgLayer.h"
#include "Synet/Layers/ReshapeLayer.h"
#include "Synet/Layers/ReverseSequenceLayer.h"
#include "Synet/Layers/RnnGruBdLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/ScatterNdLayer.h"
#include "Synet/Layers/ShortcutLayer.h"
#include "Synet/Layers/ShuffleLayer.h"
#include "Synet/Layers/SliceLayer.h"
#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Layers/SpaceToDepthLayer.h"
#include "Synet/Layers/SqueezeLayer.h"
#include "Synet/Layers/SqueezeExcitationLayer.h"
#include "Synet/Layers/StridedSliceLayer.h"
#include "Synet/Layers/StubLayer.h"
#include "Synet/Layers/SwitchLayer.h"
#include "Synet/Layers/TensorIteratorLayer.h"
#include "Synet/Layers/TileLayer.h"
#include "Synet/Layers/TopKLayer.h"
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Layers/UpsampleLayer.h"
#include "Synet/Layers/UnpackLayer.h"
#include "Synet/Layers/WhereLayer.h"
#include "Synet/Layers/YoloLayer.h"
#include "Synet/Layers/YoloV7Layer.h"

namespace Synet
{
    template <class T> class Fabric
    {
    public:
        typedef T Type;
        typedef Synet::Layer<T> Layer;
        typedef Layer * LayerPtr;

        static LayerPtr Create(const LayerParam & param, Context* context, QuantizationMethod method)
        {
            switch (param.type())
            {
            case LayerTypeAdd: return new AddLayer<T>(param, context, method);
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
                    return new Convolution8iLayer<T>(param, context, method);
                else
                    return new Convolution32fLayer<T>(param, context);
            case LayerTypeCtcGreedyDecoder: return new CtcGreedyDecoderLayer(param, context);
            case LayerTypeDeconvolution: return new DeconvolutionLayer(param, context);
            case LayerTypeDetectionOutput: return new DetectionOutputLayer<T>(param, context);
            case LayerTypeEltwise: return new EltwiseLayer<T>(param, context);
            case LayerTypeElu: return new EluLayer(param, context);
            case LayerTypeExpandDims: return new ExpandDimsLayer<T>(param, context);
            case LayerTypeFlatten: return new FlattenLayer(param, context);
            case LayerTypeFused: return new FusedLayer<T>(param, context);
            case LayerTypeGather: return new GatherLayer(param, context);
            case LayerTypeGelu: return new GeluLayer(param, context);
            case LayerTypeGridSample: return new GridSampleLayer(param, context);
            case LayerTypeHswish: return new HswishLayer(param, context);
            case LayerTypeHardSigmoid: return new HardSigmoidLayer(param, context);
            case LayerTypeInnerProduct: return new InnerProductLayer<T>(param, context, method);
            case LayerTypeInput: return new InputLayer<T>(param, context);
            case LayerTypeInterp: return new InterpLayer<T>(param, context);
            case LayerTypeLrn: return new LrnLayer<T>(param, context);
            case LayerTypeLstm: return new LstmLayer<T>(param, context);
            case LayerTypeMergedConvolution:
                if (Use8i(param.mergedConvolution()))
                    return new MergedConvolution8iLayer<T>(param, context, method);
                else
                    return new MergedConvolution32fLayer<T>(param, context);
            case LayerTypeMeta: return new MetaLayer(param, context);
            case LayerTypeMish: return new MishLayer(param, context);
            case LayerTypeNonMaxSuppression: return new NonMaxSuppressionLayer<T>(param, context);
            case LayerTypeNormalize: return new NormalizeLayer<T>(param, context);
            case LayerTypePad: return new PadLayer(param, context);
            case LayerTypePermute: return new PermuteLayer<T>(param, context);
            case LayerTypePooling: return new PoolingLayer<T>(param, context);
            case LayerTypePower: return new PowerLayer(param, context);
            case LayerTypePrelu: return new PreluLayer<T>(param, context);
            case LayerTypePriorBox: return new PriorBoxLayer(param, context);
            case LayerTypePriorBoxClustered: return new PriorBoxClusteredLayer(param, context);
            case LayerTypeReduction: return new ReductionLayer(param, context);
            case LayerTypeRegion: return new RegionLayer<T>(param, context);
            case LayerTypeRelu: return new ReluLayer(param, context);
            case LayerTypeReorg: return new ReorgLayer<T>(param, context);
            case LayerTypeReshape: return new ReshapeLayer<T>(param, context);
            case LayerTypeRestrictRange: return new RestrictRangeLayer(param, context);
            case LayerTypeReverseSequence: return new ReverseSequenceLayer<T>(param, context);
            case LayerTypeRnnGruBd: return new RnnGruBdLayer<T>(param, context);
            case LayerTypeScale: return new ScaleLayer(param, context, method);
            case LayerTypeScatterNd: return new ScatterNdLayer<T>(param, context);
            case LayerTypeShortcut: return new ShortcutLayer<T>(param, context);
            case LayerTypeShuffle: return new ShuffleLayer<T>(param, context);
            case LayerTypeSigmoid: return new SigmoidLayer(param, context);
            case LayerTypeSlice: return new SliceLayer<T>(param, context);
            case LayerTypeSoftmax: return new SoftmaxLayer(param, context);
            case LayerTypeSoftplus: return new SoftplusLayer(param, context);
            case LayerTypeSpaceToDepth: return new SpaceToDepthLayer<T>(param, context);
            case LayerTypeSqueeze: return new SqueezeLayer<T>(param, context);
            case LayerTypeSqueezeExcitation: return new SqueezeExcitationLayer<T>(param, context, method);
            case LayerTypeStridedSlice: return new StridedSliceLayer<T>(param, context);
            case LayerTypeStub: return new StubLayer(param, context);
            case LayerTypeSwish: return new SwishLayer(param, context);
            case LayerTypeSwitch: return new SwitchLayer<T>(param, context);
            case LayerTypeTensorIterator: return new TensorIteratorLayer<T>(param, context);
            case LayerTypeTile: return new TileLayer<T>(param, context);
            case LayerTypeTopK: return new TopKLayer(param, context);
            case LayerTypeUnaryOperation: return new UnaryOperationLayer(param, context);
            case LayerTypeUnpack: return new UnpackLayer(param, context);
            case LayerTypeUpsample: return new UpsampleLayer<T>(param, context);
            case LayerTypeWhere: return new WhereLayer(param, context);
            case LayerTypeYolo: return new YoloLayer<T>(param, context);
            case LayerTypeYoloV7: return new YoloV7Layer<T>(param, context);
            default:
                return NULL;
            }
        }

    private:
        static inline bool Use8i(const MergedConvolutionParam& param)
        {
            if (param.conv().size() == 3)
                return param.conv()[0].quantizationLevel() == TensorType8i && param.conv()[2].quantizationLevel() == TensorType8i;
            else
                return param.conv()[0].quantizationLevel() == TensorType8i || param.conv()[1].quantizationLevel() == TensorType8i;
        }
    };
}
