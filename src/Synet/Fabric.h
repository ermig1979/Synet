/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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

#include "Synet/Layers/AddLayer.h"
#include "Synet/Layers/BatchNormLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Layers/BinaryOperationLayer.h"
#include "Synet/Layers/BroadcastLayer.h"
#include "Synet/Layers/CastLayer.h"
#include "Synet/Layers/ConcatLayer.h"
#include "Synet/Layers/ConstLayer.h"
#include "Synet/Layers/Convolution32fLayer.h"
#include "Synet/Layers/Convolution8iLayer.h"
#include "Synet/Layers/CtcGreedyDecoderLayer.h"
#include "Synet/Layers/DeconvolutionLayer.h"
#include "Synet/Layers/DetectionOutputLayer.h"
#include "Synet/Layers/EltwiseLayer.h"
#include "Synet/Layers/ExpandDimsLayer.h"
#include "Synet/Layers/FillLayer.h"
#include "Synet/Layers/FlattenLayer.h"
#include "Synet/Layers/FusedLayer.h"
#include "Synet/Layers/GatherLayer.h"
#include "Synet/Layers/HswishLayer.h"
#include "Synet/Layers/InnerProductLayer.h"
#include "Synet/Layers/InputLayer.h"
#include "Synet/Layers/InterpLayer.h"
#include "Synet/Layers/Interp2Layer.h"
#include "Synet/Layers/LogLayer.h"
#include "Synet/Layers/LrnLayer.h"
#include "Synet/Layers/MergedConvolution32fLayer.h"
#include "Synet/Layers/MergedConvolution8iLayer.h"
#include "Synet/Layers/MetaLayer.h"
#include "Synet/Layers/MishLayer.h"
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
#include "Synet/Layers/ReluLayer.h"
#include "Synet/Layers/EluLayer.h"
#include "Synet/Layers/ReorgLayer.h"
#include "Synet/Layers/ReshapeLayer.h"
#include "Synet/Layers/RestrictRangeLayer.h"
#include "Synet/Layers/ReverseSequenceLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/ShortcutLayer.h"
#include "Synet/Layers/ShuffleLayer.h"
#include "Synet/Layers/SigmoidLayer.h"
#include "Synet/Layers/SliceLayer.h"
#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Layers/SoftplusLayer.h"
#include "Synet/Layers/SqueezeLayer.h"
#include "Synet/Layers/SqueezeExcitationLayer.h"
#include "Synet/Layers/StridedSliceLayer.h"
#include "Synet/Layers/StubLayer.h"
#include "Synet/Layers/SwitchLayer.h"
#include "Synet/Layers/TileLayer.h"
#include "Synet/Layers/TensorIteratorLayer.h"
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Layers/UpsampleLayer.h"
#include "Synet/Layers/UnpackLayer.h"
#include "Synet/Layers/YoloLayer.h"

namespace Synet
{
    template <class T> class Fabric
    {
    public:
        typedef T Type;
        typedef Synet::Layer<T> Layer;
        typedef Layer * LayerPtr;

        static LayerPtr Create(const LayerParam & param, QuantizationMethod method)
        {
            switch (param.type())
            {
            case LayerTypeAdd: return new AddLayer<T>(param, method);
            case LayerTypeBatchNorm: return new BatchNormLayer<T>(param);
            case LayerTypeBias: return new BiasLayer<T>(param);
            case LayerTypeBinaryOperation: return new BinaryOperationLayer<T>(param);
            case LayerTypeBroadcast: return new BroadcastLayer<T>(param);
            case LayerTypeCast: return new CastLayer<T>(param);
            case LayerTypeConcat: return new ConcatLayer<T>(param);
            case LayerTypeConst: return new ConstLayer<T>(param);
            case LayerTypeConvolution: 
                if (param.convolution().quantizationLevel() == TensorType8i)
                    return new Convolution8iLayer<T>(param, method);
                else
                    return new Convolution32fLayer<T>(param);
            case LayerTypeCtcGreedyDecoder: return new CtcGreedyDecoderLayer<T>(param);
            case LayerTypeDeconvolution: return new DeconvolutionLayer<T>(param);
            case LayerTypeDetectionOutput: return new DetectionOutputLayer<T>(param);
            case LayerTypeDropout: return new StubLayer<T>(param);
            case LayerTypeEltwise: return new EltwiseLayer<T>(param);
            case LayerTypeExpandDims: return new ExpandDimsLayer<T>(param);
            case LayerTypeFill: return new FillLayer<T>(param);
            case LayerTypeFlatten: return new FlattenLayer<T>(param);
            case LayerTypeFused: return new FusedLayer<T>(param);
            case LayerTypeGather: return new GatherLayer<T>(param);
            case LayerTypeHswish: return new HswishLayer<T>(param);
            case LayerTypeInnerProduct: return new InnerProductLayer<T>(param, method);
            case LayerTypeInput: return new InputLayer<T>(param);
            case LayerTypeInterp: return new InterpLayer<T>(param);
            case LayerTypeInterp2: return new Interp2Layer<T>(param);
            case LayerTypeLog: return new LogLayer<T>(param);
            case LayerTypeLrn: return new LrnLayer<T>(param);
            case LayerTypeMergedConvolution:
                if (Use8i(param.mergedConvolution()))
                    return new MergedConvolution8iLayer<T>(param, method);
                else
                    return new MergedConvolution32fLayer<T>(param);
            case LayerTypeMeta: return new MetaLayer<T>(param);
            case LayerTypeMish: return new MishLayer<T>(param);
            case LayerTypeNormalize: return new NormalizeLayer<T>(param);
            case LayerTypePad: return new PadLayer<T>(param);
            case LayerTypePermute: return new PermuteLayer<T>(param);
            case LayerTypePooling: return new PoolingLayer<T>(param);
            case LayerTypePower: return new PowerLayer<T>(param);
            case LayerTypePrelu: return new PreluLayer<T>(param);
            case LayerTypePriorBox: return new PriorBoxLayer<T>(param);
            case LayerTypePriorBoxClustered: return new PriorBoxClusteredLayer<T>(param);
            case LayerTypeReduction: return new ReductionLayer<T>(param);
            case LayerTypeRegion: return new RegionLayer<T>(param);
            case LayerTypeRelu: return new ReluLayer<T>(param);
            case LayerTypeElu: return new EluLayer<T>(param);
            case LayerTypeReorg: return new ReorgLayer<T>(param);
            case LayerTypeReshape: return new ReshapeLayer<T>(param);
            case LayerTypeRestrictRange: return new RestrictRangeLayer<T>(param);
            case LayerTypeReverseSequence: return new ReverseSequenceLayer<T>(param);
            case LayerTypeScale: return new ScaleLayer<T>(param, method);
            case LayerTypeShortcut: return new ShortcutLayer<T>(param);
            case LayerTypeShuffle: return new ShuffleLayer<T>(param);
            case LayerTypeSigmoid: return new SigmoidLayer<T>(param);
            case LayerTypeSlice: return new SliceLayer<T>(param);
            case LayerTypeSoftmax: return new SoftmaxLayer<T>(param);
            case LayerTypeSoftplus: return new SoftplusLayer<T>(param);
            case LayerTypeSqueeze: return new SqueezeLayer<T>(param);
            case LayerTypeSqueezeExcitation: return new SqueezeExcitationLayer<T>(param, method);
            case LayerTypeStridedSlice: return new StridedSliceLayer<T>(param);
            case LayerTypeStub: return new StubLayer<T>(param);
            case LayerTypeSwitch: return new SwitchLayer<T>(param);
            case LayerTypeTile: return new TileLayer<T>(param);
            case LayerTypeTensorIterator: return new TensorIteratorLayer<T>(param);
            case LayerTypeUnaryOperation: return new UnaryOperationLayer<T>(param);
            case LayerTypeUnpack: return new UnpackLayer<T>(param);
            case LayerTypeUpsample: return new UpsampleLayer<T>(param);
            case LayerTypeYolo: return new YoloLayer<T>(param);
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
