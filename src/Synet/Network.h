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

#include "Synet/AbsLayer.h"
#include "Synet/BatchNormLayer.h"
#include "Synet/BiasLayer.h"
#include "Synet/ConcatLayer.h"
#include "Synet/ConvolutionLayer.h"
#include "Synet/DetectionOutputLayer.h"
#include "Synet/EltwiseLayer.h"
#include "Synet/ExpandDimsLayer.h"
#include "Synet/FillLayer.h"
#include "Synet/FlattenLayer.h"
#include "Synet/InnerProductLayer.h"
#include "Synet/InputLayer.h"
#include "Synet/InterpLayer.h"
#include "Synet/LogLayer.h"
#include "Synet/LrnLayer.h"
#include "Synet/MetaLayer.h"
#include "Synet/NormalizeLayer.h"
#include "Synet/PermuteLayer.h"
#include "Synet/PoolingLayer.h"
#include "Synet/PriorBoxLayer.h"
#include "Synet/RegionLayer.h"
#include "Synet/ReluLayer.h"
#include "Synet/ReorgLayer.h"
#include "Synet/ReshapeLayer.h"
#include "Synet/ScaleLayer.h"
#include "Synet/SigmoidLayer.h"
#include "Synet/SliceLayer.h"
#include "Synet/SoftmaxLayer.h"
#include "Synet/SqueezeLayer.h"
#include "Synet/StubLayer.h"
#include "Synet/TanhLayer.h"
#include "Synet/UpsampleLayer.h"
#include "Synet/YoloLayer.h"

namespace Synet
{
    template <class T> class Network
    {
    public:
        typedef T Type;
        typedef Synet::Tensor<T> Tensor;
        typedef std::vector<Tensor*> TensorPtrs;
        typedef Synet::Layer<T> Layer;
        typedef Layer * LayerPtr;

        Network()
            : _empty(true)
        {
        }

        bool Empty() const 
        { 
            return _empty; 
        }

        const NetworkParam & Param() const 
        { 
            return _param(); 
        }

        bool Load(const String & param, const String & weight)
        {
            if (!_param.Load(param))
                return false;

            _layers.clear();
            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                LayerSharedPtr layer(Create(_param().layers()[i]));
                if (layer)
                    _layers.push_back(layer);
            }

            std::ifstream ifs(weight.c_str(), std::ifstream::binary);
            if (!ifs.is_open())
                return false;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                if (!_layers[i]->Load(ifs))
                {
                    ifs.close();
                    return false;
                }
            }
            ifs.close();

            return Init();
        }

        TensorPtrs & Src() 
        { 
            return _src; 
        }

        const TensorPtrs & Dst() const 
        { 
            return _dst; 
        }

        LayerPtr Back() const
        {
            return _back;
        }

        void Reshape()
        {
            for (size_t i = 0; i < _stages.size(); ++i)
                _stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst);
        }

        void Forward()
        {
            SYNET_PERF_FUNC();
            bool ftz = GetFlushToZero();
            SetFlushToZero(true);
            for (size_t i = 0; i < _stages.size(); ++i)
                _stages[i].layer->Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
            SetFlushToZero(ftz);
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        void DebugPrint(std::ostream & os, bool weight)
        {
            for (size_t i = 0; i < _stages.size(); ++i)
            {
                _stages[i].layer->Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
                os << "Layer: " << _stages[i].layer->Param().name() << " : ";
                os << ValueToString(_stages[i].layer->Param().type()) << " ( ";
                for(size_t j = 0; j < _stages[i].layer->Param().src().size(); ++j)
                    os << _stages[i].layer->Param().src()[j] << " ";
                os << ")." << std::endl;
                if (weight)
                {
                    for (size_t j = 0; j < _stages[i].layer->Weight().size(); ++j)
                        _stages[i].layer->Weight()[j].DebugPrint(os, String("weight[") + ValueToString(j) + "]");
                }
                for (size_t j = 0; j < _stages[i].dst.size(); ++j)
                    _stages[i].dst[j]->DebugPrint(os, String("dst[") + ValueToString(j) + "]");
            }
        }
#endif

    private:
        typedef std::vector<LayerPtr> LayerPtrs;
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        typedef std::vector<Tensor> Tensors;
        typedef std::shared_ptr<Tensor> TensorSharedPtr;
        typedef std::vector<TensorSharedPtr> TensorSharedPtrs;

        typedef std::map<String, size_t> NameIndexMap;
        typedef std::set<String> NameSet;

        struct Stage
        {
            LayerPtr layer;
            TensorPtrs src;
            TensorPtrs buf;
            TensorPtrs dst;
        };
        typedef std::vector<Stage> Stages;

        bool _empty;
        NetworkParamHolder _param;
        LayerSharedPtrs _layers;
        TensorSharedPtrs _tensors;

        Stages _stages;
        TensorPtrs _src, _dst;
        LayerPtr _back;

        bool Init()
        {
            const size_t bufs = 1;
            TensorPtrs buf;
            for (size_t i = 0; i < bufs; ++i)
            {
                TensorSharedPtr tensor(new Tensor());
                _tensors.push_back(tensor);
                buf.push_back(tensor.get());
            }

            NameIndexMap index;
            NameSet available;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                Stage stage;
                stage.layer = _layers[i].get();
                const LayerParam & param = stage.layer->Param();
                for (size_t j = 0; j < param.src().size(); ++j)
                {
                    const String & name = param.src()[j];
                    if (index.find(name) != index.end())
                    {
                        stage.src.push_back(_tensors[index[name]].get());
                        if (available.find(name) != available.end())
                            available.erase(name);
                    }
                    else
                        assert(0);
                }
                for (size_t j = 0; j < param.dst().size(); ++j)
                {
                    const String & name = param.dst()[j];
                    if (j < param.src().size() && name == param.src()[j])
                    {
                        stage.dst.push_back(_tensors[index[name]].get());
                    }
                    else  if (index.find(name) != index.end())
                    {
                        assert(0);
                    }
                    else
                    {
                        TensorSharedPtr tensor(new Tensor());
                        index[name] = _tensors.size();
                        _tensors.push_back(tensor);
                        stage.dst.push_back(tensor.get());
                    }
                    available.insert(name);
                    if (param.type() == LayerTypeInput)
                        _src.push_back(_tensors.back().get());
                }
                stage.buf = buf;
                stage.layer->Setup(stage.src, stage.buf, stage.dst);
                stage.layer->Reshape(stage.src, stage.buf, stage.dst);
                _stages.push_back(stage);
            }
            for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
            {
                if(InsertDst(*it))
                    _dst.push_back(_tensors[index[*it]].get());
            }
            _back = _stages.empty() ? NULL : _stages.back().layer;
            _empty = false;
            return true;
        }

        bool InsertDst(const String & name)
        {
            if (_param().dst().empty())
                return true;
            for (size_t i = 0; i < _param().dst().size(); ++i)
            {
                if (_param().dst()[i] == name)
                    return true;
            }
            return false;
        }

        static LayerPtr Create(const LayerParam & param)
        {
            switch (param.type())
            {
            case LayerTypeAbs: return new AbsLayer<T>(param);
            case LayerTypeBatchNorm: return new BatchNormLayer<T>(param);
            case LayerTypeBias: return new BiasLayer<T>(param);
            case LayerTypeConcat: return new ConcatLayer<T>(param);
            case LayerTypeConvolution: return new ConvolutionLayer<T>(param);
            case LayerTypeDetectionOutput: return new DetectionOutputLayer<T>(param);
            case LayerTypeDropout: return new StubLayer<T>(param);
            case LayerTypeEltwise: return new EltwiseLayer<T>(param);
            case LayerTypeExpandDims: return new ExpandDimsLayer<T>(param);
            case LayerTypeFill: return new FillLayer<T>(param);
            case LayerTypeFlatten: return new FlattenLayer<T>(param);
            case LayerTypeInnerProduct: return new InnerProductLayer<T>(param);
            case LayerTypeInput: return new InputLayer<T>(param);
            case LayerTypeInterp: return new InterpLayer<T>(param);
            case LayerTypeLog: return new LogLayer<T>(param);
            case LayerTypeLrn: return new LrnLayer<T>(param);
            case LayerTypeMeta: return new MetaLayer<T>(param);
            case LayerTypeNormalize: return new NormalizeLayer<T>(param);
            case LayerTypePermute: return new PermuteLayer<T>(param);
            case LayerTypePooling: return new PoolingLayer<T>(param);
            case LayerTypePriorBox: return new PriorBoxLayer<T>(param);
            case LayerTypeRegion: return new RegionLayer<T>(param);
            case LayerTypeRelu: return new ReluLayer<T>(param);
            case LayerTypeReorg: return new ReorgLayer<T>(param);
            case LayerTypeReshape: return new ReshapeLayer<T>(param);
            case LayerTypeScale: return new ScaleLayer<T>(param);
            case LayerTypeSigmoid: return new SigmoidLayer<T>(param);
            case LayerTypeSlice: return new SliceLayer<T>(param);
            case LayerTypeSoftmax: return new SoftmaxLayer<T>(param);
            case LayerTypeSqueeze: return new SqueezeLayer<T>(param);
            case LayerTypeStub: return new StubLayer<T>(param);
            case LayerTypeTanh: return new TanhLayer<T>(param);
            case LayerTypeUpsample: return new UpsampleLayer<T>(param);
            case LayerTypeYolo: return new YoloLayer<T>(param);
            default:
                return NULL;
            }
        }
        
        friend class TensorflowToSynet;
    };
}