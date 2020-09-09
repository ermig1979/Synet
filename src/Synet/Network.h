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

#include "Synet/Layers/AddLayer.h"
#include "Synet/Layers/BatchNormLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Layers/BinaryOperationLayer.h"
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
#include "Synet/Layers/MetaLayer.h"
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
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Layers/UpsampleLayer.h"
#include "Synet/Layers/UnpackLayer.h"
#include "Synet/Layers/YoloLayer.h"

#include "Synet/Utils/SetInput.h"
#include "Synet/Utils/Statistics.h"

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
        typedef std::vector<LayerPtr> LayerPtrs;
        typedef Synet::Region<T> Region;
        typedef std::vector<Region> Regions;

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

        void Clear()
        {
            _param() = NetworkParam();
            _layers.clear();
            _tensors.clear();
            _stats.clear();
            _input.clear();
            _stages.clear();
            _stats.clear();
            _src.clear();
            _dst.clear();
            _back.clear();
            _tensorId.clear();
            _layerId.clear();
            _statId.clear();
            _srcIds.clear();
            _dstIds.clear();
            _empty = true;
        }

        bool Load(const String & model, const String & weight)
        {
            Clear();

            if (!_param.Load(model))
            {
                std::cout << "Can't load model file '" << model << "' !" << std::endl;
                return false;
            }

            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                LayerSharedPtr layer(Create(_param().layers()[i], _param().quantization().method()));
                if (layer)
                    _layers.push_back(layer);
            }

            std::ifstream ifs(weight.c_str(), std::ifstream::binary);
            if (!ifs.is_open())
            {
                std::cout << "Can't open weight file '" << weight << "' !" << std::endl;
                return false;
            }
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                if (!_layers[i]->Load(ifs, _layers))
                {
                    std::cout << "Can't load weight from file '" << weight << "' !" << std::endl;
                    ifs.close();
                    return false;
                }
            }
            ifs.close();

            return Init();
        }

        bool Load(const char * modelData, size_t modelSize, const char * weightData, size_t weightSize)
        {
            Clear();

            if (!_param.Load(modelData, modelSize))
                return false;

            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                LayerSharedPtr layer(Create(_param().layers()[i], _param().quantization().method()));
                if (layer)
                    _layers.push_back(layer);
            }

            for (size_t i = 0; i < _layers.size(); ++i)
            {
                if (!_layers[i]->Load(weightData, weightSize, _layers))
                    return false;
            }

            return Init();
        }

        bool Save(const String& model) const
        {
            return _param.Save(model, false);
        }

        TensorPtrs & Src() 
        { 
            return _src; 
        }

        const TensorPtrs & Src() const
        {
            return _src;
        }

        const TensorPtrs & Dst() const 
        { 
            return _dst; 
        }

        LayerPtrs Back() const
        {
            return _back;
        }

        bool Reshape(const Strings & srcNames = Strings(), const Shapes & srcShapes = Shapes(), const Strings & dstNames = Strings())
        {
            if (srcNames.size() != srcShapes.size())
            {
                std::cout << "srcNames.size() != srcShapes.size() !" << std::endl;
                return false;
            }

            for (size_t i = 0; i < _tensors.size(); ++i)
                _tensors[i]->Clear(true);

            if (srcNames.size())
            {
                _src.clear();
                for (size_t i = 0; i < srcNames.size(); ++i)
                {
                    bool found = false;
                    for (size_t j = 0; j < _input.size(); ++j)
                    {
                        const LayerParam & param = _input[j].layer->Param();
                        if (param.name() == srcNames[i])
                        {
                            if (param.type() == LayerTypeInput)
                            {
                                _input[j].dst[0]->Reshape(srcShapes[i], Type(0), param.input().shape()[0].format(), param.name());
                                _src.push_back(_input[j].dst[0]);
                            }
                            else if (param.type() == LayerTypeMeta && (param.meta().type() == MetaTypeInput || param.meta().type() == MetaTypeInputWithDefault))
                            {
                                Synet::Tensor<int32_t> & i32 = _input[j].dst[0]->As32i();
                                i32.Reshape({ srcShapes[j].size()});
                                for (size_t l = 0; l < srcShapes[j].size(); ++l)
                                    i32.CpuData()[l] = (int)srcShapes[j][l];
                            }
                            else
                                assert(0);
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::cout << "Input layer '" << srcNames[i] << "' is not found!" << std::endl;
                        return false;
                    }
                }            
            }
            else
            {
                for (size_t i = 0; i < _input.size(); ++i)
                    _input[i].layer->Reshape(_input[i].src, _input[i].buf, _input[i].dst);
            }

            ReshapeStages();

            if (dstNames.size())
            {
                _dst.clear();
                for (size_t i = 0; i < dstNames.size(); ++i)
                {
                    bool found = false;
                    for (size_t j = 0; j < _stages.size(); ++j)
                    {
                        const LayerParam & param = _stages[j].layer->Param();
                        if (param.name() == dstNames[i])
                        {
                            _dst.push_back(_stages[j].dst[0]);
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::cout << "Output layer '" << dstNames[i] << "' is not found!" << std::endl;
                        return false;
                    }
                }
            }

            return true;
        }

        bool Reshape(size_t width, size_t height, size_t batch = 1)
        {
            if (_input.size() != 1)
                return false;
            const LayerParam & param = _input[0].layer->Param();
            if (param.type() != LayerTypeInput || param.input().shape().size() != 1)
                return false;
            const TensorFormat & format = param.input().shape()[0].format();
            Shape shape = param.input().shape()[0].dim();
            if (shape.size() != 4)
                return false;
            shape[0] = batch;
            if (format == TensorFormatNchw)
            {
                if ((shape[2] == -1 && shape[3] == -1) || Resizable())
                {
                    shape[2] = height;
                    shape[3] = width;
                }
                else if (shape[2] != height || shape[3] != width)
                    return false;
            }
            else if (format == TensorFormatNhwc)
            {
                if ((shape[1] == -1 && shape[2] == -1) || Resizable())
                {
                    shape[1] = height;
                    shape[2] = width;
                }
                else if (shape[1] != height || shape[2] != width)
                    return false;
            }
            else
                return false;
            _input[0].dst[0]->Reshape(shape, Type(0), format);
            ReshapeStages();
            return true;
        }

        bool Resizable() const
        {
            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                const LayerParam & layer = _param().layers()[i];
                if (layer.type() == LayerTypeInnerProduct)
                    return false;
            }
            return true;
        }

        Shape NchwShape() const 
        {
            assert(_src.size() == 1 && _src[0]->Count() == 4);
            Shape shape = _src[0]->Shape();
            if (_src[0]->Format() == TensorFormatNhwc)
                shape = Shape({ shape[0], shape[3] , shape[1] , shape[2] });
            return shape;
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        bool SetInput(const View & view, float lower, float upper)
        {
            return Synet::SetInput(*this, Views({ view }), Floats({ lower }), Floats({ upper }));
        }

        bool SetInput(const View & view, const Floats & lower, const Floats & upper)
        {
            return Synet::SetInput(*this, Views({ view }), lower, upper);
        }

        bool SetInput(const Views & views, float lower, float upper)
        {
            return Synet::SetInput(*this, views, Floats({ lower }), Floats({ upper }));
        }

        bool SetInput(const Views & views, const Floats & lower, const Floats & upper)
        {
            return Synet::SetInput(*this, views, lower, upper);
        }
#endif

        bool GetMetaConst(const String & name, Tensor & value) const
        {
            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                const LayerParam & layer = _param().layers()[i];
                if (layer.name() == name && layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst)
                {
                    value.Import(layer.meta().alpha());
                    return true;
                }
            }
            return false;
        }

        TensorFormat Format() const
        {
            for (size_t i = 0; i < _input.size(); ++i)
            {
                const LayerParam & param = _input[i].layer->Param();
                if (param.type() == LayerTypeInput && param.input().shape().size())
                    return param.input().shape()[0].format();
            }
            assert(0);
            return TensorFormatUnknown;
        }

        void Forward()
        {
            //SYNET_PERF_FUNC();
            bool mode = GetFastMode();
            SetFastMode(true);
            for (size_t i = 0; i < _stages.size(); ++i)
            {
#if 0
                std::cout << _stages[i].layer->Param().name() << " : { ";
                const Shape & shape = _stages[i].src[0]->Shape();
                for (size_t j = 0; j < shape.size(); ++j)
                    std::cout << shape[j] << " ";
                std::cout << "}" << std::endl;
#endif
                _stages[i].layer->Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
            }
            SetFastMode(mode);
        }

        void UpdateStatistics(float quantile, float epsilon)
        {
            SYNET_PERF_FUNC();
            for (size_t i = 0; i < _tensors.size(); ++i)
                 UpdateStatistics(*_tensors[i], quantile, epsilon);
        }

        void DebugPrint(std::ostream & os, int flag, int first, int last, int precision)
        {
            bool printOutput = (flag & (1 << DebugPrintOutput)) != 0;
            bool printLayerDst = (flag & (1 << DebugPrintLayerDst)) != 0;
            bool printLayerWeight = (flag & (1 << DebugPrintLayerWeight)) != 0;
            bool printInt8Buffers = (flag & (1 << DebugPrintInt8Buffers)) != 0;
            bool printLayerInternal = (flag & (1 << DebugPrintLayerInternal)) != 0;
            for (size_t i = 0; i < _input.size() && printLayerDst; ++i)
            {
                os << "Layer: " << _input[i].layer->Param().name() << " : ";
                os << ValueToString(_input[i].layer->Param().type()) << " ( ";
                for (size_t j = 0; j < _input[i].layer->Param().src().size(); ++j)
                    os << _input[i].layer->Param().src()[j] << " ";
                os << ")." << std::endl;
                for (size_t j = 0; j < _input[i].dst.size(); ++j)
                    _input[i].dst[j]->DebugPrint(os, String("dst[") + ValueToString(j) + "]", false, first, last, precision);
            }
            for (size_t i = 0; i < _stages.size(); ++i)
            {
                Layer & layer = *_stages[i].layer;
                if ((layer._isBack && printOutput) || printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
                {
                    if(printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
                        layer.Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
                    os << "Layer: " << layer.Param().name() << " : ";
                    os << ValueToString(layer.Param().type()) << " ( ";
                    for (size_t j = 0; j < layer.Param().src().size(); ++j)
                        os << layer.Param().src()[j] << " ";
                    os << ")." << std::endl;
                    if (printLayerWeight)
                    {
                        for (size_t j = 0; j < layer.Weight().size(); ++j)
                            layer.Weight()[j].DebugPrint(os, String("weight[") + ValueToString(j) + "]", true, first, last, precision);
                    }
                    if (printInt8Buffers && layer.Is8i())
                    {
                        const Tensor& src = *_stages[i].src[0];
                        if (src.GetType() == TensorType32f)
                            _stages[i].buf[TensorType8u * BUFFER_COUNT + 1]->As8u().DebugPrint(os, src.Shape(), src.Format(), String("src"), false, first, last, precision);
                        const Tensor& dst = *_stages[i].dst[0];
                        _stages[i].buf[TensorType32i * BUFFER_COUNT + 0]->As32i().DebugPrint(os, dst.Shape(), dst.Format(), String("sum"), false, first, last, precision);
                    }
                    if (printLayerInternal)
                    {
                        layer.DebugPrint(os, flag, first, last, precision);
                    }
                    if ((layer._isBack && printOutput) || printLayerDst)
                    {
                        for (size_t j = 0; j < _stages[i].dst.size(); ++j)
                            _stages[i].dst[j]->DebugPrint(os, String("dst[") + ValueToString(j) + "]", false, first, last, precision);
                    }
                }
            }
        }

        Regions GetRegions(size_t imageW, size_t imageH, Type threshold, Type overlap) const
        {
            size_t netW = _src[0]->Axis(-1);
            size_t netH = _src[0]->Axis(-2);
            Regions regions;
            for (size_t i = 0; i < _dst.size(); ++i)
            {
                TensorPtrs dst(1, _dst[i]);
                const Layer * layer = _back[i];
                Regions candidats;
                if (layer->Param().type() == Synet::LayerTypeYolo)
                    ((YoloLayer<float>*)layer)->GetRegions(dst, netW, netH, threshold, candidats);
                if (layer->Param().type() == Synet::LayerTypeRegion)
                    ((RegionLayer<float>*)layer)->GetRegions(dst, threshold, candidats);
                if (layer->Param().type() == Synet::LayerTypeDetectionOutput)
                    ((DetectionOutputLayer<float>*)layer)->GetRegions(dst, threshold, candidats);
                for (size_t j = 0; j < candidats.size(); ++j)
                {
                    Region & c = candidats[j];
                    c.x *= imageW;
                    c.w *= imageW;
                    c.y *= imageH;
                    c.h *= imageH;
                    bool insert = true;
                    for (size_t k = 0; k < regions.size(); ++k)
                    {
                        Region & r = regions[k];
                        if (c.id == r.id && RelativeIntersection(c, r) >= overlap)
                        {
                            if (c.prob > r.prob)
                               r = c;
                            insert = false;
                            break;
                        }
                    }
                    if (insert)
                        regions.push_back(c);
                }
            }
            std::sort(regions.begin(), regions.end(), [](const Region & a, const Region & b) {return a.prob > b.prob; });
            return regions;
        }

        size_t MemoryUsage() const
        {
            std::set<const void*> unique;
            size_t memoryUsage = 0;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                for (size_t j = 0; j < _layers[i]->Weight().size(); ++j)
                {
                    if (_layers[i]->Weight()[j].Size() == 0)
                        continue;
                    const void * ptr = _layers[i]->Weight()[j].RawCpuData();
                    if (unique.find(ptr) == unique.end())
                    {
                        memoryUsage += _layers[i]->Weight()[j].MemoryUsage();
                        unique.insert(ptr);
                    }
                }
                memoryUsage += _layers[i]->MemoryUsage();
            }
            for (size_t i = 0; i < _tensors.size(); ++i)
            {
                const void * ptr = _tensors[i]->RawCpuData();
                if (unique.find(ptr) == unique.end())
                {
                    memoryUsage += _tensors[i]->MemoryUsage();
                    unique.insert(ptr);
                }
            }
            for (size_t i = 0; i < _stats.size(); ++i)
                memoryUsage += _stats[i]->MemoryUsage();
            return memoryUsage;
        }

        void CompactWeight()
        {
            for (size_t i = 0; i < _layers.size(); ++i)
                _layers[i]->CompactWeight();
        }

        int64_t Flop() const
        {
            int64_t flop = 0;
            for (size_t i = 0; i < _layers.size(); ++i)
                flop += _layers[i]->Flop();
            return flop;
        }

        bool Is8i() const
        {
            return _param().quantization().method() != QuantizationMethodUnknown;
        }  

        const Tensor* GetInternalTensor(const String& name) const
        {
            NameIdMap::const_iterator it = _tensorId.find(name);
            if (it != _tensorId.end())
                return _tensors[it->second].get();
            return NULL;
        }

    private:
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        typedef std::vector<Tensor> Tensors;
        typedef std::shared_ptr<Tensor> TensorSharedPtr;
        typedef std::vector<TensorSharedPtr> TensorSharedPtrs;

        typedef std::map<String, size_t> NameIdMap;
        typedef std::map<size_t, String> IdNameMap;
        typedef std::set<String> NameSet;
        typedef std::set<size_t> IdSet;
        typedef std::map<String, IdSet> NameIdSetMap;

        struct Stage
        {
            Layer * layer;
            TensorPtrs src;
            TensorPtrs buf;
            TensorPtrs dst;
        };
        typedef std::vector<Stage> Stages;

        bool _empty;
        NetworkParamHolder _param;
        LayerSharedPtrs _layers;
        TensorSharedPtrs _tensors;
        StatSharedPtrs _stats;

        Stages _input, _stages;
        TensorPtrs _src, _dst;
        LayerPtrs _back;
        NameIdMap _tensorId, _layerId, _statId;
        NameIdSetMap _srcIds, _dstIds;

        bool Init()
        {
            TensorPtrs buf;
            SetBuffers(buf);
            SetStats();

            NameSet available;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                Stage stage;
                stage.layer = _layers[i].get();
                const LayerParam& param = stage.layer->Param();
                _layerId[param.name()] = i;
                for (size_t j = 0; j < param.src().size(); ++j)
                {
                    const String& name = param.src()[j];
                    if (_tensorId.find(name) != _tensorId.end())
                    {
                        stage.src.push_back(_tensors[_tensorId[name]].get());
                        if (available.find(name) != available.end())
                            available.erase(name);
                    }
                    else
                        assert(0);
                }
                for (size_t j = 0; j < param.dst().size(); ++j)
                {
                    const String& name = param.dst()[j];
                    if (j < param.src().size() && name == param.src()[j])
                    {
                        stage.dst.push_back(_tensors[_tensorId[name]].get());
                    }
                    else  if (_tensorId.find(name) != _tensorId.end())
                    {
                        assert(0);
                    }
                    else
                    {
                        TensorSharedPtr tensor(new Tensor());
                        tensor->SetName(name);
                        _tensorId[name] = _tensors.size();
                        _tensors.push_back(tensor);
                        stage.dst.push_back(tensor.get());
                    }
                    available.insert(name);
                    if (param.type() == LayerTypeInput || (param.type() == LayerTypeMeta && (param.meta().type() == MetaTypeInput || param.meta().type() == MetaTypeInputWithDefault)))
                    {
                        _src.push_back(_tensors.back().get());
                    }
                }
                stage.buf = buf;
                if (Is8i())
                    stage.layer->SetStats(_stats);
                if (param.type() == LayerTypeInput || (param.type() == LayerTypeMeta && param.meta().type() == MetaTypeInput))
                    _input.push_back(stage);
                else
                {
                    for (size_t j = 0; j < param.src().size(); ++j)
                        _srcIds[param.src()[j]].insert(_stages.size());
                    for (size_t j = 0; j < param.dst().size(); ++j)
                        _dstIds[param.dst()[j]].insert(_stages.size());
                    _stages.push_back(stage);
                }
            }
            IdNameMap sorted;
            for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
                sorted[_layerId[*it]] = *it;
            for (IdNameMap::const_iterator it = sorted.begin(); it != sorted.end(); ++it)
            {
                if (InsertDst(it->second))
                {
                    LayerPtr layer = _layers[_layerId[it->second]].get();
                    if (layer->Param().type() != LayerTypeMeta)
                    {
                        _dst.push_back(_tensors[_tensorId[it->second]].get());
                        layer->_isBack = true;
                        _back.push_back(layer);
                    }
                }
            }
            if (Is8i())
            {
                SetTensorTypes();
                UnifyStats();
            }
            if (!Dynamic())
                Reshape();
            _empty = false;
            return true;
        }

        bool Is8iInSubGraph(const Stage & stage)
        {
            const Layer & layer = *stage.layer;
            if (layer._isBack)
                return false;
            const LayerParam & param = layer.Param();
            for (size_t d = 0; d < param.dst().size(); ++d)
            {
                const String & name = param.dst()[d];
                const IdSet & ids = _srcIds[name];
                for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
                {
                    const Stage & dst = _stages[*id];
                    if (&dst == &stage)
                        continue;
                    if (dst.layer->Is8i())
                        continue;
                    if (dst.layer->Can8i() && Is8iInSubGraph(dst))
                        continue;
                    return false;
                }
            }
            return true;
        }

        void Set8iInSubGraph(const Stage & stage)
        {
            const LayerParam & param = stage.layer->Param();
            for (size_t d = 0; d < param.dst().size(); ++d)
            {
                const String & name = param.dst()[d];
                _tensors[_tensorId[name]]->SetType(TensorType8u);
                const IdSet & ids = _srcIds[name];
                for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
                {
                    const Stage & dst = _stages[*id];
                    if (&dst == &stage)
                        continue;
                    if (dst.layer->Is8i())
                        continue;
                    Set8iInSubGraph(dst);
                }
            }
        }

        void SetTensorTypes()
        {
            for (size_t t = 0; t < _tensors.size(); ++t)
                if (_tensors[t]->GetType() == TensorTypeUnknown)
                    _tensors[t]->SetType(TensorType32f);
            for (size_t s = 0; s < _stages.size(); ++s)
            {
                const Layer & layer = *_stages[s].layer;
                if (!layer.Is8i())
                    continue;
                if (Is8iInSubGraph(_stages[s]))
                    Set8iInSubGraph(_stages[s]);
            }
        }

        bool IsSubGraphEndConv(size_t s)
        {
            const Stage& stage = _stages[s];
            const Layer & layer = *stage.layer;
            if (layer._isBack)
                return false;
            const LayerParam & param = layer.Param();
            for (size_t d = 0; d < param.dst().size(); ++d)
            {
                const String & name = param.dst()[d];
                const IdSet & ids = _srcIds[name];
                for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
                {
                    if (*id <= s)
                        continue;
                    const Stage & dst = _stages[*id];
                    const LayerParam & param = dst.layer->Param();
                    if (param.type() == LayerTypeConvolution && param.convolution().group() != param.convolution().outputNum())
                        continue;
                    if (param.type() == LayerTypeMergedConvolution)
                        continue;
                    if (IsSubGraphEndConv(*id))
                        continue;
                    return false;
                }
            }
            return true;
        }

        void UnifyStats()
        {
            if (_param().quantization().method() >= QuantizationMethodSymmetricNarrowed)
                return;
            for (size_t i = 0; i < _input.size(); ++i)
                _stats[_statId[_input[i].layer->Param().name()]]->Unify();
            for (size_t s = 0; s < _stages.size(); ++s)
            {
                if (IsSubGraphEndConv(s))
                {
                    const LayerParam & param = _stages[s].layer->Param();
                    if (param.type() == LayerTypeConvolution || param.type() == LayerTypeMergedConvolution || param.type() == LayerTypeScale)
                        _stats[_statId[param.dst()[0]]]->Unify();

                    if (param.type() == LayerTypePooling && param.pooling().method() == PoolingMethodTypeMax && !_stats[_statId[param.src()[0]]]->channels)
                        _stats[_statId[param.dst()[0]]]->UnifyAs(*_stats[_statId[param.src()[0]]]);
                    if (param.type() == LayerTypeRelu && param.relu().negativeSlope() == 0.0f)
                        _stats[_statId[param.dst()[0]]]->UnifyAs(*_stats[_statId[param.src()[0]]]);
                    if (param.type() == LayerTypeConcat)
                    {
                        StatPtrs stats;
                        for (size_t c = 0; c < param.src().size(); ++c)
                            stats.push_back(_stats[_statId[param.src()[c]]].get());
                        _stats[_statId[param.dst()[0]]]->UnifyAs(stats.data(), stats.size());
                    }
                }
            }
        }

        void ReshapeStages()
        {
            for (size_t i = 0; i < _stages.size(); ++i)
            {
                _stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst);
                if (_stages[i].layer->_isBack)
                    _stages[i].dst[0]->SetName(_stages[i].layer->Param().name());
            }
        }

        void SetBuffers(TensorPtrs & buf)
        {
            for (TensorType type = TensorType32f; type <= TensorType8u; type = TensorType((int)type + 1))
            {
                for (int i = 0; i < BUFFER_COUNT; ++i)
                {
                    TensorSharedPtr tensor(new Tensor());
                    tensor->SetType(type);
                    _tensors.push_back(tensor);
                    buf.push_back(tensor.get());
                }
            }
        }

        void SetStats()
        {
            for (size_t i = 0; i < _param().quantization().statistics().size(); ++i)
            {
                const StatisticParam & src = _param().quantization().statistics()[i];
                StatSharedPtr stat(new Stat(src));
                _statId[src.name()] = _stats.size();
                _stats.push_back(stat);
            }
        }

        void UpdateStatistics(const Tensor & tensor, float quantile, float epsilon)
        {
            if (tensor.Name().empty() || tensor.GetType() != TensorType32f)
                return;
            size_t channels = GetChannels(tensor);
            if (channels == 0)
                return;
            size_t index = 0;
            for (; index < _param().quantization().statistics().size(); ++index)
                if (_param().quantization().statistics()[index].name() == tensor.Name())
                    break;
            if (index == _param().quantization().statistics().size())
                _param().quantization().statistics().push_back(StatisticParam());
            StatisticParam & stat = _param().quantization().statistics()[index];
            if(stat.name().empty())
                stat.name() = tensor.Name();
            if (stat.min().empty())
                stat.min().resize(channels, FLT_MAX);
            if (stat.max().empty())
                stat.max().resize(channels, -FLT_MAX);
            UpdateChannelsQuantile(tensor, quantile, epsilon, stat.min().data(), stat.max().data());
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

        bool Dynamic()
        {
            for (size_t i = 0; i < _param().layers().size(); ++i)
            {
                const LayerParam & layer = _param().layers()[i];
                if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeInput)
                    return true;
                if (layer.type() == LayerTypeInput)
                {
                    if (layer.input().shape().empty())
                        return true;
                    for (size_t j = 0; j < layer.input().shape().size(); ++j)
                        for (size_t k = 0; k < layer.input().shape()[j].dim().size(); ++k)
                            if (layer.input().shape()[j].dim()[k] == (size_t)-1)
                                return true;
                }
            }
            return false;
        }

        static LayerPtr Create(const LayerParam & param, QuantizationMethod method)
        {
            switch (param.type())
            {
            case LayerTypeAdd: return new AddLayer<T>(param, method);
            case LayerTypeBatchNorm: return new BatchNormLayer<T>(param);
            case LayerTypeBias: return new BiasLayer<T>(param);
            case LayerTypeBinaryOperation: return new BinaryOperationLayer<T>(param);
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
                //if (param.mergedConvolution().conv()[0].quantizationLevel() == TensorType8i)
                //    return new MergedConvolution8iLayer<T>(param, method);
                //else
                    return new MergedConvolution32fLayer<T>(param);
            case LayerTypeMeta: return new MetaLayer<T>(param);
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
            case LayerTypeUnaryOperation: return new UnaryOperationLayer<T>(param);
            case LayerTypeUnpack: return new UnpackLayer<T>(param);
            case LayerTypeUpsample: return new UpsampleLayer<T>(param);
            case LayerTypeYolo: return new YoloLayer<T>(param);
            default:
                return NULL;
            }
        }

        static SYNET_INLINE Type Overlap(Type x1, Type w1, Type x2, Type w2)
        {
            Type l1 = x1 - w1 / 2;
            Type l2 = x2 - w2 / 2;
            Type left = l1 > l2 ? l1 : l2;
            Type r1 = x1 + w1 / 2;
            Type r2 = x2 + w2 / 2;
            Type right = r1 < r2 ? r1 : r2;
            return right - left;
        }

        static SYNET_INLINE Type Intersection(const Region & a, const Region & b)
        {
            Type w = Overlap(a.x, a.w, b.x, b.w);
            Type h = Overlap(a.y, a.h, b.y, b.h);
            return (w < 0 || h < 0) ? 0 : w*h;
        }

        static SYNET_INLINE Type Union(const Region & a, const Region & b)
        {
            Type i = Intersection(a, b);
            return a.w*a.h + b.w*b.h - i;
        }

        static SYNET_INLINE Type RelativeIntersection(const Region & a, const Region & b)
        {
            return Intersection(a, b) / Union(a, b);
        }
    };
}
