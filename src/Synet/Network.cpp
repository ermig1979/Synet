/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar,
*               2018-2021 Antonenka Mikhail.
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

#include "Synet/Fabric.h"
#include "Synet/Network.h"

#include "Synet/Utils/SetInput.h"
#include "Synet/Utils/Statistics.h"

#include "Synet/Layers/Legacy/RegionLayer.h"
#include "Synet/Layers/DetectionOutputLayer.h"
#include "Synet/Layers/YoloLayer.h"

namespace Synet
{
    Network::Network()
        : _empty(true)
    {
    }

    Network::~Network()
    {
        Clear();
    }

    bool Network::Empty() const
    { 
        return _empty; 
    }

    const NetworkParam & Network::Param() const
    { 
        return _param(); 
    }

    void Network::Clear()
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

    bool Network::Load(const String & model, const String & weight, const Options & options)
    {
        Clear();

        if (!_param.Load(model))
            SYNET_ERROR("Can't load model file '" << model << "' !");

        _context.options = options;
        if (!CreateLayers())
            return false;

        std::ifstream ifs(weight.c_str(), std::ifstream::binary);
        if (!ifs.is_open())
            SYNET_ERROR("Can't open weight file '" << weight << "' !");
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(ifs, _layers))
            {
                ifs.close();
                SYNET_ERROR("Can't load weight from file '" << weight << "' !");
            }
        }
        ifs.close();

        return Init();
    }

    bool Network::Load(const char * modelData, size_t modelSize, const char * weightData, size_t weightSize, const Options& options)
    {
        Clear();

        if (!_param.Load(modelData, modelSize, Cpl::ParamFormatXml))
            SYNET_ERROR("Can't load model from memory!");

        _context.options = options;

        if (!CreateLayers())
            return false;

        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(weightData, weightSize, _layers))
                return false;
        }

        return Init();
    }

    bool Network::Save(const String& model) const
    {
        return _param.Save(model, false);
    }

    Network::TensorPtrs & Network::Src()
    { 
        return _src; 
    }

    const Network::TensorPtrs & Network::Src() const
    {
        return _src;
    }

    const Network::TensorPtrs & Network::Dst() const
    { 
        return _dst; 
    }

    const Network::Tensor * Network::Dst(const String & name) const
    {
        for (size_t i = 0; i < _dst.size(); ++i)
            if (_dst[i]->Name() == name)
                return _dst[i];
        return NULL;
    }

    Network::LayerPtrs Network::Back() const
    {
        return _back;
    }

    bool Network::Reshape(const Strings & srcNames, const Shapes & srcShapes, const Strings & dstNames)
    {
        if (srcNames.size() != srcShapes.size())
            SYNET_ERROR("srcNames.size() != srcShapes.size() !");

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
                            _input[j].dst[0]->Reshape(param.input().shape()[0].type(), srcShapes[i], param.input().shape()[0].format());
                            _input[j].dst[0]->SetName(param.name());
                            _src.push_back(_input[j].dst[0]);
                        }
                        else
                            SYNET_ERROR("Unsupported type " << Cpl::ToStr(param.type()) << " of input layer " << param.name() << " !");
                        found = true;
                        break;
                    }
                }
                if (!found)
                    SYNET_ERROR("Input layer '" << srcNames[i] << "' is not found!");
            }            
        }
        else
        {
            for (size_t i = 0; i < _input.size(); ++i)
                _input[i].layer->Reshape(_input[i].src, _input[i].buf, _input[i].dst);
        }

        if (!ReshapeStages())
            return false;

        if (dstNames.size())
        {
            _dst.clear();
            _back.clear();
            for (size_t i = 0; i < dstNames.size(); ++i)
            {
                bool found = false;
                for (size_t j = 0; j < _stages.size(); ++j)
                {
                    Layer * layer = _stages[j].layer;
                    const LayerParam & param = layer->Param();
                    if (param.name() == dstNames[i])
                    {
                        _dst.push_back(_stages[j].dst[0]);
                        layer->_isBack = true;
                        _back.push_back(layer);
                        found = true;
                        break;
                    }
                }
                if (!found)
                    SYNET_ERROR("Output layer '" << dstNames[i] << "' is not found!");
            }
        }

        return true;
    }

    bool Network::Reshape(size_t width, size_t height, size_t batch)
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
        _input[0].dst[0]->Reshape(TensorType32f, shape, format, Type(0));
        _input[0].dst[0]->SetName(param.name());
        return ReshapeStages();
    }

    bool Network::SetBatch(size_t batch)
    {
        if (_input.size() != 1)
            return false;
        const LayerParam& param = _input[0].layer->Param();
        if (param.type() != LayerTypeInput || param.input().shape().size() != 1)
            return false;
        const TensorFormat& format = param.input().shape()[0].format();
        Shape shape = param.input().shape()[0].dim();
        if (shape.size() < 2)
            return false;
        shape[0] = batch;
        _input[0].dst[0]->Reshape(TensorType32f, shape, format, Type(0));
        _input[0].dst[0]->SetName(param.name());
        ReshapeStages();
        return true;
    }

    bool Network::Dynamic() const
    {
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            const LayerParam& layer = _param().layers()[i];
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

    bool Network::Resizable() const
    {
        for (size_t i = 0; i < _layers.size(); ++i)
            if (!_layers[i]->Resizable())
                return false;
        return true;
    }

    Shape Network::NchwShape() const
    {
        assert(_src.size() >= 1);
        Shape shape = _src[0]->Shape();
        if (_src[0]->Format() == TensorFormatNhwc && shape.size() == 4)
            shape = Shape({ shape[0], shape[3] , shape[1] , shape[2] });
        return shape;
    }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    bool Network::SetInput(const View & view, float lower, float upper, bool rgb)
    {
        return Synet::SetInput(*this, Views({ view }), Floats({ lower }), Floats({ upper }), rgb);
    }

    bool Network::SetInput(const View & view, const Floats & lower, const Floats & upper, bool rgb)
    {
        return Synet::SetInput(*this, Views({ view }), lower, upper, rgb);
    }

    bool Network::SetInput(const Views & views, float lower, float upper, bool rgb)
    {
        return Synet::SetInput(*this, views, Floats({ lower }), Floats({ upper }), rgb);
    }

    bool Network::SetInput(const Views & views, const Floats & lower, const Floats & upper, bool rgb)
    {
        return Synet::SetInput(*this, views, lower, upper, rgb);
    }
#endif

    bool Network::GetMetaConst(const String & name, Tensor & value) const
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

    TensorFormat Network::Format() const
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

    void Network::Forward()
    {
        //SYNET_PERF_FUNC();
        bool mode = GetFastMode();
        SetFastMode(true);
        for (size_t i = 0; i < _stages.size(); ++i)
        {
#if 0
            std::cout << _stages[i].layer->Param().name() << " : { ";
            if (_stages[i].src.size())
            {
                const Shape& shape = _stages[i].src[0]->Shape();
                for (size_t j = 0; j < shape.size(); ++j)
                    std::cout << shape[j] << " ";
            }
            std::cout << "}" << std::endl;
#endif
            _stages[i].layer->Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
        }
        SetFastMode(mode);
    }

    void Network::UpdateStatistics(float quantile, float epsilon)
    {
        SYNET_PERF_FUNC();
        for (size_t i = 0; i < _tensors.size(); ++i)
                UpdateStatistics(*_tensors[i], quantile, epsilon);
    }

    void Network::DebugPrint(std::ostream & os, int flag, int first, int last, int precision)
    {
        bool printOutput = (flag & (1 << DebugPrintOutput)) != 0;
        bool printLayerDst = (flag & (1 << DebugPrintLayerDst)) != 0;
        bool printLayerWeight = (flag & (1 << DebugPrintLayerWeight)) != 0;
        bool printInt8Buffers = (flag & (1 << DebugPrintInt8Buffers)) != 0;
        bool printLayerInternal = (flag & (1 << DebugPrintLayerInternal)) != 0;
        for (size_t i = 0; i < _input.size() && (printLayerDst || printOutput); ++i)
        {
            Layer & layer = *_input[i].layer;
            if ((layer._isBack && printOutput) || printLayerDst)
            {
                os << "Layer: " << layer.Param().name() << " : ";
                os << Cpl::ToStr(layer.Param().type()) << " ( ";
                for (size_t j = 0; j < layer.Param().src().size(); ++j)
                    os << layer.Param().src()[j] << " ";
                os << ")." << std::endl;
                for (size_t j = 0; j < _input[i].dst.size(); ++j)
                    _input[i].dst[j]->DebugPrint(os, String("dst[") + Cpl::ToStr(j) + "]", false, first, last, precision);
            }
        }
        for (size_t i = 0; i < _stages.size(); ++i)
        {
            Layer & layer = *_stages[i].layer;
            if ((layer._isBack && printOutput) || printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
            {
                if(printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
                    layer.Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
                os << "Layer: " << layer.Param().name() << " : ";
                os << Cpl::ToStr(layer.Param().type()) << " ( ";
                for (size_t j = 0; j < layer.Param().src().size(); ++j)
                    os << layer.Param().src()[j] << " ";
                os << ")." << std::endl;
                if (printLayerWeight)
                {
                    for (size_t j = 0; j < layer.Weight().size(); ++j)
                        layer.Weight()[j].DebugPrint(os, String("weight[") + Cpl::ToStr(j) + "]", true, first, last, precision);
                }
                if (printInt8Buffers && (layer.LowPrecision(TensorType8u) == LowPrecisionTypeActive))
                {
                    const Tensor& src = *_stages[i].src[0];
                    if (src.GetType() == TensorType32f)
                        _stages[i].buf[TensorType8u * BUFFER_COUNT + 1]->DebugPrint(os, src.Shape(), src.Format(), String("src"), false, first, last, precision);
                    const Tensor& dst = *_stages[i].dst[0];
                    _stages[i].buf[TensorType32i * BUFFER_COUNT + 0]->DebugPrint(os, dst.Shape(), dst.Format(), String("sum"), false, first, last, precision);
                }
                if (printLayerInternal)
                {
                    layer.DebugPrint(os, flag, first, last, precision);
                }
                if ((layer._isBack && printOutput) || printLayerDst)
                {
                    for (size_t j = 0; j < _stages[i].dst.size(); ++j)
                        _stages[i].dst[j]->DebugPrint(os, String("dst[") + Cpl::ToStr(j) + "]", false, first, last, precision);
                }
            }
        }
    }

    Network::Regions Network::GetRegions(size_t imageW, size_t imageH, Type threshold, Type overlap) const
    {
        const Shape & netNCHW = NchwShape();
        size_t netH = netNCHW[2];
        size_t netW = netNCHW[3];
        Regions regions;
        for (size_t i = 0; i < _dst.size(); ++i)
        {
            TensorPtrs dst(1, _dst[i]);
            const Layer * layer = _back[i];
            if (layer->Param().type() == Synet::LayerTypeStub && layer->Param().src().size() == 1)
            {
                for (size_t j = 0; j < _layers.size(); ++j)
                {
                    if (_layers[j]->Param().name() == layer->Param().src()[0])
                    {
                        layer = _layers[j].get();
                        break;
                    }
                }
            }
            Regions candidats;
            if (layer->Param().type() == Synet::LayerTypeYolo)
                ((YoloLayer*)layer)->GetRegions(dst, netW, netH, threshold, candidats);
            if (layer->Param().type() == Synet::LayerTypeRegion)
                ((RegionLayer*)layer)->GetRegions(dst, threshold, candidats);
            if (layer->Param().type() == Synet::LayerTypeDetectionOutput)
                ((DetectionOutputLayer*)layer)->GetRegions(dst, threshold, candidats);
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
                    if (c.id == r.id && Overlap(c, r) >= overlap)
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

    size_t Network::MemoryUsage() const
    {
        std::set<const void*> unique;
        size_t memoryUsage = 0;
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            for (size_t j = 0; j < _layers[i]->Weight().size(); ++j)
            {
                if (_layers[i]->Weight()[j].Size() == 0)
                    continue;
                const void * ptr = _layers[i]->Weight()[j].RawData();
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
            const void * ptr = _tensors[i]->RawData();
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

    void Network::CompactWeight()
    {
        for (size_t i = 0; i < _layers.size(); ++i)
            _layers[i]->CompactWeight();
    }

    int64_t Network::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if(_layers[i]->Param().parent().empty())
                flop += _layers[i]->Flop();
        }
        return flop;
    }

    bool Network::Is8i() const
    {
        return _param().quantization().method() != QuantizationMethodUnknown;
    }  

    bool Network::Is16b() const
    {
        bool has16b = false, enable = _context.options.BFloat16Enable();
        for (size_t i = 0; i < _param().layers().size() && !has16b && enable; ++i)
        {
            const LayerParam& layer = _param().layers()[i];
            if (layer.lowPrecision().bf16Type() == LowPrecisionTypeActive)
                has16b = true;
        }
        return has16b && enable;
    }

    const Network::Tensor* Network::GetInternalTensor(const String& name) const
    {
        NameIdMap::const_iterator it = _tensorId.find(name);
        if (it != _tensorId.end())
            return _tensors[it->second].get();
        return NULL;
    }

    bool Network::CreateLayers()
    {
        NameIdMap layerId;
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            const LayerParam& param = _param().layers()[i];
            LayerSharedPtr layer(Fabric::Create(param, &_context, _param().quantization().method()));
            if (layer)
            {
                layerId[param.name()] = _layers.size();
                _layers.push_back(layer);
                if (!param.parent().empty())
                {
                    if(layerId.find(param.parent()) == layerId.end())
                        SYNET_ERROR("Can't find parent layer: " << param.parent() << " of layer " << param.name() << " !");
                    _layers[layerId[param.parent()]]->AddChild(layer);
                }
            }
            else
                SYNET_ERROR("Can't create layer " << param.name() << " of type " << Cpl::ToStr(param.type()) << " !");
        }
        return true;
    }

    bool Network::Init()
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
            if (!param.parent().empty())
                continue;
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
                    SYNET_ERROR("Can't find tensor with name: " << name << " (See inputs of layer " << param.name() << ") !");
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
                    SYNET_ERROR("Output tensor with name: " << name << " is already exist (See outputs of layer " << param.name() << ") !");
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
                if (param.type() == LayerTypeInput)
                    _src.push_back(_tensors.back().get());
            }
            stage.buf = buf;
            if (Is8i())
                stage.layer->SetStats(_stats);
            if (param.type() == LayerTypeInput)
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
            sorted[_tensorId[*it]] = *it;
        for (IdNameMap::const_iterator it = sorted.begin(); it != sorted.end(); ++it)
        {
            if (InsertDst(it->second))
            {
                LayerPtr layer = NULL;
                if(_layerId.find(it->second) != _layerId.end())
                    layer = _layers[_layerId[it->second]].get();
                assert(layer);
                //if (_dstIds.find(it->second) != _dstIds.end())
                //    layer = _stages[*_dstIds[it->second].begin()].layer;
                if (layer->Param().type() != LayerTypeMeta)
                {
                    _dst.push_back(_tensors[_tensorId[it->second]].get());
                    layer->_isBack = true;
                    _back.push_back(layer);
                }
            }
        }
        SetTensorType32f();
        if (Is8i())
        {
            SetLowPrecisionTensorType(TensorType8u);
            UnifyStats();
        }
        if (Is16b())
        {
            SetLowPrecisionTensorType(TensorType16b);
        }
        if (!Dynamic())
        {
            if (!Reshape())
                return false;
        }
        _empty = false;
        return true;
    }

    void Network::SetTensorType32f()
    {
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            const LayerParam& param = _stages[s].layer->Param();
            if (param.type() == LayerTypeMeta)
                continue;
            for (size_t d = 0; d < param.dst().size(); ++d)
            {
                Tensor& tensor = *_tensors[_tensorId[param.dst()[d]]];
                if (tensor.GetType() == TensorTypeUnknown)
                    tensor.SetType(TensorType32f);
            }
        }
    }

    bool Network::CanIgnoreInSubGraph(TensorType type, const Layer* layer, bool fromDst) const
    {
        const LayerParam& param = layer->Param();
        if (layer->LowPrecision(type) == LowPrecisionTypeActive)
            return true;
        if (!fromDst && layer->LowPrecision(type) == LowPrecisionTypeHybrid)
            return true;
        if (param.type() == LayerTypePriorBox)
            return true;
        if (param.type() == LayerTypeMeta && param.meta().type() == MetaTypeShape)
            return true;
        if (param.type() == LayerTypeMeta && param.meta().type() == MetaTypePack)
            return true;
        return false;
    }

    bool Network::ParseSubGraph(TensorType type, const Layer* layer) const
    {
        return layer->LowPrecision(type) == LowPrecisionTypePassive;
    }

    bool Network::IsLowPrecisionInSubGraph(TensorType type, size_t current, IdSet& visited, bool back)
    {
        visited.insert(current);
        const Stage& stage = _stages[current];
        const Layer& layer = *stage.layer;
        if (layer._isBack)
            return false;
        const LayerParam& param = layer.Param();
        if (param.type() == LayerTypeInput)
            return false;
        for (size_t d = 0; d < param.dst().size(); ++d)
        {
            const String& name = param.dst()[d];
            const IdSet& ids = _srcIds[name];
            for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
            {
                if (visited.find(*id) != visited.end())
                    continue;
                const Stage& dst = _stages[*id];
                if (CanIgnoreInSubGraph(type, dst.layer, true))
                    continue;
                if (ParseSubGraph(type, dst.layer) && IsLowPrecisionInSubGraph(type, *id, visited, true))
                    continue;
                return false;
            }
        }
        if (back)
        {
            for (size_t s = 0; s < param.src().size(); ++s)
            {
                const String& name = param.src()[s];
                const IdSet& dstIds = _dstIds[name];
                for (IdSet::const_iterator id = dstIds.begin(); id != dstIds.end(); ++id)
                {
                    if (visited.find(*id) != visited.end())
                        continue;
                    const Stage& src = _stages[*id];
                    if (CanIgnoreInSubGraph(type, src.layer, false))
                        continue;
                    if (ParseSubGraph(type, src.layer) && IsLowPrecisionInSubGraph(type, *id, visited, true))
                        continue;
                    return false;
                }
                const IdSet& srcIds = _srcIds[name];
                for (IdSet::const_iterator id = srcIds.begin(); id != srcIds.end(); ++id)
                {
                    if (visited.find(*id) != visited.end())
                        continue;
                    const Stage& dst = _stages[*id];
                    if (CanIgnoreInSubGraph(type, dst.layer, true))
                        continue;
                    if (ParseSubGraph(type, dst.layer) && IsLowPrecisionInSubGraph(type, *id, visited, true))
                        continue;
                    return false;
                }
            }
        }
        return true;
    }

    void Network::SetLowPrecisionInSubGraph(TensorType type, size_t current, IdSet& visited, bool back)
    {
        visited.insert(current);
        const Stage& stage = _stages[current];
        const LayerParam& param = stage.layer->Param();
        for (size_t d = 0; d < param.dst().size(); ++d)
        {
            const String& name = param.dst()[d];
            _tensors[_tensorId[name]]->SetType(type);
            const IdSet& ids = _srcIds[name];
            for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
            {
                if (visited.find(*id) != visited.end())
                    continue;
                if (CanIgnoreInSubGraph(type, _stages[*id].layer, true))
                    continue;
                SetLowPrecisionInSubGraph(type, *id, visited, true);
            }
        }
        if (back)
        {
            for (size_t s = 0; s < param.src().size(); ++s)
            {
                const String& name = param.src()[s];
                _tensors[_tensorId[name]]->SetType(type);
                const IdSet& ids = _dstIds[name];
                for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
                {
                    if (visited.find(*id) != visited.end())
                        continue;
                    if (CanIgnoreInSubGraph(type, _stages[*id].layer, false))
                        continue;
                    SetLowPrecisionInSubGraph(type, *id, visited, true);
                }
            }
        }
    }

    void Network::SetLowPrecisionTensorType(TensorType type)
    {
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            const Layer& layer = *_stages[s].layer;
            if (layer.LowPrecision(type) == LowPrecisionTypeActive)
            {
                IdSet checked, setted;
                if (IsLowPrecisionInSubGraph(type, s, checked, false))
                    SetLowPrecisionInSubGraph(type, s, setted, false);
            }
        }
    }

    bool Network::IsSubGraphEndConv(size_t s)
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
                if (param.type() == LayerTypePriorBox)
                    continue;
                if (IsSubGraphEndConv(*id))
                    continue;
                return false;
            }
        }
        return true;
    }

    void Network::UnifyStats()
    {
        if (_param().quantization().method() == QuantizationMethodSymmetricNarrowed)
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

//#define SYNET_RESHAPE_LOG

    bool Network::ReshapeStages()
    {
        for (size_t i = 0; i < _stages.size(); ++i)
        {
            const LayerParam& param = _stages[i].layer->Param();
            if (!_stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst))
            {
                std::stringstream err;
                err << "Can't reshape " << i << " layer with name: " << Cpl::ToStr(param.name()) << ", type: " << Cpl::ToStr(param.type());
                for (size_t s = 0; s < param.src().size(); ++s)
                    err << ", src[" << s << "]: " << Cpl::ToStr(_stages[i].src[s]->GetType()) << " " << ToStr(_stages[i].src[s]->Shape());
                err << " !";
                SYNET_ERROR(err.str());
            }
            if (_stages[i].dst.size() && _stages[i].dst[0]->Size() == 0 && !_stages[i].layer->_isBack && _stages[i].dst[0]->Count() > 1)
            {
                std::stringstream err;
                err << "Reshape " << i << " layer with name: " << Cpl::ToStr(param.name()) << ", type: " << Cpl::ToStr(param.type());
                for (size_t s = 0; s < param.src().size(); ++s)
                    err << ", src[" << s << "]: " << Cpl::ToStr(_stages[i].src[s]->GetType()) << " " << ToStr(_stages[i].src[s]->Shape());
                err << " gets dst[0]: " << Cpl::ToStr(_stages[i].dst[0]->GetType()) << " " << ToStr(_stages[i].dst[0]->Shape()) << " !";
                SYNET_ERROR(err.str());
            }
#ifdef SYNET_RESHAPE_LOG
            {
                std::stringstream msg;
                msg << "Layer " << i << " with name: " << Cpl::ToStr(param.name()) << ", type: " << Cpl::ToStr(param.type());
                for (size_t s = 0; s < param.src().size(); ++s)
                    msg << ", src[" << s << "]: " << Cpl::ToStr(_stages[i].src[s]->GetType()) << " " << ToStr(_stages[i].src[s]->Shape());
                msg << " -> ";
                for (size_t d = 0; d < param.dst().size(); ++d)
                    msg << ", dst[" << d << "]: " << Cpl::ToStr(_stages[i].dst[d]->GetType()) << " " << ToStr(_stages[i].dst[d]->Shape());
                msg << ".";
                CPL_LOG_SS(Info, msg.str());
            }
#endif
            if (_stages[i].layer->_isBack && /*param.type() != LayerTypeStub &&*/ param.name().find("/sink_port") == String::npos)
                _stages[i].dst[0]->SetName(param.name());
        }
        return true;
    }

    void Network::SetBuffers(TensorPtrs & buf)
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

    void Network::SetStats()
    {
        for (size_t i = 0; i < _param().quantization().statistics().size(); ++i)
        {
            const StatisticParam & src = _param().quantization().statistics()[i];
            StatSharedPtr stat(new Stat(src));
            _statId[src.name()] = _stats.size();
            _stats.push_back(stat);
        }
    }

    void Network::UpdateStatistics(const Tensor & tensor, float quantile, float epsilon)
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

    bool Network::InsertDst(const String & name)
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
}
